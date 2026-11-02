//! Where FFmpeg reads from: a file, a Rust reader, or an HLS playlist whose
//! playlists, keys and segments Rust fetches. FFmpeg is built without network
//! protocols; every byte of a reader or an HLS stream comes from Rust.
//!
//! - A reader is wrapped as an `AVIOContext` (ffmpeg-next's `StreamIo`); its
//!   `Seek` answers FFmpeg's seeks, and `SeekFrom::End(0)` its size probe.
//! - HLS runs FFmpeg's `hls` and `mpegts` demuxers with a custom `io_open`:
//!   each URI the demuxer opens (the playlist, variant playlists, keys,
//!   segments) is fetched whole through the Rust callback and served from
//!   memory. AES-128 segments (`crypto:<uri>`, with the key and IV as
//!   options) are decrypted here with libavutil's AES.

use anyhow::{anyhow, bail, Context as _, Result};
use ffmpeg::ffi;
use ffmpeg::format::context::{Input, StreamIo};
use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CStr, CString};
use std::fmt;
use std::io::{self, Cursor};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

/// Opens a fresh reader over the same bytes; each playback thread opens its own.
pub(super) type OpenReader = Arc<dyn Fn() -> io::Result<Box<dyn super::ReadSeek>> + Send + Sync>;
/// Fetches one HLS URI (playlist, key or segment) whole.
pub(super) type Fetch = Arc<dyn Fn(&str) -> io::Result<Vec<u8>> + Send + Sync>;

#[derive(Clone)]
pub(super) enum Origin {
    File(PathBuf),
    Reader(OpenReader),
    Hls { playlist: String, fetch: Fetch },
}

impl fmt::Debug for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => write!(f, "{path:?}"),
            Self::Reader(_) => f.write_str("a Rust reader"),
            Self::Hls { playlist, .. } => write!(f, "HLS {playlist:?}"),
        }
    }
}

impl Origin {
    /// What an FFmpeg backend opens for `source`.
    pub fn of(source: super::Source) -> Result<Self> {
        match source {
            super::Source::File(path) => Ok(Self::File(path.to_owned())),
            super::Source::Ffmpeg(origin) => Ok(origin.clone()),
            // Built without network: streams come through a reader or HLS.
            super::Source::Url(url) => bail!("the FFmpeg backend can't open URL {url:?}; use open_reader or open_hls"),
        }
    }
}

/// An open input, with what its IO callbacks need (dropped after it).
pub(super) struct Opened {
    input: Input,
    _hls: Option<Box<Hls>>,
}

impl Deref for Opened {
    type Target = Input;
    fn deref(&self) -> &Input {
        &self.input
    }
}

impl DerefMut for Opened {
    fn deref_mut(&mut self) -> &mut Input {
        &mut self.input
    }
}

pub(super) fn open(origin: &Origin) -> Result<Opened> {
    super::ffmpeg::init();
    let (input, hls) = match origin {
        Origin::File(path) => (ffmpeg::format::input(path)?, None),
        Origin::Reader(open) => {
            let reader = open().context("opening the reader")?;
            (ffmpeg::format::input_from_stream(StreamIo::from_read_seek(reader)?, None, None)?, None)
        }
        Origin::Hls { playlist, fetch } => {
            let hls = Box::new(Hls {
                fetch: fetch.clone(),
                open: Mutex::new(HashMap::new()),
            });
            (open_hls(playlist, &hls)?, Some(hls))
        }
    };
    Ok(Opened { input, _hls: hls })
}

/// The `io_open` state: the fetch callback and the contexts it handed out.
struct Hls {
    fetch: Fetch,
    open: Mutex<HashMap<usize, StreamIo>>,
}

fn open_hls(playlist: &str, hls: &Hls) -> Result<Input> {
    let url = CString::new(playlist)?;
    // SAFETY: the context is set up before it is opened; `hls` outlives it
    // (`Opened` drops the input first). On failure FFmpeg frees the context.
    unsafe {
        let mut context = ffi::avformat_alloc_context();
        if context.is_null() {
            bail!("avformat_alloc_context failed");
        }
        (*context).opaque = hls as *const Hls as *mut _;
        (*context).io_open = Some(io_open);
        (*context).io_close2 = Some(io_close);
        let format = ffi::av_find_input_format(c"hls".as_ptr());
        if format.is_null() {
            ffi::avformat_free_context(context);
            bail!("FFmpeg built without the hls demuxer");
        }
        let mut options = std::ptr::null_mut();
        // Rust decides what each URI is; FFmpeg's extension guards are for
        // files and network URLs.
        for (key, value) in [
            (c"allowed_extensions", c"ALL"),
            (c"allowed_segment_extensions", c"ALL"),
            (c"extension_picky", c"0"),
        ] {
            ffi::av_dict_set(&mut options, key.as_ptr(), value.as_ptr(), 0);
        }
        let ret = ffi::avformat_open_input(&mut context, url.as_ptr(), format, &mut options);
        ffi::av_dict_free(&mut options);
        if ret < 0 {
            bail!("FFmpeg opening HLS {playlist:?}: {}", ffmpeg::Error::from(ret));
        }
        let input = Input::wrap(context);
        let ret = ffi::avformat_find_stream_info(context, std::ptr::null_mut());
        if ret < 0 {
            bail!("reading HLS {playlist:?}: {}", ffmpeg::Error::from(ret));
        }
        Ok(input)
    }
}

unsafe extern "C" fn io_open(
    context: *mut ffi::AVFormatContext,
    pb: *mut *mut ffi::AVIOContext,
    url: *const c_char,
    _flags: c_int,
    options: *mut *mut ffi::AVDictionary,
) -> c_int {
    // SAFETY: FFmpeg passes its context (whose opaque is our `Hls`), a URL
    // string and its options.
    let (hls, url, option) = unsafe {
        let options = if options.is_null() { std::ptr::null_mut() } else { *options };
        let option = move |key: &CStr| {
            let entry = ffi::av_dict_get(options, key.as_ptr(), std::ptr::null(), 0);
            (!entry.is_null()).then(|| CStr::from_ptr((*entry).value).to_string_lossy().into_owned())
        };
        (&*((*context).opaque as *const Hls), CStr::from_ptr(url).to_string_lossy(), option)
    };
    let served = (|| -> Result<Vec<u8>> {
        // AES-128 segments come as `crypto:<uri>` or `crypto+<uri>`.
        if let Some(uri) = url.strip_prefix("crypto:").or_else(|| url.strip_prefix("crypto+")) {
            let key = option(c"key").ok_or_else(|| anyhow!("no key"))?;
            let iv = option(c"iv").ok_or_else(|| anyhow!("no IV"))?;
            let data = (hls.fetch)(uri).with_context(|| format!("fetching {uri:?}"))?;
            return decrypt(data, &hex16(&key)?, &hex16(&iv)?).with_context(|| format!("decrypting {uri:?}"));
        }
        (hls.fetch)(&url).with_context(|| format!("fetching {url:?}"))
    })();
    let data = match served {
        Ok(data) => data,
        Err(error) => {
            log::warn!("video: HLS {error:#}");
            return ffi::AVERROR(ffi::EIO);
        }
    };
    let mut io = match StreamIo::from_read_seek(Cursor::new(data)) {
        Ok(io) => io,
        Err(error) => return error.into(),
    };
    let raw = io.as_mut_ptr();
    hls.open.lock().unwrap_or_else(PoisonError::into_inner).insert(raw as usize, io);
    // SAFETY: FFmpeg gave a place for the context; it stays ours until `io_close`.
    unsafe { *pb = raw };
    0
}

unsafe extern "C" fn io_close(context: *mut ffi::AVFormatContext, pb: *mut ffi::AVIOContext) -> c_int {
    // SAFETY: as in `io_open`.
    let hls = unsafe { &*((*context).opaque as *const Hls) };
    hls.open.lock().unwrap_or_else(PoisonError::into_inner).remove(&(pb as usize));
    0
}

fn hex16(hex: &str) -> Result<[u8; 16]> {
    let hex = hex.trim_start_matches("0x").trim_start_matches("0X");
    if hex.len() != 32 {
        bail!("not 16 hex bytes: {hex:?}");
    }
    let mut bytes = [0; 16];
    for (byte, pair) in bytes.iter_mut().zip(hex.as_bytes().chunks(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(bytes)
}

/// AES-128-CBC with PKCS#7 padding, as HLS encrypts segments.
fn decrypt(mut data: Vec<u8>, key: &[u8; 16], iv: &[u8; 16]) -> Result<Vec<u8>> {
    if data.is_empty() || data.len() % 16 != 0 {
        bail!("{} bytes is not whole AES blocks", data.len());
    }
    let mut iv = *iv;
    // SAFETY: the context is freed below; `data` holds whole blocks and is
    // decrypted in place, which av_aes_crypt allows.
    unsafe {
        let aes = ffi::av_aes_alloc();
        if aes.is_null() {
            bail!("av_aes_alloc failed");
        }
        ffi::av_aes_init(aes, key.as_ptr(), 128, 1);
        ffi::av_aes_crypt(aes, data.as_mut_ptr(), data.as_ptr(), (data.len() / 16) as c_int, iv.as_mut_ptr(), 1);
        ffi::av_free(aes.cast());
    }
    let pad = *data.last().unwrap_or(&0) as usize;
    if pad == 0 || pad > 16 || data[data.len() - pad..].iter().any(|&b| b as usize != pad) {
        bail!("bad padding (wrong key?)");
    }
    data.truncate(data.len() - pad);
    Ok(data)
}

#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod tests {
    use super::*;
    use crate::media::ffmpeg::{probe, FfmpegAudio, FfmpegVideo};
    use crate::media::player::{AudioOutput, AudioSource, VideoOutput, VideoSource};
    use std::path::Path;

    /// The example's clips; `GPUI_MEDIA_ASSETS` points elsewhere (on a phone).
    fn assets() -> PathBuf {
        std::env::var_os("GPUI_MEDIA_ASSETS")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("example/assets/video"))
    }

    fn reader(path: PathBuf) -> Origin {
        Origin::Reader(Arc::new(move || {
            Ok(Box::new(std::fs::File::open(&path)?) as Box<dyn crate::media::ReadSeek>)
        }))
    }

    fn hls() -> Origin {
        let dir = assets().join("hls");
        Origin::Hls {
            playlist: "hls/clip.m3u8".into(),
            fetch: Arc::new(move |uri: &str| {
                let name = uri.strip_prefix("hls/").ok_or(io::ErrorKind::NotFound)?;
                std::fs::read(dir.join(name))
            }),
        }
    }

    /// The timestamp of the next decoded video frame, in microseconds.
    fn next_frame(video: &mut FfmpegVideo, slot: &Mutex<Option<super::super::planes::Picture>>) -> i64 {
        loop {
            match video.next().unwrap() {
                VideoOutput::Frame(pts, frame) => {
                    video.release(frame, true).unwrap();
                    assert!(slot.lock().unwrap().take().is_some());
                    return pts;
                }
                VideoOutput::Pending => {}
                VideoOutput::End => panic!("no frame"),
            }
        }
    }

    /// Decodes, seeks to 4 s and decodes again, in software.
    fn plays_and_seeks(origin: Origin, duration_s: f64) {
        let info = probe(&origin).unwrap();
        assert!((info.duration_us as f64 / 1e6 - duration_s).abs() < 0.2, "duration {}", info.duration_us);
        assert!(info.has_audio);
        let slot = Arc::new(Mutex::new(None));
        let (mut video, hardware) = FfmpegVideo::open(&origin, None, slot.clone()).unwrap();
        assert!(!hardware);
        assert!(next_frame(&mut video, &slot) < 100_000);
        video.seek(4_000_000).unwrap();
        // A keyframe at or before 4 s: one a second in the file, a segment
        // start (every 2 s) in HLS. Decoding then reaches 4 s.
        let pts = next_frame(&mut video, &slot);
        assert!((1_900_000..=4_000_000).contains(&pts), "seeked to {pts} µs");
        let mut last = pts;
        for _ in 0..70 {
            last = next_frame(&mut video, &slot);
            if last >= 4_000_000 {
                break;
            }
        }
        assert!((4_000_000..4_100_000).contains(&last), "decoded on to {last} µs");

        let mut audio = FfmpegAudio::open(&origin).unwrap();
        audio.seek(4_000_000).unwrap();
        let pts = loop {
            match audio.next().unwrap() {
                AudioOutput::Samples { pts, frames, .. } if frames > 0 => break pts,
                AudioOutput::End => panic!("no audio"),
                _ => {}
            }
        };
        assert!((1_900_000..=4_100_000).contains(&pts), "audio seeked to {pts} µs");
    }

    #[test]
    fn a_reader_plays_and_seeks() {
        plays_and_seeks(reader(assets().join("av-60s-640x360.mp4")), 60.0);
    }

    #[test]
    fn encrypted_hls_plays_and_seeks_through_callbacks() {
        plays_and_seeks(hls(), 6.0);
    }

    #[test]
    fn a_failed_fetch_is_an_error() {
        let origin = Origin::Hls {
            playlist: "missing.m3u8".into(),
            fetch: Arc::new(|_: &str| Err(io::ErrorKind::NotFound.into())),
        };
        assert!(probe(&origin).is_err());
    }

    #[test]
    fn decrypt_rejects_a_wrong_key() {
        let data = std::fs::read(assets().join("hls/clip0.ts")).unwrap();
        assert!(decrypt(data, &[0; 16], &[0; 16]).is_err());
    }
}
