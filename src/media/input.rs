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
//! - The probe and the playback threads each demux on their own, so their
//!   fetches go through one [`Shelf`]: each URI is fetched once, and the
//!   segments after the one asked for are fetched ahead.
//! - A master playlist is narrowed to one variant before FFmpeg sees it
//!   ([`hls`]): its `hls` demuxer reads every variant's segments otherwise.

use anyhow::{anyhow, bail, Context as _, Result};
use ffmpeg::ffi;
use ffmpeg::format::context::{Input, StreamIo};
use ffmpeg_next as ffmpeg;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_char, c_int, CStr, CString};
use std::fmt;
use std::io::{self, Cursor};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

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

/// Segments fetched ahead of the one a demuxer asks for: a stall between
/// segments otherwise starves the audio and stutters the picture.
const AHEAD: usize = 3;
/// Segments kept in memory: the ones ahead, and a few behind for the
/// playback thread that trails the other.
const KEPT: usize = AHEAD + 4;
/// Replies smaller than this (keys) are kept for good, like playlists.
const SMALL: usize = 4096;
/// The tallest variant picked from a master playlist: a phone's screen.
const MAX_HEIGHT: u32 = 1080;

/// HLS for `playlist`: its fetches shared ([`Shelf`]), and a master
/// playlist narrowed to one variant.
pub(super) fn hls(playlist: &str, fetch: Fetch) -> Origin {
    let shelf = Shelf::new(fetch);
    let playlist = match shelf.get(playlist) {
        Ok(text) => variant(&String::from_utf8_lossy(&text), playlist).unwrap_or_else(|| playlist.to_owned()),
        // FFmpeg's open says what went wrong.
        Err(_) => playlist.to_owned(),
    };
    let fetch: Fetch = Arc::new(move |uri: &str| shelf.get(uri));
    Origin::Hls { playlist, fetch }
}

/// The variant of master playlist `text` (at `base`) to play: the richest
/// up to [`MAX_HEIGHT`] tall, else the smallest. `None` for a media playlist,
/// or one whose audio comes as a rendition of its own, which needs the
/// master.
fn variant(text: &str, base: &str) -> Option<String> {
    if text.lines().any(|line| line.starts_with("#EXT-X-MEDIA:") && line.contains("TYPE=AUDIO") && line.contains("URI=")) {
        return None;
    }
    let attr = |line: &str, key: &str| {
        line.split(',').find_map(|pair| pair.trim().strip_prefix(key)?.strip_prefix('=').map(str::to_owned))
    };
    // (bandwidth, height, uri)
    let mut variants = Vec::new();
    let mut lines = text.lines().map(str::trim);
    while let Some(line) = lines.next() {
        let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") else { continue };
        let bandwidth: u64 = attr(attrs, "BANDWIDTH").and_then(|b| b.parse().ok()).unwrap_or(0);
        let height: Option<u32> = attr(attrs, "RESOLUTION").and_then(|r| r.split_once('x')?.1.parse().ok());
        if let Some(uri) = lines.by_ref().find(|l| !l.is_empty() && !l.starts_with('#')) {
            variants.push((bandwidth, height, resolve(base, uri)));
        }
    }
    let fits = variants.iter().filter(|v| v.1.is_none_or(|h| h <= MAX_HEIGHT)).max_by_key(|v| v.0);
    let picked = fits.or_else(|| variants.iter().min_by_key(|v| v.1))?;
    Some(picked.2.clone())
}

/// `uri` as FFmpeg resolves it against playlist `base`.
fn resolve(base: &str, uri: &str) -> String {
    if uri.starts_with('/') || uri.contains("://") {
        return uri.to_owned();
    }
    match base.rfind('/') {
        Some(slash) => format!("{}{uri}", &base[..=slash]),
        None => uri.to_owned(),
    }
}

/// The segment URIs of media playlist `text` (at `base`), in order.
fn segments(text: &str, base: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|uri| resolve(base, uri))
        .collect()
}

/// One HLS stream's fetches, shared by its probe and playback threads.
pub(super) struct Shelf {
    fetch: Fetch,
    state: Mutex<Shelved>,
    fetched: Condvar,
}

#[derive(Default)]
struct Shelved {
    /// Playlists and keys.
    small: HashMap<String, Vec<u8>>,
    /// The segments fetched last, oldest first.
    segments: VecDeque<(String, Vec<u8>)>,
    /// URIs being fetched now.
    fetching: HashSet<String>,
    /// The media playlist's segments, in order.
    order: Vec<String>,
}

impl Shelved {
    fn kept(&self, uri: &str) -> Option<&Vec<u8>> {
        self.small.get(uri).or_else(|| self.segments.iter().find(|(u, _)| u == uri).map(|(_, b)| b))
    }

    fn keep(&mut self, uri: &str, bytes: &[u8]) {
        if bytes.starts_with(b"#EXTM3U") {
            let text = String::from_utf8_lossy(bytes);
            // A master playlist lists playlists, not segments.
            if !text.contains("#EXT-X-STREAM-INF") {
                self.order = segments(&text, uri);
            }
            self.small.insert(uri.to_owned(), bytes.to_vec());
        } else if bytes.len() < SMALL {
            self.small.insert(uri.to_owned(), bytes.to_vec());
        } else {
            if self.segments.len() == KEPT {
                self.segments.pop_front();
            }
            self.segments.push_back((uri.to_owned(), bytes.to_vec()));
        }
    }
}

impl Shelf {
    fn new(fetch: Fetch) -> Arc<Self> {
        Arc::new(Self { fetch, state: Mutex::default(), fetched: Condvar::new() })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Shelved> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `uri`'s bytes: kept, fetched by another thread meanwhile, or fetched
    /// now; then the segments after it are fetched ahead.
    fn get(self: &Arc<Self>, uri: &str) -> io::Result<Vec<u8>> {
        let mut state = self.lock();
        let bytes = loop {
            if let Some(bytes) = state.kept(uri) {
                break bytes.clone();
            }
            if state.fetching.contains(uri) {
                state = self.fetched.wait(state).unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            state.fetching.insert(uri.to_owned());
            drop(state);
            let fetched = (self.fetch)(uri);
            state = self.lock();
            state.fetching.remove(uri);
            if let Ok(bytes) = &fetched {
                state.keep(uri, bytes);
            }
            self.fetched.notify_all();
            break fetched?;
        };
        // The next few, not kept or coming already.
        let ahead: Vec<String> = match state.order.iter().position(|u| u == uri) {
            Some(at) => state.order[at + 1..]
                .iter()
                .take(AHEAD)
                .filter(|u| state.kept(u).is_none() && !state.fetching.contains(*u))
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        state.fetching.extend(ahead.iter().cloned());
        drop(state);
        if !ahead.is_empty() {
            let shelf = self.clone();
            std::thread::spawn(move || {
                for uri in ahead {
                    let fetched = (shelf.fetch)(&uri);
                    let mut state = shelf.lock();
                    state.fetching.remove(&uri);
                    match fetched {
                        Ok(bytes) => state.keep(&uri, &bytes),
                        Err(error) => log::warn!("video: HLS fetching {uri:?} ahead: {error}"),
                    }
                    shelf.fetched.notify_all();
                }
            });
        }
        Ok(bytes)
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
    use crate::media::ffmpeg::{probe, Choices, FfmpegAudio, FfmpegVideo};
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
        assert!(info.audio_track.is_some());
        let slot = Arc::new(Mutex::new(None));
        let choices = Choices::new(&info);
        let (mut video, hardware) = FfmpegVideo::open(&origin, None, slot.clone(), choices.clone()).unwrap();
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

        assert!(choices.lock().unwrap().demuxed_us >= 4_000_000, "demuxed to 4 s");
        let mut audio = FfmpegAudio::open(&origin, choices).unwrap();
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
    fn hls_fetches_each_uri_once_and_segments_ahead() {
        let dir = assets().join("hls");
        let asked: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = asked.clone();
        let origin = super::hls(
            "hls/clip.m3u8",
            Arc::new(move |uri: &str| {
                log.lock().unwrap().push(uri.to_owned());
                std::fs::read(dir.join(uri.strip_prefix("hls/").ok_or(io::ErrorKind::NotFound)?))
            }),
        );
        let info = probe(&origin).unwrap();
        let choices = Choices::new(&info);
        let mut audio = FfmpegAudio::open(&origin, choices.clone()).unwrap();
        while !matches!(audio.next().unwrap(), AudioOutput::End) {
            audio.release().unwrap();
        }
        let asked = asked.lock().unwrap().clone();
        let mut once = asked.clone();
        once.sort();
        once.dedup();
        assert_eq!(asked.len(), once.len(), "fetched twice: {asked:?}");
        assert!(asked.contains(&"hls/clip2.ts".to_owned()));
    }

    #[test]
    fn a_master_playlist_plays_one_variant() {
        let master = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\nlow/index.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080\nhd/index.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=9000000,RESOLUTION=3840x2160\n/x/4k/index.m3u8\n";
        assert_eq!(variant(master, "v/master.m3u8").as_deref(), Some("v/hd/index.m3u8"));
        let tall = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=9000000,RESOLUTION=3840x2160\n/x/4k.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=7000000,RESOLUTION=2560x1440\n/x/2k.m3u8\n";
        assert_eq!(variant(tall, "/x/m.m3u8").as_deref(), Some("/x/2k.m3u8"));
        // Audio of its own needs the master; a media playlist is played as it is.
        let rendition = "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",URI=\"a.m3u8\"\n\
            #EXT-X-STREAM-INF:BANDWIDTH=1,AUDIO=\"a\"\nv.m3u8\n";
        assert_eq!(variant(rendition, "m.m3u8"), None);
        assert_eq!(variant("#EXTM3U\n#EXTINF:2,\nclip0.ts\n", "clip.m3u8"), None);
    }

    #[test]
    fn tracks_come_with_their_language_and_title() {
        let info = probe(&Origin::File(assets().join("tracks.mp4"))).unwrap();
        let audio: Vec<_> = info
            .audio_tracks
            .iter()
            .map(|track| (track.id, track.language.as_deref(), track.title.as_deref(), track.codec.as_str()))
            .collect();
        assert_eq!(audio, [(1, Some("eng"), Some("Main"), "aac"), (2, Some("jpn"), Some("Commentary"), "aac")]);
        assert_eq!(info.audio_track, Some(1));
        let subtitles: Vec<_> = info
            .subtitle_tracks
            .iter()
            .map(|track| (track.id, track.language.as_deref(), track.title.as_deref()))
            .collect();
        assert_eq!(subtitles, [(3, Some("eng"), Some("English")), (4, Some("fre"), Some("Français"))]);
    }

    /// Decodes `file` with subtitle track `id` up to 4.5 s and returns the
    /// text showing at each of `times` (seconds).
    fn subtitles_at(file: &str, id: usize, times: &[f64]) -> Vec<Option<String>> {
        let origin = Origin::File(assets().join(file));
        let info = probe(&origin).unwrap();
        let choices = Choices::new(&info);
        choices.lock().unwrap().subtitle = Some(id);
        let slot = Arc::new(Mutex::new(None));
        let (mut video, _) = FfmpegVideo::open(&origin, None, slot.clone(), choices.clone()).unwrap();
        // The playback loop seeks first, which picks up the choice.
        video.seek(0).unwrap();
        while next_frame(&mut video, &slot) < 4_500_000 {}
        let mut choices = choices.lock().unwrap();
        times.iter().map(|&time| choices.text_at((time * 1e6) as i64)).collect()
    }

    #[test]
    fn mp4_text_subtitles_give_their_cues() {
        let text = subtitles_at("tracks.mp4", 3, &[0.2, 1.0, 2.2, 3.0]);
        let line = |text: &str| Some(text.to_owned());
        assert_eq!(text, [None, line("Hello from track one"), None, line("Second line\nover two rows")]);
        assert_eq!(subtitles_at("tracks.mp4", 4, &[2.0]), [line("Bonjour, piste deux")]);
    }

    #[test]
    fn subrip_and_ass_subtitles_give_plain_text() {
        let line = |text: &str| Some(text.to_owned());
        assert_eq!(subtitles_at("subtitles.mkv", 1, &[1.0]), [line("Hello from track one")]);
        assert_eq!(subtitles_at("subtitles.mkv", 2, &[2.0, 3.5]), [line("Styled, with commas\nNext row"), None]);
    }

    #[test]
    fn a_seek_switches_to_the_chosen_audio_track() {
        let origin = reader(assets().join("tracks.mp4"));
        let info = probe(&origin).unwrap();
        let choices = Choices::new(&info);
        let mut audio = FfmpegAudio::open(&origin, choices.clone()).unwrap();
        assert_eq!(audio.format().unwrap(), (48_000, 1));
        choices.lock().unwrap().audio = Some(2);
        audio.seek(1_000_000).unwrap();
        assert!(matches!(audio.next().unwrap(), AudioOutput::Format((44_100, 1))));
        let pts = loop {
            match audio.next().unwrap() {
                AudioOutput::Samples { pts, frames, .. } if frames > 0 => break pts,
                AudioOutput::End => panic!("no audio"),
                _ => {}
            }
        };
        // The packet holding 1 s may start a little after it.
        assert!((900_000..=1_050_000).contains(&pts), "audio seeked to {pts} µs");
    }

    #[test]
    fn a_maximum_height_scales_software_frames() {
        let origin = Origin::File(assets().join("tracks.mp4"));
        let info = probe(&origin).unwrap();
        let choices = Choices::new(&info);
        choices.lock().unwrap().max_height = Some(90);
        let slot = Arc::new(Mutex::new(None));
        let (mut video, _) = FfmpegVideo::open(&origin, None, slot.clone(), choices).unwrap();
        loop {
            match video.next().unwrap() {
                VideoOutput::Frame(_, frame) => {
                    video.release(frame, true).unwrap();
                    let picture = slot.lock().unwrap().take().unwrap();
                    assert_eq!((picture.frame.width(), picture.frame.height()), (160, 90));
                    break;
                }
                VideoOutput::Pending => {}
                VideoOutput::End => panic!("no frame"),
            }
        }
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
