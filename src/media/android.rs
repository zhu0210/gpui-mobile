//! Android: playback with `AMediaExtractor` + `AMediaCodec`, video decoded
//! into an `AImageReader` and audio played through AAudio, and the GPU import
//! of each video frame on the main thread.
//!
//! Video and audio run on their own threads, each with its own extractor, and
//! follow one clock ([`super::player`]). The video thread renders each frame
//! into the reader once the clock reaches it; the audio thread writes to
//! AAudio, whose timestamps drive the clock.
//!
//! The import (ported from lumina-video's `android_import.rs`) wraps the
//! frame's `AHardwareBuffer` in a `VkImage` with the buffer's external format
//! and draws it once through a YCbCr-converting sampler into an RGBA8 texture,
//! applying the crop and the display rotation. No pixels pass through the CPU.
//! It needs the Vulkan device extensions that the vendored `gpui-pre-wgpu`
//! enables on Android.

use anyhow::{anyhow, bail, Context as _, Result};
use ash::vk;
use gpui::{DevicePixels, ImageId, Size};
use gpui_wgpu::wgpu;
use ndk_sys as ffi;
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, CStr, CString};
use std::os::fd::AsRawFd;
use super::player::{monotonic_ns, play_video, Shared, VideoOutput, VideoSource, POLL};
use super::Source;
use std::ptr::{self, NonNull};
use std::sync::Arc;
#[cfg(feature = "ffmpeg")]
use std::sync::{Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Images the app may hold at once: the newest, the one still being drawn,
/// and one in flight on the GPU.
const MAX_IMAGES: i32 = 4;
const DEQUEUE_TIMEOUT_US: i64 = 10_000;
/// How long a seek waits for the audio stream's first timestamp.
const HOLD_TIMEOUT: Duration = Duration::from_millis(500);
/// Imported frames per drift report in the log.
const DRIFT_REPORT_FRAMES: u32 = 300;
/// `MediaFormat.KEY_ROTATION`, spelled out: the NDK constant needs API 28.
const KEY_ROTATION: &CStr = c"rotation-degrees";
const KEY_PCM_ENCODING: &CStr = c"pcm-encoding";
/// `AudioFormat.ENCODING_PCM_16BIT`, what decoders output unless asked otherwise.
const ENCODING_PCM_16BIT: i32 = 2;

pub(super) struct Decoder {
    reader: Reader,
    size: Size<DevicePixels>,
    rotation: i32,
    duration_us: i64,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    importer: RefCell<Importer>,
    frames_shown: Cell<u64>,
    last_import: Cell<Option<Instant>>,
    drift: Cell<Drift>,
    /// Software-decoded frames waiting for upload (FFmpeg's fallback).
    #[cfg(feature = "ffmpeg")]
    software: Option<Arc<Mutex<Option<super::planes::Picture>>>>,
    #[cfg(feature = "ffmpeg")]
    uploader: RefCell<super::planes::Uploader>,
}

/// How far imported frames were from the clock, between log reports.
#[derive(Clone, Copy, Default)]
struct Drift {
    frames: u32,
    total_us: i64,
    max_us: i64,
}

impl Decoder {
    pub fn open(source: Source, backend: super::Backend) -> Result<Self> {
        match backend {
            super::Backend::Platform => Self::open_platform(source),
            #[cfg(feature = "ffmpeg")]
            super::Backend::Ffmpeg => Self::open_ffmpeg(source, false),
            #[cfg(feature = "ffmpeg")]
            super::Backend::FfmpegSoftware => Self::open_ffmpeg(source, true),
        }
    }

    /// `AMediaExtractor` + `AMediaCodec`.
    fn open_platform(source: Source) -> Result<Self> {
        let video = Track::open(source, "video/")?.ok_or_else(|| anyhow!("no video track"))?;
        let audio = Track::open(source, "audio/").unwrap_or_else(|error| {
            log::warn!("video: audio track skipped: {error:#}");
            None
        });
        // SAFETY: the NDK key statics are initialised C strings.
        let (width, height, duration_us) = unsafe {
            (
                video.format.int32(ffi::AMEDIAFORMAT_KEY_WIDTH).unwrap_or(0),
                video.format.int32(ffi::AMEDIAFORMAT_KEY_HEIGHT).unwrap_or(0),
                video.format.int64(ffi::AMEDIAFORMAT_KEY_DURATION).unwrap_or(0),
            )
        };
        if width <= 0 || height <= 0 {
            bail!("video track without a size");
        }
        let rotation = video.format.int32(KEY_ROTATION.as_ptr()).unwrap_or(0).rem_euclid(360);
        let reader = Reader::new(width, height)?;
        // The reader's window stays valid until the reader is deleted, which
        // `Decoder::drop` does after the threads have deleted the codecs.
        video.start(reader.window()?)?;
        let audio = match audio {
            Some(audio) => {
                audio.start(ptr::null_mut())?;
                Some(CodecAudio::new(audio)?)
            }
            None => None,
        };
        log::info!(
            "video: decoding {} {width}x{height} (rotated {rotation}°, audio {}) from {source:?}",
            video.mime,
            audio.as_ref().map_or("none", |audio| audio.track.mime.as_str()),
        );
        let video = CodecVideo {
            track: video,
            input_done: false,
        };
        Self::start(
            reader,
            (width, height, rotation, duration_us),
            video,
            audio.map(|audio| Box::new(audio) as Box<dyn AudioSource + Send>),
        )
    }

    /// FFmpeg demuxing, decoding through MediaCodec into the reader, or in
    /// software when MediaCodec can't (or `software` asks for it).
    #[cfg(feature = "ffmpeg")]
    fn open_ffmpeg(source: Source, software: bool) -> Result<Self> {
        use super::ffmpeg::{probe, FfmpegAudio, FfmpegVideo};
        let Source::File(path) = source else {
            // Built without network: Cirrus will feed FFmpeg through its own IO.
            bail!("the FFmpeg backend plays files only, not {source:?}");
        };
        let info = probe(path)?;
        let reader = Reader::new(info.width, info.height)?;
        let slot = Arc::new(Mutex::new(None));
        let window = if software { None } else { Some(reader.window()?) };
        let (video, _hardware) = FfmpegVideo::open(path, window, slot.clone())?;
        let audio = if info.has_audio {
            FfmpegAudio::open(path)
                .map_err(|error| log::warn!("video: audio stream skipped: {error:#}"))
                .ok()
        } else {
            None
        };
        log::info!(
            "video: FFmpeg playing {}x{} (rotated {}°, audio {}) from {path:?}",
            info.width,
            info.height,
            info.rotation,
            audio.is_some(),
        );
        let mut decoder = Self::start(
            reader,
            (info.width, info.height, info.rotation, info.duration_us),
            video,
            audio.map(|audio| Box::new(audio) as Box<dyn AudioSource + Send>),
        )?;
        decoder.software = Some(slot);
        Ok(decoder)
    }

    /// Starts the playback threads.
    fn start<V: VideoSource + Send + 'static>(
        reader: Reader,
        (width, height, rotation, duration_us): (i32, i32, i32, i64),
        mut video: V,
        audio: Option<Box<dyn AudioSource + Send>>,
    ) -> Result<Self> {
        let shared = Shared::new(audio.is_some());
        let mut threads = vec![std::thread::Builder::new()
            .name("gpui-video".into())
            .spawn({
                let shared = shared.clone();
                move || {
                    if let Err(error) = play_video(&mut video, &shared) {
                        log::error!("video: decoding stopped: {error:#}");
                    }
                }
            })?];
        if let Some(mut audio) = audio {
            threads.push(std::thread::Builder::new().name("gpui-audio".into()).spawn({
                let shared = shared.clone();
                move || {
                    if let Err(error) = play_audio(audio.as_mut(), &shared) {
                        log::error!("video: audio stopped: {error:#}");
                    }
                    // The video goes on by the monotonic clock.
                    shared.update(|state| {
                        state.audio = false;
                        state.holding = false;
                    });
                }
            })?);
        }
        let size = if rotation % 180 == 90 { (height, width) } else { (width, height) };
        Ok(Self {
            reader,
            size: Size {
                width: DevicePixels(size.0),
                height: DevicePixels(size.1),
            },
            rotation,
            duration_us,
            shared,
            threads,
            importer: RefCell::new(Importer::default()),
            frames_shown: Cell::new(0),
            last_import: Cell::new(None),
            drift: Cell::new(Drift::default()),
            #[cfg(feature = "ffmpeg")]
            software: None,
            #[cfg(feature = "ffmpeg")]
            uploader: RefCell::new(Default::default()),
        })
    }

    /// The displayed size: the picture size, turned by the track's rotation.
    pub fn video_size(&self) -> Size<DevicePixels> {
        self.size
    }

    pub fn frames_shown(&self) -> u64 {
        self.frames_shown.get()
    }

    pub fn play(&self) {
        self.shared.play()
    }

    pub fn pause(&self) {
        self.shared.pause()
    }

    pub fn is_paused(&self) -> bool {
        self.shared.is_paused()
    }

    pub fn seek(&self, position: Duration) {
        self.shared.seek((position.as_micros() as i64).clamp(0, self.duration_us))
    }

    pub fn set_looping(&self, looping: bool) {
        self.shared.set_looping(looping)
    }

    pub fn position(&self) -> Duration {
        Duration::from_micros(self.shared.position_us().clamp(0, self.duration_us) as u64)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us as u64)
    }

    pub fn ended(&self) -> bool {
        self.shared.ended()
    }

    pub fn is_active(&self) -> bool {
        self.shared.is_active(self.last_import.get())
    }

    /// Imports the newest decoded image, if there is one, and makes it the
    /// texture drawn for `image_id`.
    pub fn update(&self, image_id: ImageId) -> Result<()> {
        #[cfg(feature = "ffmpeg")]
        if let Some(slot) = &self.software {
            let frame = slot.lock().unwrap_or_else(PoisonError::into_inner).take();
            if let Some(frame) = frame {
                return self.upload(image_id, &frame);
            }
        }
        let Some(image) = self.reader.acquire_latest()? else {
            return Ok(());
        };
        let timestamp_us = image.timestamp_us;
        let platform =
            crate::android::jni::platform().ok_or_else(|| anyhow!("platform not initialised"))?;
        let context = platform.gpu_context();
        let context = context.borrow();
        let context = context.as_ref().ok_or_else(|| anyhow!("no GPU context yet"))?;
        // SAFETY: the image came from our GPU-sampleable reader and was
        // acquired with `acquireLatestImage`, which waits for the decoder's fence.
        let prepared = unsafe {
            self.importer
                .borrow_mut()
                .prepare(image, self.rotation, &context.device)?
        };
        // Submitted before the frame that samples the texture, on the same queue.
        context.queue.submit(prepared.commands);
        gpui_wgpu::set_external_texture(image_id, &prepared.texture);
        self.last_import.set(Some(Instant::now()));
        let shown = self.frames_shown.get() + 1;
        self.frames_shown.set(shown);
        if shown == 1 {
            log::info!(
                "video: AHardwareBuffer frame imported as a {}x{} Vulkan texture",
                prepared.texture.width(),
                prepared.texture.height()
            );
        }
        self.record_drift(timestamp_us);
        Ok(())
    }

    /// Uploads a software-decoded frame and makes it the texture drawn for
    /// `image_id`.
    #[cfg(feature = "ffmpeg")]
    fn upload(&self, image_id: ImageId, frame: &super::planes::Picture) -> Result<()> {
        let platform =
            crate::android::jni::platform().ok_or_else(|| anyhow!("platform not initialised"))?;
        let context = platform.gpu_context();
        let context = context.borrow();
        let context = context.as_ref().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let texture = self
            .uploader
            .borrow_mut()
            .convert(frame, self.rotation, &context.device, &context.queue)?;
        gpui_wgpu::set_external_texture(image_id, &texture);
        self.last_import.set(Some(Instant::now()));
        let shown = self.frames_shown.get() + 1;
        self.frames_shown.set(shown);
        if shown == 1 {
            log::info!(
                "video: software frame uploaded as {}x{} planes, converted on the GPU",
                texture.width(),
                texture.height()
            );
        }
        self.record_drift(frame.timestamp_us);
        Ok(())
    }

    /// Logs how far imported frames trail or lead the clock while playing.
    fn record_drift(&self, timestamp_us: i64) {
        let state = self.shared.lock();
        if !state.running {
            return;
        }
        let drift_us = state.position_at(monotonic_ns()) - timestamp_us;
        drop(state);
        let mut drift = self.drift.get();
        drift.frames += 1;
        drift.total_us += drift_us;
        drift.max_us = drift.max_us.max(drift_us.abs());
        if drift.frames == DRIFT_REPORT_FRAMES {
            log::info!(
                "video: A/V drift over {} frames: mean {:.1} ms, max {:.1} ms",
                drift.frames,
                drift.total_us as f64 / drift.frames as f64 / 1000.0,
                drift.max_us as f64 / 1000.0,
            );
            drift = Drift::default();
        }
        self.drift.set(drift);
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.shared.update(|state| state.stop = true);
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                log::error!("video: playback thread panicked");
            }
        }
    }
}

/// What an audio source produced.
pub(super) enum AudioOutput {
    Pending,
    End,
    /// The output switched to this sample rate and channel count.
    Format((i32, i32)),
    /// Interleaved 16-bit samples, valid until `release`.
    Samples { pts: i64, data: *const i16, frames: usize },
}

pub(super) trait AudioSource {
    /// The sample rate and channel count before any `Format` output.
    fn format(&self) -> Result<(i32, i32)>;
    fn seek(&mut self, target_us: i64) -> Result<()>;
    fn next(&mut self) -> Result<AudioOutput>;
    /// Gives back the samples of the last `Samples` output.
    fn release(&mut self) -> Result<()>;
}

/// Decodes audio and writes it to an AAudio stream, which paces the thread;
/// the stream's timestamps drive the clock.
fn play_audio(source: &mut dyn AudioSource, shared: &Shared) -> Result<()> {
    let mut stream: Option<Stream> = None;
    let (mut generation, mut target) = (0, 0);
    // When the first samples since the last seek went to the stream.
    let mut first_write: Option<Instant> = None;
    // Frames written before the first sample since the last seek, and its time.
    let mut base: Option<(i64, i64)> = None;
    let mut end_us = 0;
    loop {
        let state = shared.lock();
        if state.stop {
            return Ok(());
        }
        if state.seek_generation != generation {
            generation = state.seek_generation;
            target = state.seek_target_us;
            drop(state);
            source.seek(target)?;
            if let Some(stream) = &stream {
                stream.pause_and_flush();
            }
            (first_write, base) = (None, None);
            continue;
        }
        let (running, holding, finished) = (state.running, state.holding, state.audio_end_us.is_some());
        if let Some(stream) = &stream {
            stream.follow(state.wants_sound());
        }
        if (!running && !holding) || finished {
            drop(shared.wait(state, POLL));
            continue;
        }
        drop(state);

        let (mut pts, data, frames) = match source.next()? {
            AudioOutput::Pending => continue,
            AudioOutput::Format((rate, channels)) => {
                stream = Some(Stream::open(rate, channels)?);
                continue;
            }
            AudioOutput::End => {
                shared.update(|state| {
                    if state.seek_generation == generation {
                        state.audio_end_us = Some(end_us.max(target));
                        state.holding = false;
                    }
                });
                continue;
            }
            AudioOutput::Samples { pts, data, frames } => (pts, data, frames),
        };
        if stream.is_none() {
            let (rate, channels) = source.format()?;
            stream = Some(Stream::open(rate, channels)?);
        }
        let stream = stream.as_ref().ok_or_else(|| anyhow!("no audio stream"))?;
        let (rate, channels) = (stream.rate as i64, stream.channels as usize);
        let mut offset = 0;
        if pts < target {
            offset = (((target - pts) * rate / 1_000_000) as usize).min(frames);
            pts += offset as i64 * 1_000_000 / rate;
        }
        if offset < frames {
            let (base_frames, base_us) = *base.get_or_insert((stream.frames_written(), pts));
            end_us = pts + (frames - offset) as i64 * 1_000_000 / rate;
            while offset < frames {
                let state = shared.lock();
                if state.stop || state.seek_generation != generation {
                    break;
                }
                let sound = state.wants_sound();
                drop(state);
                stream.follow(sound);
                // Blocks while the stream's buffer is full, which paces this thread.
                // SAFETY: the source keeps `frames` frames at `data` until `release`.
                let written = unsafe { stream.write(data.add(offset * channels).cast(), frames - offset)? };
                offset += written;
                if written > 0 {
                    first_write.get_or_insert_with(Instant::now);
                }
                // Without timestamps the clock starts on its own.
                if first_write.is_some_and(|time| time.elapsed() > HOLD_TIMEOUT) {
                    shared.update(|state| {
                        if state.seek_generation == generation {
                            state.holding = false;
                        }
                    });
                }
                if let Some((frame, time_ns)) = stream.timestamp() {
                    if frame >= base_frames {
                        shared.sync(generation, base_us + (frame - base_frames) * 1_000_000 / rate, time_ns);
                    }
                }
            }
        }
        source.release()?;
    }
}

/// The platform video decoder: an `AMediaCodec` rendering into the reader.
struct CodecVideo {
    track: Track,
    input_done: bool,
}

impl VideoSource for CodecVideo {
    /// The codec's output buffer index.
    type Frame = usize;

    fn seek(&mut self, target_us: i64) -> Result<()> {
        self.input_done = false;
        self.track.seek(target_us)
    }

    fn next(&mut self) -> Result<VideoOutput<usize>> {
        if !self.input_done {
            self.input_done = self.track.feed()?;
        }
        let codec = self.track.codec.0.as_ptr();
        // SAFETY: the codec is owned by this thread; the index comes from it.
        unsafe {
            let mut info = std::mem::zeroed::<ffi::AMediaCodecBufferInfo>();
            let index = ffi::AMediaCodec_dequeueOutputBuffer(codec, &mut info, DEQUEUE_TIMEOUT_US);
            if index < 0 {
                // Try-again, format and buffer changes need nothing here.
                return Ok(VideoOutput::Pending);
            }
            if info.flags & ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32 != 0 {
                status(ffi::AMediaCodec_releaseOutputBuffer(codec, index as usize, false))?;
                return Ok(VideoOutput::End);
            }
            Ok(VideoOutput::Frame(info.presentationTimeUs, index as usize))
        }
    }

    fn release(&mut self, index: usize, render: bool) -> Result<()> {
        // SAFETY: the index came from this codec and is released once.
        status(unsafe { ffi::AMediaCodec_releaseOutputBuffer(self.track.codec.0.as_ptr(), index, render) })
    }
}

/// The platform audio decoder: an `AMediaCodec` decoding to memory.
struct CodecAudio {
    track: Track,
    input_done: bool,
    channels: i32,
    /// The output buffer being played.
    index: Option<usize>,
}

impl CodecAudio {
    fn new(track: Track) -> Result<Self> {
        let (_, channels) = pcm_format(&track.format)?;
        Ok(Self {
            track,
            input_done: false,
            channels,
            index: None,
        })
    }
}

impl AudioSource for CodecAudio {
    fn format(&self) -> Result<(i32, i32)> {
        pcm_format(&self.track.format)
    }

    fn seek(&mut self, target_us: i64) -> Result<()> {
        self.input_done = false;
        self.track.seek(target_us)
    }

    fn next(&mut self) -> Result<AudioOutput> {
        if !self.input_done {
            self.input_done = self.track.feed()?;
        }
        let codec = self.track.codec.0.as_ptr();
        // SAFETY: the codec is owned by this thread; the index, buffer and
        // length come from it and stay valid until the buffer is released.
        unsafe {
            let mut info = std::mem::zeroed::<ffi::AMediaCodecBufferInfo>();
            let index = ffi::AMediaCodec_dequeueOutputBuffer(codec, &mut info, DEQUEUE_TIMEOUT_US);
            if index == ffi::AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED as isize {
                let format = NonNull::new(ffi::AMediaCodec_getOutputFormat(codec))
                    .map(Format)
                    .ok_or_else(|| anyhow!("no audio output format"))?;
                let format = pcm_format(&format)?;
                self.channels = format.1;
                return Ok(AudioOutput::Format(format));
            }
            if index < 0 {
                return Ok(AudioOutput::Pending);
            }
            let index = index as usize;
            if info.flags & ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32 != 0 {
                status(ffi::AMediaCodec_releaseOutputBuffer(codec, index, false))?;
                return Ok(AudioOutput::End);
            }
            let mut capacity = 0;
            let data = ffi::AMediaCodec_getOutputBuffer(codec, index, &mut capacity).add(info.offset as usize);
            self.index = Some(index);
            Ok(AudioOutput::Samples {
                pts: info.presentationTimeUs,
                data: data.cast(),
                frames: info.size as usize / (self.channels as usize * 2),
            })
        }
    }

    fn release(&mut self) -> Result<()> {
        match self.index.take() {
            // SAFETY: the index came from this codec and is released once.
            Some(index) => status(unsafe {
                ffi::AMediaCodec_releaseOutputBuffer(self.track.codec.0.as_ptr(), index, false)
            }),
            None => Ok(()),
        }
    }
}

/// The sample rate and channel count of a decoder's 16-bit PCM output.
fn pcm_format(format: &Format) -> Result<(i32, i32)> {
    // SAFETY: the NDK key statics are initialised C strings.
    let (rate, channels) = unsafe {
        (
            format.int32(ffi::AMEDIAFORMAT_KEY_SAMPLE_RATE),
            format.int32(ffi::AMEDIAFORMAT_KEY_CHANNEL_COUNT),
        )
    };
    let (Some(rate), Some(channels)) = (rate, channels) else {
        bail!("audio format without a sample rate or channel count");
    };
    if let Some(encoding) = format.int32(KEY_PCM_ENCODING.as_ptr()) {
        if encoding != ENCODING_PCM_16BIT {
            bail!("audio decoder outputs PCM encoding {encoding}, not 16-bit");
        }
    }
    Ok((rate, channels))
}

fn status(status: ffi::media_status_t) -> Result<()> {
    if status == ffi::media_status_t::AMEDIA_OK {
        Ok(())
    } else {
        bail!("media status {}", status.0)
    }
}

/// One track of a file: its own extractor with only that track selected, its
/// format, and a decoder for it.
struct Track {
    extractor: Extractor,
    format: Format,
    codec: Codec,
    mime: String,
}

// SAFETY: a track moves to its playback thread and is only used there.
unsafe impl Send for Track {}

impl Track {
    /// The first track whose MIME type starts with `kind`.
    fn open(source: Source, kind: &str) -> Result<Option<Self>> {
        // SAFETY: plain constructor.
        let extractor = NonNull::new(unsafe { ffi::AMediaExtractor_new() })
            .map(Extractor)
            .ok_or_else(|| anyhow!("AMediaExtractor_new failed"))?;
        match source {
            Source::File(path) => {
                let file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
                let length = file.metadata()?.len() as i64;
                // SAFETY: the extractor is live and the descriptor open for
                // the call (the extractor dups what it keeps).
                status(unsafe {
                    ffi::AMediaExtractor_setDataSourceFd(extractor.0.as_ptr(), file.as_raw_fd(), 0, length)
                })
                .context("AMediaExtractor_setDataSourceFd")?;
            }
            Source::Url(url) => {
                let url = CString::new(url)?;
                // ponytail: connects on the calling thread (the UI thread when
                // opened from a view); move it to a thread if streams start slowly.
                // SAFETY: a live extractor and a C string for the call.
                status(unsafe { ffi::AMediaExtractor_setDataSource(extractor.0.as_ptr(), url.as_ptr()) })
                    .with_context(|| format!("AMediaExtractor_setDataSource {url:?}"))?;
            }
        }
        // SAFETY: the extractor is live; formats are owned by `Format`.
        unsafe {
            for track in 0..ffi::AMediaExtractor_getTrackCount(extractor.0.as_ptr()) {
                let Some(format) =
                    NonNull::new(ffi::AMediaExtractor_getTrackFormat(extractor.0.as_ptr(), track))
                        .map(Format)
                else {
                    continue;
                };
                let Some(mime) = format.string(ffi::AMEDIAFORMAT_KEY_MIME) else {
                    continue;
                };
                if !mime.starts_with(kind) {
                    continue;
                }
                status(ffi::AMediaExtractor_selectTrack(extractor.0.as_ptr(), track))
                    .context("AMediaExtractor_selectTrack")?;
                let mime_c = CString::new(mime.clone())?;
                let codec = NonNull::new(ffi::AMediaCodec_createDecoderByType(mime_c.as_ptr()))
                    .map(Codec)
                    .ok_or_else(|| anyhow!("no decoder for {mime}"))?;
                return Ok(Some(Self {
                    extractor,
                    format,
                    codec,
                    mime,
                }));
            }
        }
        Ok(None)
    }

    /// Configures the decoder, onto `window` or to memory when it is null,
    /// and starts it.
    fn start(&self, window: *mut ffi::ANativeWindow) -> Result<()> {
        // SAFETY: the codec and format are live; the window outlives the codec
        // (see `Decoder::open`).
        unsafe {
            status(ffi::AMediaCodec_configure(
                self.codec.0.as_ptr(),
                self.format.0.as_ptr(),
                window,
                ptr::null_mut(),
                0,
            ))
            .context("AMediaCodec_configure")?;
            status(ffi::AMediaCodec_start(self.codec.0.as_ptr())).context("AMediaCodec_start")
        }
    }

    /// Queues the next sample, if the decoder has room; returns true once the
    /// end of the stream is queued.
    fn feed(&self) -> Result<bool> {
        let (codec, extractor) = (self.codec.0.as_ptr(), self.extractor.0.as_ptr());
        // SAFETY: codec and extractor are live and used by this thread only;
        // the buffer comes from the codec just before use.
        unsafe {
            let index = ffi::AMediaCodec_dequeueInputBuffer(codec, 0);
            if index < 0 {
                return Ok(false);
            }
            let index = index as usize;
            let mut capacity = 0;
            let buffer = ffi::AMediaCodec_getInputBuffer(codec, index, &mut capacity);
            let size = ffi::AMediaExtractor_readSampleData(extractor, buffer, capacity);
            if size < 0 {
                status(ffi::AMediaCodec_queueInputBuffer(
                    codec,
                    index,
                    0,
                    0,
                    0,
                    ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32,
                ))?;
                return Ok(true);
            }
            let time = ffi::AMediaExtractor_getSampleTime(extractor).max(0);
            status(ffi::AMediaCodec_queueInputBuffer(codec, index, 0, size as usize, time as u64, 0))?;
            ffi::AMediaExtractor_advance(extractor);
            Ok(false)
        }
    }

    /// Moves to the sync sample at or before `target_us` and drops what the
    /// decoder holds.
    fn seek(&self, target_us: i64) -> Result<()> {
        // SAFETY: codec and extractor are live and used by this thread only.
        unsafe {
            status(ffi::AMediaExtractor_seekTo(
                self.extractor.0.as_ptr(),
                target_us,
                ffi::SeekMode::AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC,
            ))?;
            status(ffi::AMediaCodec_flush(self.codec.0.as_ptr()))
        }
    }
}

struct Extractor(NonNull<ffi::AMediaExtractor>);

impl Drop for Extractor {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once.
        unsafe { ffi::AMediaExtractor_delete(self.0.as_ptr()) };
    }
}

struct Format(NonNull<ffi::AMediaFormat>);

impl Format {
    fn int32(&self, key: *const c_char) -> Option<i32> {
        let mut value = 0;
        // SAFETY: the format is live and `key` a C string.
        unsafe { ffi::AMediaFormat_getInt32(self.0.as_ptr(), key, &mut value) }.then_some(value)
    }

    fn int64(&self, key: *const c_char) -> Option<i64> {
        let mut value = 0;
        // SAFETY: the format is live and `key` a C string.
        unsafe { ffi::AMediaFormat_getInt64(self.0.as_ptr(), key, &mut value) }.then_some(value)
    }

    fn string(&self, key: *const c_char) -> Option<String> {
        let mut value = ptr::null();
        // SAFETY: the format is live and `key` a C string; the returned string
        // stays valid while the format lives.
        unsafe {
            if !ffi::AMediaFormat_getString(self.0.as_ptr(), key, &mut value) || value.is_null() {
                return None;
            }
            Some(CStr::from_ptr(value).to_string_lossy().into_owned())
        }
    }
}

impl Drop for Format {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once.
        unsafe { ffi::AMediaFormat_delete(self.0.as_ptr()) };
    }
}

struct Codec(NonNull<ffi::AMediaCodec>);

impl Drop for Codec {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once; stopping first returns its buffers.
        unsafe {
            ffi::AMediaCodec_stop(self.0.as_ptr());
            ffi::AMediaCodec_delete(self.0.as_ptr());
        }
    }
}

/// An AAudio output stream of 16-bit PCM.
struct Stream {
    stream: NonNull<ffi::AAudioStream>,
    rate: i32,
    channels: i32,
    started: Cell<bool>,
}

impl Stream {
    /// Opens a stream of 16-bit PCM at `rate` with `channels` channels.
    fn open(rate: i32, channels: i32) -> Result<Self> {
        // SAFETY: the builder is deleted once, after opening; the stream it
        // opens is owned by the returned value.
        unsafe {
            let mut builder = ptr::null_mut();
            aaudio(ffi::AAudio_createStreamBuilder(&mut builder)).context("AAudio_createStreamBuilder")?;
            ffi::AAudioStreamBuilder_setFormat(builder, ffi::AAUDIO_FORMAT_PCM_I16 as i32);
            ffi::AAudioStreamBuilder_setSampleRate(builder, rate);
            ffi::AAudioStreamBuilder_setChannelCount(builder, channels);
            let mut stream = ptr::null_mut();
            let opened = aaudio(ffi::AAudioStreamBuilder_openStream(builder, &mut stream));
            ffi::AAudioStreamBuilder_delete(builder);
            opened.context("AAudioStreamBuilder_openStream")?;
            log::info!("video: audio through AAudio, {rate} Hz, {channels} channels");
            Ok(Self {
                stream: NonNull::new(stream).ok_or_else(|| anyhow!("null AAudio stream"))?,
                rate,
                channels,
                started: Cell::new(false),
            })
        }
    }

    /// Starts or pauses the stream to match the clock.
    fn follow(&self, running: bool) {
        if running == self.started.get() {
            return;
        }
        // SAFETY: the stream is live.
        unsafe {
            if running {
                ffi::AAudioStream_requestStart(self.stream.as_ptr());
            } else {
                ffi::AAudioStream_requestPause(self.stream.as_ptr());
            }
        }
        self.started.set(running);
    }

    /// Pauses the stream and drops what it has not played.
    fn pause_and_flush(&self) {
        self.follow(false);
        // SAFETY: the stream is live; flushing needs the pause to complete.
        unsafe {
            let mut next = 0;
            ffi::AAudioStream_waitForStateChange(
                self.stream.as_ptr(),
                ffi::AAUDIO_STREAM_STATE_PAUSING as i32,
                &mut next,
                100_000_000,
            );
            ffi::AAudioStream_requestFlush(self.stream.as_ptr());
        }
    }

    fn frames_written(&self) -> i64 {
        // SAFETY: the stream is live.
        unsafe { ffi::AAudioStream_getFramesWritten(self.stream.as_ptr()) }
    }

    /// Writes up to `frames` frames from `data`; returns how many it took.
    ///
    /// # Safety
    /// `data` must hold `frames` frames of the stream's format.
    unsafe fn write(&self, data: *const std::ffi::c_void, frames: usize) -> Result<usize> {
        // SAFETY: the stream is live; the caller vouches for `data`.
        let written = unsafe {
            ffi::AAudioStream_write(self.stream.as_ptr(), data, frames as i32, POLL.as_nanos() as i64)
        };
        aaudio(written).context("AAudioStream_write")?;
        Ok(written as usize)
    }

    /// The frame position last presented, and when, on the monotonic clock.
    fn timestamp(&self) -> Option<(i64, i64)> {
        let (mut frame, mut time) = (0, 0);
        // SAFETY: the stream is live; outputs are local.
        let result = unsafe {
            ffi::AAudioStream_getTimestamp(
                self.stream.as_ptr(),
                libc::CLOCK_MONOTONIC as _,
                &mut frame,
                &mut time,
            )
        };
        (result == ffi::AAUDIO_OK as i32).then_some((frame, time))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: owned stream, closed once.
        unsafe {
            ffi::AAudioStream_requestStop(self.stream.as_ptr());
            ffi::AAudioStream_close(self.stream.as_ptr());
        }
    }
}

fn aaudio(result: ffi::aaudio_result_t) -> Result<()> {
    if result >= 0 {
        Ok(())
    } else {
        bail!("AAudio error {result}")
    }
}

struct Reader(NonNull<ffi::AImageReader>);

impl Reader {
    fn new(width: i32, height: i32) -> Result<Self> {
        let mut reader = ptr::null_mut();
        // SAFETY: output pointer to local storage.
        status(unsafe {
            ffi::AImageReader_newWithUsage(
                width,
                height,
                ffi::AIMAGE_FORMATS::AIMAGE_FORMAT_PRIVATE.0 as i32,
                ffi::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0 as u64,
                MAX_IMAGES,
                &mut reader,
            )
        })
        .context("AImageReader_newWithUsage")?;
        NonNull::new(reader)
            .map(Self)
            .ok_or_else(|| anyhow!("AImageReader_newWithUsage returned null"))
    }

    fn window(&self) -> Result<*mut ffi::ANativeWindow> {
        let mut window = ptr::null_mut();
        // SAFETY: the reader is live; the window it returns is owned by it.
        status(unsafe { ffi::AImageReader_getWindow(self.0.as_ptr(), &mut window) })?;
        Ok(window)
    }

    fn acquire_latest(&self) -> Result<Option<Arc<AcquiredImage>>> {
        let mut image = ptr::null_mut();
        // SAFETY: the reader is live and `image` is local output storage.
        let result = unsafe { ffi::AImageReader_acquireLatestImage(self.0.as_ptr(), &mut image) };
        if result == ffi::media_status_t::AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE
            || result == ffi::media_status_t::AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED
        {
            return Ok(None);
        }
        status(result).context("AImageReader_acquireLatestImage")?;
        let image = NonNull::new(image).ok_or_else(|| anyhow!("null AImage"))?;
        // SAFETY: the image is live; `AcquiredImage` owns it (and the buffer
        // reference taken here) from here on, so error paths release both.
        unsafe {
            let mut acquired = AcquiredImage {
                image,
                buffer: ptr::null_mut(),
                buffer_size: (0, 0),
                crop: vk::Rect2D::default(),
                timestamp_us: 0,
            };
            let mut buffer = ptr::null_mut();
            status(ffi::AImage_getHardwareBuffer(image.as_ptr(), &mut buffer))?;
            if buffer.is_null() {
                bail!("AImage without a hardware buffer");
            }
            ffi::AHardwareBuffer_acquire(buffer);
            acquired.buffer = buffer;
            let mut description = std::mem::zeroed::<ffi::AHardwareBuffer_Desc>();
            ffi::AHardwareBuffer_describe(buffer, &mut description);
            acquired.buffer_size = (description.width, description.height);
            let mut crop = std::mem::zeroed::<ffi::AImageCropRect>();
            status(ffi::AImage_getCropRect(image.as_ptr(), &mut crop))?;
            acquired.crop = vk::Rect2D {
                offset: vk::Offset2D {
                    x: crop.left,
                    y: crop.top,
                },
                extent: vk::Extent2D {
                    width: (crop.right - crop.left).max(1) as u32,
                    height: (crop.bottom - crop.top).max(1) as u32,
                },
            };
            let mut timestamp_ns = 0;
            status(ffi::AImage_getTimestamp(image.as_ptr(), &mut timestamp_ns))?;
            acquired.timestamp_us = timestamp_ns / 1000;
            Ok(Some(Arc::new(acquired)))
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once, after the codec writing into it.
        unsafe { ffi::AImageReader_delete(self.0.as_ptr()) };
    }
}

/// A decoded image held until the GPU has finished reading its buffer, so the
/// codec cannot reuse the buffer underneath the conversion.
struct AcquiredImage {
    image: NonNull<ffi::AImage>,
    buffer: *mut ffi::AHardwareBuffer,
    buffer_size: (u32, u32),
    crop: vk::Rect2D,
    /// The presentation time the decoder gave the frame.
    timestamp_us: i64,
}

// SAFETY: AImage and AHardwareBuffer handles may be released from any thread;
// nothing reads through them after acquisition.
unsafe impl Send for AcquiredImage {}
unsafe impl Sync for AcquiredImage {}

impl Drop for AcquiredImage {
    fn drop(&mut self) {
        // SAFETY: the buffer reference was taken in `acquire_latest`; the
        // image is owned and deleted once.
        unsafe {
            if !self.buffer.is_null() {
                ffi::AHardwareBuffer_release(self.buffer);
            }
            ffi::AImage_delete(self.image.as_ptr());
        }
    }
}

struct PreparedFrame {
    commands: Vec<wgpu::CommandBuffer>,
    texture: Arc<wgpu::Texture>,
}

#[derive(Default)]
struct Importer {
    pipeline: Option<Arc<ConversionPipeline>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ConversionKey {
    external_format: u64,
    model: vk::SamplerYcbcrModelConversion,
    range: vk::SamplerYcbcrRange,
    x: vk::ChromaLocation,
    y: vk::ChromaLocation,
    components: [vk::ComponentSwizzle; 4],
}

struct ConversionPipeline {
    /// Keeps the device alive while frames converted on it still exist,
    /// including after a device loss replaced it.
    _owner: wgpu::Device,
    device: ash::Device,
    key: ConversionKey,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

impl Drop for ConversionPipeline {
    fn drop(&mut self) {
        // SAFETY: every frame retains this pipeline until its GPU work is done;
        // the handles belong to this device and are destroyed in dependency order.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_sampler_ycbcr_conversion(self.conversion, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

struct FrameResources {
    pipeline: Arc<ConversionPipeline>,
    producer: Arc<AcquiredImage>,
    input: vk::Image,
    input_memory: vk::DeviceMemory,
    input_view: vk::ImageView,
    output: vk::Image,
    output_memory: vk::DeviceMemory,
    output_view: vk::ImageView,
    descriptor_pool: vk::DescriptorPool,
    framebuffer: vk::Framebuffer,
}

impl Drop for FrameResources {
    fn drop(&mut self) {
        // SAFETY: before submission this object owns every handle; afterwards
        // the output texture's HAL drop callback runs only once the GPU is done
        // with it. Views and the framebuffer go before images and memory.
        unsafe {
            let device = &self.pipeline.device;
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_image_view(self.input_view, None);
            device.destroy_image_view(self.output_view, None);
            device.destroy_image(self.input, None);
            device.destroy_image(self.output, None);
            device.free_memory(self.input_memory, None);
            device.free_memory(self.output_memory, None);
        }
    }
}

fn memory_type(bits: u32) -> Result<u32> {
    if bits == 0 {
        bail!("no compatible Vulkan memory type");
    }
    Ok(bits.trailing_zeros())
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

impl Importer {
    /// Records the conversion of `frame` into a new RGBA8 texture of its crop
    /// size, turned clockwise by `rotation` degrees. The caller submits
    /// `commands` before sampling `texture`.
    ///
    /// # Safety
    /// `frame` must come from a `GPU_SAMPLED_IMAGE` reader with its producer
    /// fence already waited on, and `device` must be GPUI's Vulkan device.
    unsafe fn prepare(
        &mut self,
        frame: Arc<AcquiredImage>,
        rotation: i32,
        device: &wgpu::Device,
    ) -> Result<PreparedFrame> {
        // SAFETY: a live wgpu device; no raw queue operations happen here.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| anyhow!("video frames need the Vulkan backend"))?;
        for extension in [
            ash::android::external_memory_android_hardware_buffer::NAME,
            ash::ext::queue_family_foreign::NAME,
        ] {
            if !hal.enabled_device_extensions().contains(&extension) {
                bail!("Vulkan device lacks {extension:?}");
            }
        }
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let android = ash::android::external_memory_android_hardware_buffer::Device::new(instance, raw);
        let mut format = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
        let (allocation_size, memory_bits) = {
            let mut properties =
                vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format);
            // SAFETY: the buffer is referenced by `frame`; the output chain is local.
            unsafe { android.get_android_hardware_buffer_properties(frame.buffer.cast(), &mut properties) }?;
            (properties.allocation_size, properties.memory_type_bits)
        };
        if !format.format_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            bail!("AHardwareBuffer cannot be sampled");
        }
        // Always import through the external format: the buffer's reported
        // features and suggested conversion describe it, even when a matching
        // VkFormat exists.
        let key = ConversionKey {
            external_format: format.external_format,
            model: format.suggested_ycbcr_model,
            range: format.suggested_ycbcr_range,
            x: format.suggested_x_chroma_offset,
            y: format.suggested_y_chroma_offset,
            components: [
                format.sampler_ycbcr_conversion_components.r,
                format.sampler_ycbcr_conversion_components.g,
                format.sampler_ycbcr_conversion_components.b,
                format.sampler_ycbcr_conversion_components.a,
            ],
        };
        if key.external_format == 0 {
            bail!("AHardwareBuffer has no external format");
        }
        // A new device (after a device loss) needs its own pipeline.
        if self
            .pipeline
            .as_ref()
            .is_none_or(|pipeline| pipeline.key != key || pipeline.device.handle() != raw.handle())
        {
            // SAFETY: the device enabled samplerYcbcrConversion (vendored
            // gpui-pre-wgpu) and the key comes from this device's AHB properties.
            self.pipeline = Some(Arc::new(unsafe { ConversionPipeline::new(device.clone(), raw.clone(), key) }?));
        }
        let pipeline = self.pipeline.clone().ok_or_else(|| anyhow!("missing pipeline"))?;
        let (buffer_width, buffer_height) = frame.buffer_size;
        let crop = frame.crop;
        let (width, height) = if rotation % 180 == 90 {
            (crop.extent.height, crop.extent.width)
        } else {
            (crop.extent.width, crop.extent.height)
        };
        let mapping = super::uv_mapping(
            (crop.offset.x, crop.offset.y, crop.extent.width, crop.extent.height),
            (buffer_width, buffer_height),
            rotation,
        );
        let mut resources = FrameResources {
            pipeline,
            producer: frame,
            input: vk::Image::null(),
            input_memory: vk::DeviceMemory::null(),
            input_view: vk::ImageView::null(),
            output: vk::Image::null(),
            output_memory: vk::DeviceMemory::null(),
            output_view: vk::ImageView::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            framebuffer: vk::Framebuffer::null(),
        };
        let mut external_memory = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
        let mut external_format = vk::ExternalFormatANDROID::default().external_format(key.external_format);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::UNDEFINED)
            .extent(vk::Extent3D {
                width: buffer_width,
                height: buffer_height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory)
            .push_next(&mut external_format);
        // SAFETY: the descriptors match the queried buffer; `resources` owns
        // every handle created here and frees partial work on any error.
        unsafe {
            resources.input = raw.create_image(&image_info, None)?;
            let mut import =
                vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(resources.producer.buffer.cast());
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(resources.input);
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(memory_type(memory_bits)?)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            resources.input_memory = raw.allocate_memory(&allocation, None)?;
            raw.bind_image_memory(resources.input, resources.input_memory, 0)?;
            let mut conversion =
                vk::SamplerYcbcrConversionInfo::default().conversion(resources.pipeline.conversion);
            resources.input_view = raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(resources.input)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::UNDEFINED)
                    .subresource_range(color_range())
                    .push_next(&mut conversion),
                None,
            )?;

            resources.output = raw.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let requirements = raw.get_image_memory_requirements(resources.output);
            resources.output_memory = raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type(requirements.memory_type_bits)?),
                None,
            )?;
            raw.bind_image_memory(resources.output, resources.output_memory, 0)?;
            resources.output_view = raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(resources.output)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(color_range()),
                None,
            )?;
            // The spec bounds combined YCbCr descriptors by the plane count;
            // four covers external formats with alpha.
            let pool_size = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4,
            }];
            resources.descriptor_pool = raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_size),
                None,
            )?;
            let layouts = [resources.pipeline.descriptor_layout];
            let descriptor_set = raw
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool)
                        .set_layouts(&layouts),
                )?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no descriptor set returned"))?;
            let image = [vk::DescriptorImageInfo::default()
                .image_view(resources.input_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            raw.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image)],
                &[],
            );
            let attachments = [resources.output_view];
            resources.framebuffer = raw.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(resources.pipeline.render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?;

            let resources = Arc::new(resources);
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let retained = Arc::clone(&resources);
            let texture = hal.texture_from_raw(
                resources.output,
                &wgpu::hal::TextureDescriptor {
                    label: Some("video frame"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COLOR_TARGET,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: Vec::new(),
                },
                Some(Box::new(move || drop(retained))),
                wgpu::hal::vulkan::TextureMemory::External,
            );
            let texture = device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                texture,
                &wgpu::TextureDescriptor {
                    label: Some("video frame"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
                wgpu::TextureUses::UNINITIALIZED,
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("video frame initialization"),
            });
            // Clear through wgpu so its init tracker won't clear over the raw
            // conversion later; this also ties the output's lifetime to the
            // submission.
            let output_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("video frame initialization"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            encoder.transition_resources(
                std::iter::empty(),
                std::iter::once(wgpu::TextureTransition {
                    texture: &texture,
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                }),
            );
            let initialize = encoder.finish();
            // wgpu 30 doesn't allow mixing wgpu and raw encoding on one encoder.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("video frame YCbCr conversion"),
            });
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                let encoder = encoder.ok_or_else(|| anyhow!("missing Vulkan command encoder"))?;
                resources.record(
                    encoder.raw_handle(),
                    hal.queue_family_index(),
                    descriptor_set,
                    vk::Extent2D { width, height },
                    &mapping,
                );
                anyhow::Ok(())
            })?;
            let texture = Arc::new(texture);
            // Raw encoding bypasses wgpu's tracker: keep the texture (and the
            // decoded image behind it) alive until this buffer has run.
            let retained = Arc::clone(&texture);
            encoder.on_submitted_work_done(move || drop(retained));
            Ok(PreparedFrame {
                commands: vec![initialize, encoder.finish()],
                texture,
            })
        }
    }
}

impl FrameResources {
    /// Draws the input into the whole output, sampling where `mapping` (see
    /// `media::uv_mapping`) points.
    unsafe fn record(
        &self,
        command: vk::CommandBuffer,
        family: u32,
        descriptor: vk::DescriptorSet,
        extent: vk::Extent2D,
        mapping: &[f32; 8],
    ) {
        let raw = &self.pipeline.device;
        let acquire = vk::ImageMemoryBarrier::default()
            .image(self.input)
            .subresource_range(color_range())
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(family);
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        // SAFETY: an active, exclusively borrowed wgpu Vulkan command buffer;
        // every resource and the decoded image outlive its submission.
        unsafe {
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[acquire],
            );
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            }];
            raw.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pipeline.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(area)
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout,
                0,
                &[descriptor],
                &[],
            );
            raw.cmd_set_viewport(
                command,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            raw.cmd_set_scissor(command, 0, &[area]);
            let bytes: Vec<u8> = mapping.iter().flat_map(|value| value.to_ne_bytes()).collect();
            raw.cmd_push_constants(
                command,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &bytes,
            );
            raw.cmd_draw(command, 3, 1, 0, 0);
            raw.cmd_end_render_pass(command);
            let release = vk::ImageMemoryBarrier::default()
                .image(self.input)
                .subresource_range(color_range())
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT);
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[release],
            );
        }
    }
}

impl ConversionPipeline {
    unsafe fn new(owner: wgpu::Device, device: ash::Device, key: ConversionKey) -> Result<Self> {
        let mut state = Self {
            _owner: owner,
            device,
            key,
            conversion: vk::SamplerYcbcrConversion::null(),
            sampler: vk::Sampler::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
        };
        // SAFETY: the device enabled samplerYcbcrConversion, the key is from
        // its AHB properties, and `state` frees partial work on errors.
        unsafe {
            let mut external = vk::ExternalFormatANDROID::default().external_format(key.external_format);
            let [r, g, b, a] = key.components;
            state.conversion = state.device.create_sampler_ycbcr_conversion(
                &vk::SamplerYcbcrConversionCreateInfo::default()
                    .format(vk::Format::UNDEFINED)
                    .ycbcr_model(key.model)
                    .ycbcr_range(key.range)
                    .components(vk::ComponentMapping { r, g, b, a })
                    .x_chroma_offset(key.x)
                    .y_chroma_offset(key.y)
                    .chroma_filter(vk::Filter::NEAREST)
                    .push_next(&mut external),
                None,
            )?;
            let mut conversion = vk::SamplerYcbcrConversionInfo::default().conversion(state.conversion);
            state.sampler = state.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0)
                    .push_next(&mut conversion),
                None,
            )?;
            let samplers = [state.sampler];
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&samplers)];
            state.descriptor_layout = state.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?;
            state.pipeline_layout = state.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[state.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        offset: 0,
                        size: size_of::<[f32; 8]>() as u32,
                    }]),
                None,
            )?;
            let attachment = [vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let color = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let subpass = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color)];
            let dependencies = [
                // The wgpu initialization pass runs first on the same queue.
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::SubpassDependency::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            ];
            state.render_pass = state.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachment)
                    .subpasses(&subpass)
                    .dependencies(&dependencies),
                None,
            )?;
            // Built from the GLSL next to them with
            // `glslc -o shaders/<name>.spv shaders/<name>`.
            let vertex = shader(&state.device, include_bytes!("shaders/blit.vert.spv"))?;
            let fragment = match shader(&state.device, include_bytes!("shaders/ycbcr_blit.frag.spv")) {
                Ok(shader) => shader,
                Err(error) => {
                    state.device.destroy_shader_module(vertex, None);
                    return Err(error);
                }
            };
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment)
                    .name(c"main"),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend =
                [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
            let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .dynamic_state(&dynamic_state)
                .layout(state.pipeline_layout)
                .render_pass(state.render_pass);
            let result = state
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
            state.device.destroy_shader_module(fragment, None);
            state.device.destroy_shader_module(vertex, None);
            match result {
                Ok(pipelines) => {
                    state.pipeline = pipelines
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("no conversion pipeline returned"))?
                }
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        state.device.destroy_pipeline(pipeline, None);
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(state)
    }
}

fn shader(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    // SAFETY: shipped SPIR-V; Vulkan copies the words.
    Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }?)
}
