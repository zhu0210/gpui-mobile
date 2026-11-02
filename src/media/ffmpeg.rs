//! FFmpeg as a playback source: FFmpeg demuxes, and video decodes through the
//! platform decoder:
//!
//! - Android: FFmpeg's MediaCodec decoders (`h264_mediacodec`, …) render
//!   straight into the same `AImageReader` as the platform source, so frames
//!   reach the GPU through the same AHardwareBuffer import. Audio decodes in
//!   software into the AAudio stream.
//! - iOS: FFmpeg's VideoToolbox hwaccel hands out BGRA pixel buffers, which
//!   take the same Metal import as `AVPlayer`'s frames.
//!
//! When the platform decoder can't take the stream, FFmpeg decodes in
//! software and the planes are uploaded and converted on the GPU
//! ([`super::planes`]).
//!
//! Each playback thread opens the input itself ([`super::input`]: a file, a
//! Rust reader or Rust-fed HLS), like the platform source's extractors, so the
//! threads share nothing but the clock.

use super::player::{AudioOutput, AudioSource, VideoOutput, VideoSource};
#[cfg(any(target_os = "ios", target_os = "linux"))]
use super::player::{play_audio, play_video, AudioSink, Drift, Shared};
#[cfg(any(target_os = "ios", target_os = "linux"))]
use gpui::{DevicePixels, ImageId, Size};
#[cfg(any(target_os = "ios", target_os = "linux"))]
use std::cell::{Cell, RefCell};
#[cfg(any(target_os = "ios", target_os = "linux"))]
use std::time::{Duration, Instant};
use super::input::{Opened, Origin};
use super::planes::Picture;
use super::Track;
use anyhow::{anyhow, bail, Context as _, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg::ffi;
use ffmpeg::format::{Pixel, Sample};
#[cfg(target_os = "android")]
use ndk_sys::ANativeWindow;
use std::sync::{Arc, Mutex, Once, PoisonError};

// `libavcodec/mediacodec.h`, which ffmpeg-sys-next doesn't bind.
#[cfg(target_os = "android")]
extern "C" {
    fn av_mediacodec_release_buffer(buffer: *mut std::ffi::c_void, render: std::ffi::c_int) -> std::ffi::c_int;
}

/// Microseconds, FFmpeg's `AV_TIME_BASE`.
const MICROS: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000);

pub(super) fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Err(error) = ffmpeg::init() {
            log::error!("video: FFmpeg init failed: {error}");
        }
    });
}

fn open_input(origin: &Origin) -> Result<Opened> {
    super::input::open(origin).with_context(|| format!("FFmpeg opening {origin:?}"))
}

/// What the video stream needs before playback starts.
pub(super) struct VideoInfo {
    pub width: i32,
    pub height: i32,
    /// Clockwise degrees to turn the picture for display.
    pub rotation: i32,
    pub duration_us: i64,
    /// The audio stream played unless another is chosen.
    pub audio_track: Option<usize>,
    pub audio_tracks: Vec<Track>,
    pub subtitle_tracks: Vec<Track>,
}

pub(super) fn probe(origin: &Origin) -> Result<VideoInfo> {
    let input = open_input(origin)?;
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow!("no video stream"))?;
    let parameters = stream.parameters();
    // SAFETY: the parameters belong to the open input.
    let (width, height, rotation) = unsafe {
        let raw = &*parameters.as_ptr();
        (raw.width, raw.height, rotation(raw))
    };
    if width <= 0 || height <= 0 {
        bail!("video stream without a size");
    }
    let duration_us = if input.duration() > 0 {
        input.duration()
    } else {
        stream.duration().max(0).rescale(stream.time_base(), MICROS)
    };
    Ok(VideoInfo {
        width,
        height,
        rotation,
        duration_us,
        audio_track: input.streams().best(ffmpeg::media::Type::Audio).map(|stream| stream.index()),
        audio_tracks: tracks(&input, ffmpeg::media::Type::Audio),
        subtitle_tracks: tracks(&input, ffmpeg::media::Type::Subtitle),
    })
}

/// The input's streams of `kind`, text ones only for subtitles.
fn tracks(input: &ffmpeg::format::context::Input, kind: ffmpeg::media::Type) -> Vec<Track> {
    input
        .streams()
        .filter(|stream| stream.parameters().medium() == kind)
        .filter_map(|stream| {
            let codec = stream.parameters().id();
            // SAFETY: a plain lookup; a known id has a descriptor.
            let text = unsafe {
                let descriptor = ffi::avcodec_descriptor_get(codec.into());
                !descriptor.is_null() && (*descriptor).props & ffi::AV_CODEC_PROP_TEXT_SUB as i32 != 0
            };
            if kind == ffmpeg::media::Type::Subtitle && !text {
                return None;
            }
            let metadata = stream.metadata();
            let tag = |key| metadata.get(key).filter(|value| !value.is_empty()).map(str::to_owned);
            Some(Track {
                id: stream.index(),
                language: tag("language").filter(|language| language != "und"),
                // MP4 keeps the title as `name`.
                title: tag("title").or_else(|| tag("name")),
                codec: codec.name().to_owned(),
            })
        })
        .collect()
}

/// A subtitle cue, in media time.
#[derive(Clone, Debug)]
pub(super) struct Cue {
    pub start_us: i64,
    pub end_us: i64,
    pub text: String,
}

/// What the UI chose, and what the playback threads found, shared between
/// them.
#[derive(Default)]
pub(super) struct Choices {
    /// The audio stream, from the next seek on.
    pub audio: Option<usize>,
    /// The subtitle stream, from the next seek on.
    pub subtitle: Option<usize>,
    /// The chosen subtitle stream's cues demuxed since the last seek.
    pub cues: Vec<Cue>,
    /// The furthest media time demuxed since the last seek.
    pub demuxed_us: i64,
    /// Software-decoded frames taller than this are scaled down to it.
    pub max_height: Option<u32>,
}

/// The track, subtitle and size choices of an FFmpeg player, on every
/// platform.
pub(super) struct Controls {
    choices: Arc<Mutex<Choices>>,
    shared: Arc<super::player::Shared>,
    audio_tracks: Vec<Track>,
    subtitle_tracks: Vec<Track>,
    /// Whether the input is a local file, all of it at hand.
    local: bool,
    duration_us: i64,
}

impl Controls {
    pub fn new(origin: &Origin, info: &VideoInfo, choices: Arc<Mutex<Choices>>, shared: Arc<super::player::Shared>) -> Self {
        Self {
            choices,
            shared,
            audio_tracks: info.audio_tracks.clone(),
            subtitle_tracks: info.subtitle_tracks.clone(),
            local: matches!(origin, Origin::File(_)),
            duration_us: info.duration_us,
        }
    }

    fn choices(&self) -> std::sync::MutexGuard<'_, Choices> {
        self.choices.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Seeks to where playback is, so that the playback threads pick up a
    /// new choice from there.
    fn refresh(&self) {
        self.shared.seek(self.shared.position_us().clamp(0, self.duration_us));
    }

    pub fn audio_tracks(&self) -> &[Track] {
        &self.audio_tracks
    }

    pub fn audio_track(&self) -> Option<usize> {
        self.choices().audio
    }

    pub fn set_audio_track(&self, id: usize) {
        if !self.audio_tracks.iter().any(|track| track.id == id) || self.choices().audio == Some(id) {
            return;
        }
        self.choices().audio = Some(id);
        self.refresh();
    }

    pub fn subtitle_tracks(&self) -> &[Track] {
        &self.subtitle_tracks
    }

    pub fn subtitle_track(&self) -> Option<usize> {
        self.choices().subtitle
    }

    pub fn set_subtitle_track(&self, id: Option<usize>) {
        let id = id.filter(|id| self.subtitle_tracks.iter().any(|track| track.id == *id));
        if self.choices().subtitle == id {
            return;
        }
        let mut choices = self.choices();
        choices.subtitle = id;
        choices.cues.clear();
        drop(choices);
        // From the seek on, the cue showing now is demuxed again.
        self.refresh();
    }

    pub fn subtitle_text(&self) -> Option<String> {
        let position = self.shared.position_us();
        self.choices().text_at(position)
    }

    pub fn buffered_us(&self) -> i64 {
        if self.local {
            return self.duration_us;
        }
        self.choices().demuxed_us.max(self.shared.position_us()).clamp(0, self.duration_us)
    }

    pub fn set_max_height(&self, height: Option<u32>) {
        self.choices().max_height = height;
    }
}

/// Cues kept at most; older ones go first.
const MAX_CUES: usize = 64;

impl Choices {
    pub fn new(info: &VideoInfo) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            audio: info.audio_track,
            ..Self::default()
        }))
    }

    /// The text of the cues showing at `position_us`, one per line.
    pub fn text_at(&mut self, position_us: i64) -> Option<String> {
        self.cues.retain(|cue| cue.end_us > position_us || cue.start_us > position_us);
        let lines: Vec<&str> = self
            .cues
            .iter()
            .filter(|cue| cue.start_us <= position_us)
            .map(|cue| cue.text.as_str())
            .collect();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }
}

/// An ASS dialogue line as FFmpeg's text decoders give it
/// (`ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`), as
/// plain text: override blocks dropped, `\N` as a line break.
fn ass_text(line: &str) -> String {
    let text = line.splitn(9, ',').nth(8).unwrap_or(line);
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                for c in chars.by_ref() {
                    if c == '}' {
                        break;
                    }
                }
            }
            '\\' => match chars.peek() {
                Some('N' | 'n') => {
                    chars.next();
                    plain.push('\n');
                }
                Some('h') => {
                    chars.next();
                    plain.push(' ');
                }
                _ => plain.push(c),
            },
            c => plain.push(c),
        }
    }
    plain.trim().to_owned()
}

/// The display matrix's rotation, as clockwise degrees in steps of 90.
///
/// # Safety
/// `parameters` must be valid codec parameters.
unsafe fn rotation(parameters: &ffi::AVCodecParameters) -> i32 {
    // SAFETY: the side data array belongs to the parameters.
    unsafe {
        let data = ffi::av_packet_side_data_get(
            parameters.coded_side_data,
            parameters.nb_coded_side_data,
            ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
        );
        if data.is_null() || (*data).size < 36 {
            return 0;
        }
        // FFmpeg gives the counter-clockwise angle.
        let degrees = -ffi::av_display_rotation_get((*data).data as *const i32);
        if degrees.is_nan() {
            return 0;
        }
        ((degrees / 90.0).round() as i32 * 90).rem_euclid(360)
    }
}

/// Reads packets of one stream and feeds them to a decoder; shared by the
/// video and audio sources.
struct Demuxer {
    input: Opened,
    stream: usize,
    time_base: ffmpeg::Rational,
    /// Where the input's timestamps start (1.4 s in MPEG-TS, so in HLS);
    /// media time 0 is there.
    start_us: i64,
    /// A packet the decoder had no room for yet.
    pending: Option<ffmpeg::Packet>,
    input_done: bool,
    choices: Arc<Mutex<Choices>>,
    /// The chosen subtitle stream, decoded on the side (video only).
    subtitles: Option<(usize, ffmpeg::decoder::Subtitle)>,
    /// Whether streams nobody reads are skipped.
    discard: bool,
}

impl Demuxer {
    /// Opens the input on stream `index`, or the best of `kind`.
    fn open(
        origin: &Origin,
        kind: ffmpeg::media::Type,
        index: Option<usize>,
        choices: Arc<Mutex<Choices>>,
    ) -> Result<(Self, ffmpeg::codec::Parameters)> {
        let input = open_input(origin)?;
        let index = match index {
            Some(index) => index,
            None => input.streams().best(kind).ok_or_else(|| anyhow!("no {kind:?} stream"))?.index(),
        };
        // SAFETY: reads a field of the open input.
        let start_us = match unsafe { (*input.as_ptr()).start_time } {
            ffi::AV_NOPTS_VALUE => 0,
            start => start,
        };
        let mut demuxer = Self {
            input,
            stream: index,
            time_base: MICROS,
            start_us,
            pending: None,
            input_done: false,
            choices,
            subtitles: None,
            // ponytail: FFmpeg's `hls` demuxer drops the timestamps of the
            // streams left when others in their segments are discarded; skip
            // whole variant playlists once multi-variant HLS matters.
            discard: !matches!(origin, Origin::Hls { .. }),
        };
        let parameters = demuxer.switch(index)?;
        Ok((demuxer, parameters))
    }

    /// Reads stream `index` from now on, and with it the chosen subtitles
    /// when they are decoded here; returns the stream's parameters.
    fn switch(&mut self, index: usize) -> Result<ffmpeg::codec::Parameters> {
        let stream = self.input.stream(index).ok_or_else(|| anyhow!("no stream {index}"))?;
        (self.stream, self.time_base) = (index, stream.time_base());
        let parameters = stream.parameters();
        self.discard_others();
        Ok(parameters)
    }

    /// Skips the streams nobody here reads: FFmpeg doesn't hand out their
    /// packets, and HLS doesn't fetch their playlists.
    fn discard_others(&mut self) {
        if !self.discard {
            return;
        }
        let keep = [Some(self.stream), self.subtitles.as_ref().map(|(index, _)| *index)];
        // SAFETY: the streams belong to the open input.
        unsafe {
            let raw = &*self.input.as_ptr();
            for index in 0..raw.nb_streams as usize {
                (**raw.streams.add(index)).discard = if keep.contains(&Some(index)) {
                    ffi::AVDiscard::AVDISCARD_DEFAULT
                } else {
                    ffi::AVDiscard::AVDISCARD_ALL
                };
            }
        }
    }

    /// Decodes the subtitle stream chosen since the last call (on the video
    /// thread, from the packets it reads anyway).
    fn choose_subtitles(&mut self) -> Result<()> {
        let wanted = self.choices.lock().unwrap_or_else(PoisonError::into_inner).subtitle;
        if wanted == self.subtitles.as_ref().map(|(index, _)| *index) {
            return Ok(());
        }
        self.subtitles = match wanted {
            Some(index) => {
                let stream = self.input.stream(index).ok_or_else(|| anyhow!("no stream {index}"))?;
                let mut context = ffmpeg::codec::Context::from_parameters(stream.parameters())?;
                // SAFETY: the context is not open yet.
                unsafe { (*context.as_mut_ptr()).pkt_timebase = stream.time_base().into() };
                Some((index, context.decoder().subtitle()?))
            }
            None => None,
        };
        self.discard_others();
        Ok(())
    }

    /// Decodes a subtitle packet into a cue.
    fn add_cue(&mut self, packet: &ffmpeg::Packet) {
        let Some((index, decoder)) = &mut self.subtitles else { return };
        if packet.stream() != *index {
            return;
        }
        let mut subtitle = ffmpeg::Subtitle::new();
        match decoder.decode(packet, &mut subtitle) {
            Ok(true) => {}
            Ok(false) => return,
            Err(error) => {
                log::warn!("video: subtitle packet skipped: {error}");
                return;
            }
        }
        let text: Vec<String> = subtitle
            .rects()
            .filter_map(|rect| match rect {
                ffmpeg::codec::subtitle::Rect::Ass(ass) => Some(ass_text(ass.get())),
                ffmpeg::codec::subtitle::Rect::Text(text) => Some(text.get().trim().to_owned()),
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .collect();
        if text.is_empty() {
            return;
        }
        let Some(time_base) = self.input.stream(*index).map(|stream| stream.time_base()) else { return };
        let start_us = packet.pts().map_or(0, |pts| pts.rescale(time_base, MICROS) - self.start_us)
            + subtitle.start() as i64 * 1000;
        let end_us = match subtitle.end() {
            // Until the next cue, as players show it: a few seconds.
            0 | u32::MAX if packet.duration() <= 0 => start_us + 5_000_000,
            0 | u32::MAX => start_us + packet.duration().rescale(time_base, MICROS),
            end => start_us - subtitle.start() as i64 * 1000 + end as i64 * 1000,
        };
        let mut choices = self.choices.lock().unwrap_or_else(PoisonError::into_inner);
        if choices.cues.len() == MAX_CUES {
            choices.cues.remove(0);
        }
        choices.cues.push(Cue {
            start_us,
            end_us,
            text: text.join("\n"),
        });
    }

    /// Notes that the input has data up to `packet`.
    fn demuxed(&self, packet: &ffmpeg::Packet) {
        let Some(time_base) = self.input.stream(packet.stream()).map(|stream| stream.time_base()) else { return };
        if let Some(ts) = packet.pts().or(packet.dts()) {
            let us = ts.rescale(time_base, MICROS) - self.start_us;
            let mut choices = self.choices.lock().unwrap_or_else(PoisonError::into_inner);
            choices.demuxed_us = choices.demuxed_us.max(us);
        }
    }

    /// Gives the decoder the next packet, if it takes one.
    fn feed(&mut self, decoder: &mut ffmpeg::decoder::Opened) -> Result<()> {
        if self.input_done {
            return Ok(());
        }
        let packet = match self.pending.take() {
            Some(packet) => packet,
            None => loop {
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) if packet.stream() == self.stream => {
                        self.demuxed(&packet);
                        break packet;
                    }
                    Ok(()) => self.add_cue(&packet),
                    Err(ffmpeg::Error::Eof) => {
                        decoder.send_eof()?;
                        self.input_done = true;
                        return Ok(());
                    }
                    Err(error) => return Err(error).context("reading a packet"),
                }
            },
        };
        match decoder.send_packet(&packet) {
            Ok(()) => Ok(()),
            Err(ffmpeg::Error::Other { errno: libc::EAGAIN }) => {
                self.pending = Some(packet);
                Ok(())
            }
            // A damaged packet: skip it, as players do.
            Err(ffmpeg::Error::InvalidData) => Ok(()),
            Err(error) => Err(error).context("decoding a packet"),
        }
    }

    fn seek(&mut self, target_us: i64, decoder: &mut ffmpeg::decoder::Opened) -> Result<()> {
        // The keyframe at or before the target; the playback loop drops what
        // comes before the target.
        let target = target_us.saturating_add(self.start_us);
        self.input.seek(target, ..target.saturating_add(1)).context("seeking")?;
        {
            let mut choices = self.choices.lock().unwrap_or_else(PoisonError::into_inner);
            choices.demuxed_us = target_us;
            if self.subtitles.is_some() {
                choices.cues.clear();
            }
        }
        if let Some((_, subtitles)) = &mut self.subtitles {
            subtitles.flush();
        }
        decoder.flush();
        self.pending = None;
        self.input_done = false;
        Ok(())
    }

    fn micros(&self, timestamp: Option<i64>) -> i64 {
        timestamp.map_or(0, |ts| ts.rescale(self.time_base, MICROS) - self.start_us)
    }
}

/// Where the platform decoder puts its frames: the reader's window on
/// Android; VideoToolbox needs nothing; VA-API runs on a render node (the GPU
/// the window renders with).
#[cfg(target_os = "android")]
pub(super) type HardwareTarget = *mut ANativeWindow;
#[cfg(target_os = "ios")]
pub(super) type HardwareTarget = ();
#[cfg(target_os = "linux")]
pub(super) type HardwareTarget = std::path::PathBuf;

/// A frame FFmpeg decoded.
pub(super) enum Frame {
    /// A MediaCodec output buffer, which renders into the reader.
    #[cfg(target_os = "android")]
    MediaCodec(ffmpeg::frame::Video),
    /// A frame for the main thread: 8-bit 4:2:0 planes, or a VideoToolbox
    /// pixel buffer (iOS) or VA-API surface (Linux).
    Picture(ffmpeg::frame::Video),
}

pub(super) struct FfmpegVideo {
    demuxer: Demuxer,
    decoder: ffmpeg::decoder::Video,
    /// Converts software frames that aren't 8-bit 4:2:0 to `yuv420p`.
    scaler: Option<ffmpeg::software::scaling::Context>,
    /// Where pictures go to be imported on the main thread.
    slot: Arc<Mutex<Option<Picture>>>,
}

// SAFETY: the source moves to its playback thread and is only used there; the
// window it renders into outlives the decoder (see `Decoder::open`).
unsafe impl Send for FfmpegVideo {}

impl FfmpegVideo {
    /// Opens the video stream, decoding with the platform decoder when
    /// `hardware` is given and the decoder takes the stream, in software
    /// otherwise. Returns the source and whether it decodes in hardware.
    pub fn open(
        origin: &Origin,
        hardware: Option<HardwareTarget>,
        slot: Arc<Mutex<Option<Picture>>>,
        choices: Arc<Mutex<Choices>>,
    ) -> Result<(Self, bool)> {
        let (demuxer, parameters) = Demuxer::open(origin, ffmpeg::media::Type::Video, None, choices)?;
        let hardware = match hardware {
            Some(target) => match open_hardware(&parameters, demuxer.time_base, target) {
                Ok(decoder) => Some(decoder),
                Err(error) => {
                    log::warn!("video: FFmpeg hardware decoding unavailable, decoding in software: {error:#}");
                    None
                }
            },
            None => None,
        };
        let is_hardware = hardware.is_some();
        let decoder = match hardware {
            Some(decoder) => decoder,
            None => open_software(&parameters, demuxer.time_base)?,
        };
        log::info!(
            "video: FFmpeg {} demuxer, {} decoder ({})",
            // SAFETY: the input's format is set once it is open.
            unsafe { std::ffi::CStr::from_ptr((*(*demuxer.input.as_ptr()).iformat).name) }.to_string_lossy(),
            decoder.codec().map_or_else(|| "?".into(), |codec| codec.name().to_owned()),
            match is_hardware {
                true if cfg!(target_os = "android") => "MediaCodec into the AImageReader",
                true if cfg!(target_os = "ios") => "VideoToolbox into BGRA pixel buffers",
                true => "VA-API, surfaces imported as DMA-BUFs",
                false => "software, planes uploaded",
            },
        );
        Ok((
            Self {
                demuxer,
                decoder,
                scaler: None,
                slot,
            },
            is_hardware,
        ))
    }

    /// The frame as 8-bit 4:2:0, no taller than the chosen maximum height.
    fn yuv420p(&mut self, frame: ffmpeg::frame::Video) -> Result<ffmpeg::frame::Video> {
        let max_height = self.demuxer.choices.lock().unwrap_or_else(PoisonError::into_inner).max_height;
        let (width, height) = match max_height {
            Some(max) if frame.height() > max.max(2) => {
                let max = max.max(2) & !1;
                ((frame.width() as u64 * max as u64 / frame.height() as u64) as u32 & !1, max)
            }
            _ => (frame.width(), frame.height()),
        };
        if frame.format() == Pixel::YUV420P && height == frame.height() {
            return Ok(frame);
        }
        // ponytail: 10-bit and 4:4:4 go through swscale to 8-bit 4:2:0; upload
        // them natively if the software path ever matters for them.
        let scaler = match &mut self.scaler {
            Some(scaler)
                if scaler.input().format == frame.format()
                    && (scaler.input().width, scaler.input().height) == (frame.width(), frame.height())
                    && (scaler.output().width, scaler.output().height) == (width, height) =>
            {
                scaler
            }
            scaler => scaler.insert(ffmpeg::software::scaling::Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                Pixel::YUV420P,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            )?),
        };
        let mut converted = ffmpeg::frame::Video::empty();
        scaler.run(&frame, &mut converted)?;
        converted.set_pts(frame.pts());
        // SAFETY: both frames are live; only colour metadata is copied.
        unsafe {
            (*converted.as_mut_ptr()).colorspace = (*frame.as_ptr()).colorspace;
            (*converted.as_mut_ptr()).color_range = (*frame.as_ptr()).color_range;
        }
        Ok(converted)
    }
}

/// Opens FFmpeg's MediaCodec decoder for the stream, rendering into `window`.
#[cfg(target_os = "android")]
fn open_hardware(
    parameters: &ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    window: *mut ANativeWindow,
) -> Result<ffmpeg::decoder::Video> {
    use ffmpeg::codec::Id;
    let name = match parameters.id() {
        Id::H264 => "h264_mediacodec",
        Id::HEVC => "hevc_mediacodec",
        Id::VP9 => "vp9_mediacodec",
        Id::AV1 => "av1_mediacodec",
        id => bail!("no MediaCodec decoder for {id:?}"),
    };
    let codec = ffmpeg::decoder::find_by_name(name).ok_or_else(|| anyhow!("FFmpeg built without {name}"))?;
    let mut context = ffmpeg::codec::Context::new_with_codec(codec);
    context.set_parameters(parameters.clone())?;
    // SAFETY: the context is not open yet; the device context is handed to it
    // (it keeps its own reference) and unreferenced here either way.
    unsafe {
        let raw = context.as_mut_ptr();
        // MediaCodec takes microseconds; FFmpeg converts with this.
        (*raw).pkt_timebase = time_base.into();
        let mut device = ffi::av_hwdevice_ctx_alloc(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_MEDIACODEC);
        if device.is_null() {
            bail!("av_hwdevice_ctx_alloc(MEDIACODEC) failed");
        }
        let hwctx = (*((*device).data as *mut ffi::AVHWDeviceContext)).hwctx as *mut ffi::AVMediaCodecDeviceContext;
        (*hwctx).native_window = window.cast();
        let ret = ffi::av_hwdevice_ctx_init(device);
        if ret >= 0 {
            (*raw).hw_device_ctx = ffi::av_buffer_ref(device);
            (*raw).get_format = Some(pick_mediacodec);
        }
        ffi::av_buffer_unref(&mut device);
        if ret < 0 {
            bail!("av_hwdevice_ctx_init(MEDIACODEC): {}", ffmpeg::Error::from(ret));
        }
    }
    let mut options = ffmpeg::Dictionary::new();
    // MediaCodec through the NDK: no JVM needed.
    options.set("ndk_codec", "1");
    Ok(context.decoder().open_as_with(codec, options)?.video()?)
}

#[cfg(target_os = "android")]
extern "C" fn pick_mediacodec(
    _context: *mut ffi::AVCodecContext,
    mut formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: FFmpeg passes a list ending in AV_PIX_FMT_NONE.
    unsafe {
        while *formats != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *formats == ffi::AVPixelFormat::AV_PIX_FMT_MEDIACODEC {
                return *formats;
            }
            formats = formats.add(1);
        }
    }
    ffi::AVPixelFormat::AV_PIX_FMT_NONE
}

/// Opens the native decoder with FFmpeg's VideoToolbox hwaccel.
#[cfg(target_os = "ios")]
fn open_hardware(
    parameters: &ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    _target: HardwareTarget,
) -> Result<ffmpeg::decoder::Video> {
    use ffmpeg::codec::Id;
    // FFmpeg requires VideoToolbox's hardware decoder (the simulator has none).
    let codec_type = match parameters.id() {
        Id::H264 => *b"avc1",
        Id::HEVC => *b"hvc1",
        Id::VP9 => *b"vp09",
        Id::AV1 => *b"av01",
        id => bail!("no VideoToolbox decoder for {id:?}"),
    };
    // SAFETY: a plain query.
    if !unsafe { VTIsHardwareDecodeSupported(u32::from_be_bytes(codec_type)) } {
        bail!("no VideoToolbox hardware decoder for {:?}", parameters.id());
    }
    let mut context = ffmpeg::codec::Context::from_parameters(parameters.clone())?;
    // SAFETY: the context is not open yet and takes the device reference.
    unsafe {
        let raw = context.as_mut_ptr();
        (*raw).pkt_timebase = time_base.into();
        let mut device = std::ptr::null_mut();
        let ret = ffi::av_hwdevice_ctx_create(
            &mut device,
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        if ret < 0 {
            bail!("av_hwdevice_ctx_create(VIDEOTOOLBOX): {}", ffmpeg::Error::from(ret));
        }
        (*raw).hw_device_ctx = device;
        (*raw).get_format = Some(pick_videotoolbox);
    }
    Ok(context.decoder().video()?)
}

#[cfg(target_os = "ios")]
#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTIsHardwareDecodeSupported(codec_type: u32) -> bool;
}

/// Picks VideoToolbox, with BGRA frames so they take the same Metal import as
/// `AVPlayer`'s (VideoToolbox converts on the GPU); software when that fails.
#[cfg(target_os = "ios")]
extern "C" fn pick_videotoolbox(
    context: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: FFmpeg passes its open context and a list ending in
    // AV_PIX_FMT_NONE; the frames context replaces the context's own.
    unsafe {
        let mut format = formats;
        while *format != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *format == ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX {
                let mut frames = std::ptr::null_mut();
                if ffi::avcodec_get_hw_frames_parameters(context, (*context).hw_device_ctx, *format, &mut frames) >= 0 {
                    (*((*frames).data as *mut ffi::AVHWFramesContext)).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_BGRA;
                    if ffi::av_hwframe_ctx_init(frames) >= 0 {
                        ffi::av_buffer_unref(&mut (*context).hw_frames_ctx);
                        (*context).hw_frames_ctx = frames;
                        // FFmpeg maps BGRA to VideoToolbox only as full range
                        // (RGB has no video range) and takes the range from
                        // the context; VideoToolbox still reads the stream's
                        // own range from the bitstream.
                        (*context).color_range = ffi::AVColorRange::AVCOL_RANGE_JPEG;
                        return *format;
                    }
                    ffi::av_buffer_unref(&mut frames);
                }
                log::warn!("video: VideoToolbox BGRA frames unavailable, decoding in software");
                break;
            }
            format = format.add(1);
        }
        first_software(formats)
    }
}

/// The first software format in FFmpeg's list. Not `avcodec_default_get_format`:
/// with a device set it picks the hardware format again.
///
/// # Safety
/// `formats` must end in AV_PIX_FMT_NONE.
#[cfg(any(target_os = "ios", target_os = "linux"))]
unsafe fn first_software(mut formats: *const ffi::AVPixelFormat) -> ffi::AVPixelFormat {
    // SAFETY: the caller vouches for the list.
    unsafe {
        while *formats != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            let descriptor = ffi::av_pix_fmt_desc_get(*formats);
            if !descriptor.is_null() && (*descriptor).flags & ffi::AV_PIX_FMT_FLAG_HWACCEL as u64 == 0 {
                return *formats;
            }
            formats = formats.add(1);
        }
    }
    ffi::AVPixelFormat::AV_PIX_FMT_NONE
}

/// Opens the native decoder with FFmpeg's VA-API hwaccel on `render_node`.
#[cfg(target_os = "linux")]
fn open_hardware(
    parameters: &ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    render_node: HardwareTarget,
) -> Result<ffmpeg::decoder::Video> {
    use std::os::unix::ffi::OsStrExt as _;
    let node = std::ffi::CString::new(render_node.as_os_str().as_bytes())?;
    let mut context = ffmpeg::codec::Context::from_parameters(parameters.clone())?;
    // SAFETY: the context is not open yet and takes the device reference.
    unsafe {
        let raw = context.as_mut_ptr();
        (*raw).pkt_timebase = time_base.into();
        let mut device = std::ptr::null_mut();
        let ret = ffi::av_hwdevice_ctx_create(
            &mut device,
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            node.as_ptr(),
            std::ptr::null_mut(),
            0,
        );
        if ret < 0 {
            bail!("av_hwdevice_ctx_create(VAAPI, {render_node:?}): {}", ffmpeg::Error::from(ret));
        }
        (*raw).hw_device_ctx = device;
        (*raw).get_format = Some(pick_vaapi);
    }
    Ok(context.decoder().video()?)
}

/// Picks VA-API when FFmpeg offers it (it does when the GPU decodes the
/// stream's profile), software otherwise.
#[cfg(target_os = "linux")]
extern "C" fn pick_vaapi(
    _context: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: FFmpeg passes a list ending in AV_PIX_FMT_NONE.
    unsafe {
        let mut format = formats;
        while *format != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *format == ffi::AVPixelFormat::AV_PIX_FMT_VAAPI {
                return *format;
            }
            format = format.add(1);
        }
        first_software(formats)
    }
}

fn open_software(parameters: &ffmpeg::codec::Parameters, time_base: ffmpeg::Rational) -> Result<ffmpeg::decoder::Video> {
    let mut context = ffmpeg::codec::Context::from_parameters(parameters.clone())?;
    // SAFETY: the context is not open yet.
    unsafe { (*context.as_mut_ptr()).pkt_timebase = time_base.into() };
    context.set_threading(ffmpeg::threading::Config {
        kind: ffmpeg::threading::Type::Frame,
        count: 0,
    });
    Ok(context.decoder().video()?)
}

impl VideoSource for FfmpegVideo {
    type Frame = Frame;

    fn seek(&mut self, target_us: i64) -> Result<()> {
        self.demuxer.choose_subtitles()?;
        self.demuxer.seek(target_us, &mut self.decoder)
    }

    fn next(&mut self) -> Result<VideoOutput<Frame>> {
        let mut frame = ffmpeg::frame::Video::empty();
        match self.decoder.receive_frame(&mut frame) {
            Ok(()) => {
                let pts = self.demuxer.micros(frame.timestamp());
                let frame = match frame.format() {
                    #[cfg(target_os = "android")]
                    Pixel::MEDIACODEC => Frame::MediaCodec(frame),
                    #[cfg(target_os = "ios")]
                    Pixel::VIDEOTOOLBOX => Frame::Picture(frame),
                    #[cfg(target_os = "linux")]
                    Pixel::VAAPI => Frame::Picture(frame),
                    _ => Frame::Picture(self.yuv420p(frame)?),
                };
                Ok(VideoOutput::Frame(pts, frame))
            }
            Err(ffmpeg::Error::Eof) => Ok(VideoOutput::End),
            Err(ffmpeg::Error::Other { errno: libc::EAGAIN }) => {
                self.demuxer.feed(&mut self.decoder)?;
                Ok(VideoOutput::Pending)
            }
            Err(error) => Err(error).context("decoding video"),
        }
    }

    fn release(&mut self, frame: Frame, render: bool) -> Result<()> {
        match frame {
            #[cfg(target_os = "android")]
            Frame::MediaCodec(frame) => {
                // SAFETY: a MEDIACODEC frame's data[3] is its codec buffer,
                // released once here (dropping the frame then does nothing).
                let ret = unsafe {
                    av_mediacodec_release_buffer((*frame.as_ptr()).data[3].cast(), render as i32)
                };
                if ret < 0 {
                    bail!("av_mediacodec_release_buffer: {}", ffmpeg::Error::from(ret));
                }
            }
            Frame::Picture(frame) if render => {
                let pts = self.demuxer.micros(frame.timestamp());
                *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(Picture::new(frame, pts));
            }
            Frame::Picture(_) => {}
        }
        Ok(())
    }
}

pub(super) struct FfmpegAudio {
    demuxer: Demuxer,
    decoder: ffmpeg::decoder::Audio,
    resampler: Option<ffmpeg::software::resampling::Context>,
    /// The current output: interleaved 16-bit samples and their channel count.
    samples: Vec<i16>,
    /// Whether a track switch changed the format, to report it next.
    switched: bool,
}

// SAFETY: the source moves to its playback thread and is only used there.
unsafe impl Send for FfmpegAudio {}

impl FfmpegAudio {
    /// Opens the chosen audio stream, or the best one.
    pub fn open(origin: &Origin, choices: Arc<Mutex<Choices>>) -> Result<Self> {
        let index = choices.lock().unwrap_or_else(PoisonError::into_inner).audio;
        let (demuxer, parameters) = Demuxer::open(origin, ffmpeg::media::Type::Audio, index, choices)?;
        let decoder = open_audio(parameters, demuxer.time_base)?;
        Ok(Self {
            demuxer,
            decoder,
            resampler: None,
            samples: Vec::new(),
            switched: false,
        })
    }

    /// Every output plays mono and stereo; more channels are downmixed.
    fn output_channels(&self) -> i32 {
        (self.decoder.channels() as i32).clamp(1, 2)
    }
}

fn open_audio(parameters: ffmpeg::codec::Parameters, time_base: ffmpeg::Rational) -> Result<ffmpeg::decoder::Audio> {
    let mut context = ffmpeg::codec::Context::from_parameters(parameters)?;
    // SAFETY: the context is not open yet.
    unsafe { (*context.as_mut_ptr()).pkt_timebase = time_base.into() };
    let decoder = context.decoder().audio()?;
    log::info!(
        "video: FFmpeg audio {} decoder, {} Hz, {} channels",
        decoder.codec().map_or_else(|| "?".into(), |codec| codec.name().to_owned()),
        decoder.rate(),
        decoder.channels(),
    );
    Ok(decoder)
}

impl AudioSource for FfmpegAudio {
    fn format(&self) -> Result<(i32, i32)> {
        Ok((self.decoder.rate() as i32, self.output_channels()))
    }

    fn seek(&mut self, target_us: i64) -> Result<()> {
        let wanted = self.demuxer.choices.lock().unwrap_or_else(PoisonError::into_inner).audio;
        if let Some(index) = wanted.filter(|&index| index != self.demuxer.stream) {
            let parameters = self.demuxer.switch(index)?;
            self.decoder = open_audio(parameters, self.demuxer.time_base)?;
            (self.resampler, self.switched) = (None, true);
        }
        self.demuxer.seek(target_us, &mut self.decoder)
    }

    fn next(&mut self) -> Result<AudioOutput> {
        if std::mem::take(&mut self.switched) {
            return Ok(AudioOutput::Format(self.format()?));
        }
        let mut frame = ffmpeg::frame::Audio::empty();
        match self.decoder.receive_frame(&mut frame) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => return Ok(AudioOutput::End),
            Err(ffmpeg::Error::Other { errno: libc::EAGAIN }) => {
                self.demuxer.feed(&mut self.decoder)?;
                return Ok(AudioOutput::Pending);
            }
            Err(error) => return Err(error).context("decoding audio"),
        }
        let pts = self.demuxer.micros(frame.timestamp());
        let channels = self.output_channels();
        let resampler = match &mut self.resampler {
            Some(resampler) if resampler.input().format == frame.format() => resampler,
            resampler => resampler.insert(ffmpeg::software::resampling::Context::get(
                frame.format(),
                frame.channel_layout(),
                frame.rate(),
                Sample::I16(ffmpeg::format::sample::Type::Packed),
                ffmpeg::ChannelLayout::default(channels),
                self.decoder.rate(),
            )?),
        };
        let mut output = ffmpeg::frame::Audio::empty();
        resampler.run(&frame, &mut output)?;
        let values = output.samples() * channels as usize;
        let bytes = &output.data(0)[..values * 2];
        self.samples.clear();
        self.samples
            .extend(bytes.as_chunks::<2>().0.iter().map(|pair| i16::from_ne_bytes(*pair)));
        Ok(AudioOutput::Samples {
            pts,
            data: self.samples.as_ptr(),
            frames: output.samples(),
        })
    }

    fn release(&mut self) -> Result<()> {
        Ok(())
    }
}

trait Rescale {
    fn rescale(self, from: ffmpeg::Rational, to: ffmpeg::Rational) -> i64;
}

impl Rescale for i64 {
    fn rescale(self, from: ffmpeg::Rational, to: ffmpeg::Rational) -> i64 {
        // SAFETY: pure arithmetic.
        unsafe { ffi::av_rescale_q(self, from.into(), to.into()) }
    }
}

/// What a platform adds to [`FfmpegPlayer`]: the window's GPU, the import of
/// hardware-decoded pictures, and the audio output.
#[cfg(any(target_os = "ios", target_os = "linux"))]
pub(super) trait Platform: Default {
    type Sink: AudioSink;
    /// How [`Platform::import`] gets frames to the GPU, for the log.
    const IMPORT: &str;
    /// The context the window renders with, once there is one.
    fn context() -> Option<gpui_wgpu::WgpuContext>;
    /// Where to decode in hardware for `context`; `None` to decode in software.
    fn hardware(context: Option<&gpui_wgpu::WgpuContext>) -> Option<HardwareTarget>;
    /// Imports a hardware-decoded picture as an RGBA texture turned clockwise
    /// by `rotation`; `None` for a software one.
    fn import(
        &self,
        picture: &Picture,
        rotation: i32,
        uploader: &mut super::planes::Uploader,
        context: &gpui_wgpu::WgpuContext,
    ) -> Result<Option<Arc<gpui_wgpu::wgpu::Texture>>>;
    /// Around the audio thread (the iOS audio session).
    fn audio_started() {}
    fn audio_stopped() {}
}

/// FFmpeg playing a file on its own threads: video decoded in hardware (or in
/// software) and audio, both on the shared clock ([`super::player`]).
#[cfg(any(target_os = "ios", target_os = "linux"))]
pub(super) struct FfmpegPlayer<P: Platform> {
    platform: P,
    size: Size<DevicePixels>,
    rotation: i32,
    duration_us: i64,
    /// Whether an audio thread was started ([`Platform::audio_started`]).
    has_audio: bool,
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    slot: Arc<Mutex<Option<Picture>>>,
    controls: Controls,
    uploader: RefCell<super::planes::Uploader>,
    frames_shown: Cell<u64>,
    last_import: Cell<Option<Instant>>,
    drift: Cell<Drift>,
}

#[cfg(any(target_os = "ios", target_os = "linux"))]
impl<P: Platform> FfmpegPlayer<P> {
    pub fn open(source: super::Source, backend: super::Backend) -> Result<Self> {
        let origin = Origin::of(source)?;
        let info = probe(&origin)?;
        let hardware = match backend {
            super::Backend::FfmpegSoftware => None,
            _ => P::hardware(P::context().as_ref()),
        };
        let slot = Arc::new(Mutex::new(None));
        let choices = Choices::new(&info);
        let (mut video, _) = FfmpegVideo::open(&origin, hardware, slot.clone(), choices.clone())?;
        let audio = if info.audio_track.is_some() {
            FfmpegAudio::open(&origin, choices.clone())
                .map_err(|error| log::warn!("video: audio stream skipped: {error:#}"))
                .ok()
        } else {
            None
        };
        log::info!(
            "video: FFmpeg playing {}x{} (rotated {}°, audio {}) from {origin:?}",
            info.width,
            info.height,
            info.rotation,
            audio.is_some(),
        );
        let shared = Shared::new(audio.is_some());
        let mut threads = vec![std::thread::Builder::new().name("gpui-video".into()).spawn({
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
                    if let Err(error) = play_audio::<P::Sink>(&mut audio, &shared) {
                        log::error!("video: audio stopped: {error:#}");
                    }
                    // The video goes on by the monotonic clock.
                    shared.update(|state| {
                        state.audio = false;
                        state.holding = false;
                    });
                }
            })?);
            P::audio_started();
        }
        let (width, height) =
            if info.rotation % 180 == 90 { (info.height, info.width) } else { (info.width, info.height) };
        Ok(Self {
            platform: P::default(),
            size: Size {
                width: DevicePixels(width),
                height: DevicePixels(height),
            },
            rotation: info.rotation,
            duration_us: info.duration_us,
            has_audio: threads.len() > 1,
            controls: Controls::new(&origin, &info, choices, shared.clone()),
            shared,
            threads,
            slot,
            uploader: RefCell::new(Default::default()),
            frames_shown: Cell::new(0),
            last_import: Cell::new(None),
            drift: Cell::new(Drift::default()),
        })
    }

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

    pub fn set_rate(&self, rate: f32) {
        self.shared.set_rate(rate as f64)
    }

    pub fn rate(&self) -> f32 {
        self.shared.rate() as f32
    }

    pub fn set_volume(&self, volume: f32) {
        self.shared.set_volume(volume)
    }

    pub fn volume(&self) -> f32 {
        self.shared.volume()
    }

    pub fn set_muted(&self, muted: bool) {
        self.shared.set_muted(muted)
    }

    pub fn is_muted(&self) -> bool {
        self.shared.is_muted()
    }

    pub fn buffered(&self) -> Duration {
        Duration::from_micros(self.controls.buffered_us() as u64)
    }

    pub fn controls(&self) -> Option<&Controls> {
        Some(&self.controls)
    }

    /// Imports the newest decoded picture, if there is one, and makes it the
    /// texture drawn for `image_id`.
    pub fn update(&self, image_id: ImageId) -> Result<()> {
        let Some(picture) = self.slot.lock().unwrap_or_else(PoisonError::into_inner).take() else {
            return Ok(());
        };
        let context = P::context().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let mut uploader = self.uploader.borrow_mut();
        let (texture, path) = match self.platform.import(&picture, self.rotation, &mut uploader, &context)? {
            Some(texture) => (texture, P::IMPORT),
            None => (
                uploader.convert(&picture, self.rotation, &context.device, &context.queue)?,
                "software frame uploaded as planes, converted on the GPU",
            ),
        };
        gpui_wgpu::set_external_texture(image_id, &texture);
        self.last_import.set(Some(Instant::now()));
        let shown = self.frames_shown.get() + 1;
        self.frames_shown.set(shown);
        if shown == 1 {
            log::info!("video: {path}, {}x{}", texture.width(), texture.height());
        }
        self.shared.record_drift(&self.drift, picture.timestamp_us);
        Ok(())
    }
}

#[cfg(any(target_os = "ios", target_os = "linux"))]
impl<P: Platform> Drop for FfmpegPlayer<P> {
    fn drop(&mut self) {
        self.shared.update(|state| state.stop = true);
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                log::error!("video: playback thread panicked");
            }
        }
        if self.has_audio {
            P::audio_stopped();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ass_lines_become_plain_text() {
        assert_eq!(ass_text(r"0,0,Default,,0,0,0,,{\i1}Styled, with commas{\i0}\NNext\hrow"), "Styled, with commas\nNext row");
        assert_eq!(ass_text("3,0,Default,,0,0,0,,plain"), "plain");
    }

    #[test]
    fn cues_show_while_they_last() {
        let mut choices = Choices::default();
        let cue = |start_us, end_us, text: &str| Cue { start_us, end_us, text: text.into() };
        choices.cues = vec![cue(0, 2_000_000, "one"), cue(1_000_000, 3_000_000, "two"), cue(5_000_000, 6_000_000, "later")];
        assert_eq!(choices.text_at(1_500_000).as_deref(), Some("one\ntwo"));
        assert_eq!(choices.text_at(4_000_000), None);
        assert_eq!(choices.cues.len(), 1, "ended cues are dropped");
        assert_eq!(choices.text_at(5_000_000).as_deref(), Some("later"));
    }
}
