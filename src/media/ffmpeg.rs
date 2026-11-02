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
    pub has_audio: bool,
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
        has_audio: input.streams().best(ffmpeg::media::Type::Audio).is_some(),
    })
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
}

impl Demuxer {
    fn open(origin: &Origin, kind: ffmpeg::media::Type) -> Result<(Self, ffmpeg::codec::Parameters)> {
        let input = open_input(origin)?;
        let stream = input.streams().best(kind).ok_or_else(|| anyhow!("no {kind:?} stream"))?;
        let (index, time_base, parameters) = (stream.index(), stream.time_base(), stream.parameters());
        // SAFETY: reads a field of the open input.
        let start_us = match unsafe { (*input.as_ptr()).start_time } {
            ffi::AV_NOPTS_VALUE => 0,
            start => start,
        };
        Ok((
            Self {
                input,
                stream: index,
                time_base,
                start_us,
                pending: None,
                input_done: false,
            },
            parameters,
        ))
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
                    Ok(()) if packet.stream() == self.stream => break packet,
                    Ok(()) => continue,
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
    ) -> Result<(Self, bool)> {
        let (demuxer, parameters) = Demuxer::open(origin, ffmpeg::media::Type::Video)?;
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

    fn yuv420p(&mut self, frame: ffmpeg::frame::Video) -> Result<ffmpeg::frame::Video> {
        if frame.format() == Pixel::YUV420P {
            return Ok(frame);
        }
        // ponytail: 10-bit and 4:4:4 go through swscale to 8-bit 4:2:0; upload
        // them natively if the software path ever matters for them.
        let scaler = match &mut self.scaler {
            Some(scaler) if scaler.input().format == frame.format() => scaler,
            scaler => scaler.insert(ffmpeg::software::scaling::Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                Pixel::YUV420P,
                frame.width(),
                frame.height(),
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
}

// SAFETY: the source moves to its playback thread and is only used there.
unsafe impl Send for FfmpegAudio {}

impl FfmpegAudio {
    pub fn open(origin: &Origin) -> Result<Self> {
        let (demuxer, parameters) = Demuxer::open(origin, ffmpeg::media::Type::Audio)?;
        let mut context = ffmpeg::codec::Context::from_parameters(parameters)?;
        // SAFETY: the context is not open yet.
        unsafe { (*context.as_mut_ptr()).pkt_timebase = demuxer.time_base.into() };
        let decoder = context.decoder().audio()?;
        log::info!(
            "video: FFmpeg audio {} decoder, {} Hz, {} channels",
            decoder.codec().map_or_else(|| "?".into(), |codec| codec.name().to_owned()),
            decoder.rate(),
            decoder.channels(),
        );
        Ok(Self {
            demuxer,
            decoder,
            resampler: None,
            samples: Vec::new(),
        })
    }

    /// Every output plays mono and stereo; more channels are downmixed.
    fn output_channels(&self) -> i32 {
        (self.decoder.channels() as i32).clamp(1, 2)
    }
}

impl AudioSource for FfmpegAudio {
    fn format(&self) -> Result<(i32, i32)> {
        Ok((self.decoder.rate() as i32, self.output_channels()))
    }

    fn seek(&mut self, target_us: i64) -> Result<()> {
        self.demuxer.seek(target_us, &mut self.decoder)
    }

    fn next(&mut self) -> Result<AudioOutput> {
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
        let (mut video, _) = FfmpegVideo::open(&origin, hardware, slot.clone())?;
        let audio = if info.has_audio {
            FfmpegAudio::open(&origin)
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
