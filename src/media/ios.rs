//! iOS: `AVPlayer` plays the file (audio, clock, pause, seek); an
//! `AVPlayerItemVideoOutput` hands out each frame as an IOSurface-backed
//! `CVPixelBuffer` in BGRA, which `CVMetalTextureCache` wraps as an
//! `MTLTexture` and wgpu takes over through `texture_from_raw`. The decoder
//! and the conversion to BGRA run in VideoToolbox on the GPU; no pixels pass
//! through the CPU.
//!
//! With the `ffmpeg` feature, [`FfmpegPlayer`] plays files through FFmpeg
//! instead: VideoToolbox decodes into BGRA pixel buffers that take the same
//! import, and audio plays through an AudioQueue that drives the shared
//! clock ([`super::player`]).

use anyhow::{anyhow, bail, Result};
use gpui::{DevicePixels, ImageId, Size};
use gpui_wgpu::wgpu;
use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{class, msg_send};
use objc2_metal::{MTLTexture, MTLTextureType};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use super::{Backend, Source};
use std::ptr;
use std::time::{Duration, Instant};
#[cfg(feature = "ffmpeg")]
use super::planes::{Picture, Uploader};
#[cfg(feature = "ffmpeg")]
use super::player::{play_audio, play_video, AudioSink, Shared};
#[cfg(feature = "ffmpeg")]
use std::sync::{Arc, Condvar, Mutex, PoisonError};
#[cfg(feature = "ffmpeg")]
use std::thread::JoinHandle;

/// `kCVPixelFormatType_32BGRA`.
const PIXEL_FORMAT_BGRA: u32 = u32::from_be_bytes(*b"BGRA");
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8: usize = 80;
/// `AVPlayerActionAtItemEnd`.
const ACTION_PAUSE: isize = 1;
const ACTION_NONE: isize = 2;
/// `AVAudioSessionSetActiveOptionNotifyOthersOnDeactivation`.
const NOTIFY_OTHERS_ON_DEACTIVATION: usize = 1;
/// `AVPlayerItemStatusReadyToPlay`.
const ITEM_READY: isize = 1;
/// `AVPlayerItemStatusFailed`.
const ITEM_FAILED: isize = 2;

/// What plays the file.
pub(super) enum Decoder {
    Player(Player),
    #[cfg(feature = "ffmpeg")]
    Ffmpeg(Box<FfmpegPlayer>),
}

impl Decoder {
    pub fn open(source: Source, backend: Backend) -> Result<Self> {
        Ok(match backend {
            Backend::Platform => Self::Player(Player::open(source)?),
            #[cfg(feature = "ffmpeg")]
            Backend::Ffmpeg => Self::Ffmpeg(Box::new(FfmpegPlayer::open(source, false)?)),
            #[cfg(feature = "ffmpeg")]
            Backend::FfmpegSoftware => Self::Ffmpeg(Box::new(FfmpegPlayer::open(source, true)?)),
        })
    }
}

macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*) => {
        impl Decoder {
            $(pub fn $name(&self $(, $arg: $ty)*) $(-> $ret)? {
                match self {
                    Self::Player(player) => player.$name($($arg),*),
                    #[cfg(feature = "ffmpeg")]
                    Self::Ffmpeg(player) => player.$name($($arg),*),
                }
            })*
        }
    };
}

forward! {
    video_size() -> Size<DevicePixels>;
    frames_shown() -> u64;
    play();
    pause();
    is_paused() -> bool;
    seek(position: Duration);
    set_looping(looping: bool);
    position() -> Duration;
    duration() -> Duration;
    ended() -> bool;
    is_active() -> bool;
    update(image_id: ImageId) -> Result<()>;
}

/// `AVPlayer` with an `AVPlayerItemVideoOutput`.
pub(super) struct Player {
    player: *mut AnyObject,
    output: *mut AnyObject,
    /// Whether the track's transform has been looked at (once the item is
    /// ready, so a stream's tracks are known without blocking).
    transform_checked: Cell<bool>,
    looping: Cell<bool>,
    size: Cell<Size<DevicePixels>>,
    cache: TextureCache,
    frames_shown: Cell<u64>,
    last_import: Cell<Option<Instant>>,
}

/// The texture cache, with the `MTLDevice` it was made for.
type TextureCache = RefCell<Option<(*const c_void, CFRef)>>;

impl Player {
    pub fn open(source: Source) -> Result<Self> {
        // SAFETY: called on the main thread, which has an autorelease pool;
        // the objects kept are retained here and released in `drop`.
        unsafe {
            let url: *mut AnyObject = match source {
                Source::File(path) => {
                    let path = path.to_str().ok_or_else(|| anyhow!("non-UTF-8 path {path:?}"))?;
                    msg_send![class!(NSURL), fileURLWithPath: crate::ios::util::nsstring(path)]
                }
                Source::Url(url) => msg_send![class!(NSURL), URLWithString: crate::ios::util::nsstring(url)],
            };
            if url.is_null() {
                bail!("not a URL: {source:?}");
            }
            let item: *mut AnyObject = msg_send![class!(AVPlayerItem), playerItemWithURL: url];
            if item.is_null() {
                bail!("AVPlayerItem for {source:?} failed");
            }
            let keys = [kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferMetalCompatibilityKey];
            let values: [*mut AnyObject; 2] = [
                msg_send![class!(NSNumber), numberWithUnsignedInt: PIXEL_FORMAT_BGRA],
                msg_send![class!(NSNumber), numberWithBool: true],
            ];
            let attributes: *mut AnyObject = msg_send![
                class!(NSDictionary),
                dictionaryWithObjects: values.as_ptr(),
                forKeys: keys.as_ptr(),
                count: keys.len()
            ];
            let output: *mut AnyObject = msg_send![class!(AVPlayerItemVideoOutput), alloc];
            let output: *mut AnyObject = msg_send![output, initWithPixelBufferAttributes: attributes];
            if output.is_null() {
                bail!("AVPlayerItemVideoOutput failed");
            }
            let _: () = msg_send![item, addOutput: output];
            let player: *mut AnyObject = msg_send![class!(AVPlayer), playerWithPlayerItem: item];
            if player.is_null() {
                let _: () = msg_send![output, release];
                bail!("AVPlayer failed");
            }
            let player: *mut AnyObject = msg_send![player, retain];
            audio_session_acquire();
            let _: () = msg_send![player, setActionAtItemEnd: ACTION_PAUSE];
            let _: () = msg_send![player, play];
            log::info!("video: playing {source:?} through AVPlayer");
            Ok(Self {
                player,
                output,
                transform_checked: Cell::new(false),
                looping: Cell::new(false),
                size: Cell::new(Size::default()),
                cache: RefCell::new(None),
                frames_shown: Cell::new(0),
                last_import: Cell::new(None),
            })
        }
    }

    /// The size of the frames imported so far.
    pub fn video_size(&self) -> Size<DevicePixels> {
        self.size.get()
    }

    pub fn frames_shown(&self) -> u64 {
        self.frames_shown.get()
    }

    pub fn play(&self) {
        if self.ended() {
            self.seek(Duration::ZERO);
        }
        // SAFETY: a live player, on the main thread.
        unsafe { msg_send![self.player, play] }
    }

    pub fn pause(&self) {
        // SAFETY: a live player, on the main thread.
        unsafe { msg_send![self.player, pause] }
    }

    pub fn is_paused(&self) -> bool {
        // SAFETY: a live player, on the main thread.
        let rate: f32 = unsafe { msg_send![self.player, rate] };
        rate == 0.0
    }

    pub fn seek(&self, position: Duration) {
        let mut micros = position.as_micros() as i64;
        if self.duration_us() > 0 {
            micros = micros.min(self.duration_us());
        }
        let time = CMTime::from_micros(micros);
        let zero = CMTime::from_micros(0);
        // SAFETY: a live player, on the main thread.
        unsafe { msg_send![self.player, seekToTime: time, toleranceBefore: zero, toleranceAfter: zero] }
    }

    pub fn set_looping(&self, looping: bool) {
        self.looping.set(looping);
        // The end is handled in `update`: keep playing there.
        let action = if looping { ACTION_NONE } else { ACTION_PAUSE };
        // SAFETY: a live player, on the main thread.
        unsafe { msg_send![self.player, setActionAtItemEnd: action] }
    }

    fn item(&self) -> *mut AnyObject {
        // SAFETY: a live player, on the main thread; the item stays current.
        unsafe { msg_send![self.player, currentItem] }
    }

    /// The item's duration, 0 while a stream is still loading.
    fn duration_us(&self) -> i64 {
        // SAFETY: a live item, on the main thread.
        let duration: CMTime = unsafe { msg_send![self.item(), duration] };
        duration.micros()
    }

    fn position_us(&self) -> i64 {
        // SAFETY: a live player, on the main thread.
        let time: CMTime = unsafe { msg_send![self.player, currentTime] };
        time.micros()
    }

    pub fn position(&self) -> Duration {
        let position = self.position_us().max(0);
        let duration = self.duration_us();
        Duration::from_micros(if duration > 0 { position.min(duration) } else { position } as u64)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us() as u64)
    }

    fn at_end(&self) -> bool {
        let duration = self.duration_us();
        duration > 0 && self.position_us() >= duration
    }

    pub fn ended(&self) -> bool {
        !self.looping.get() && self.at_end()
    }

    pub fn is_active(&self) -> bool {
        !self.is_paused()
            || self
                .last_import
                .get()
                .is_some_and(|time| time.elapsed() < Duration::from_millis(500))
    }

    /// Imports the frame due now, if the output has a new one, and makes it
    /// the texture drawn for `image_id`.
    pub fn update(&self, image_id: ImageId) -> Result<()> {
        if self.looping.get() && self.at_end() {
            self.seek(Duration::ZERO);
        }
        if !self.transform_checked.get() {
            self.check_transform();
        }
        // SAFETY: a live output, on the main thread; the copied buffer is
        // owned (+1) and released by `CFRef`.
        let buffer = unsafe {
            let time: CMTime = msg_send![self.output, itemTimeForHostTime: CACurrentMediaTime()];
            let fresh: bool = msg_send![self.output, hasNewPixelBufferForItemTime: time];
            if !fresh {
                return Ok(());
            }
            let buffer: *mut CVBuffer =
                msg_send![self.output, copyPixelBufferForItemTime: time, itemTimeForDisplay: ptr::null_mut::<CMTime>()];
            match CFRef::new(buffer.cast()) {
                Some(buffer) => buffer,
                None => return Ok(()),
            }
        };
        let context = crate::ios::gpu_context().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let context = context.borrow();
        let context = context.as_ref().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let texture = import(&self.cache, buffer, &context.device)?;
        gpui_wgpu::set_external_texture(image_id, &texture);
        self.size.set(Size {
            width: DevicePixels(texture.width() as i32),
            height: DevicePixels(texture.height() as i32),
        });
        self.last_import.set(Some(Instant::now()));
        let shown = self.frames_shown.get() + 1;
        self.frames_shown.set(shown);
        if shown == 1 {
            log::info!(
                "video: IOSurface frame imported as a {}x{} Metal texture",
                texture.width(),
                texture.height()
            );
        }
        Ok(())
    }

    /// The output hands out frames as stored. A composition built from the
    /// asset turns them by the track's transform, at the cost of one extra
    /// GPU pass, so only rotated tracks get one.
    fn check_transform(&self) {
        // SAFETY: a live item, on the main thread. Once the item is ready to
        // play its asset's tracks are loaded, so asking for them won't block.
        unsafe {
            let item = self.item();
            let status: isize = msg_send![item, status];
            if status == ITEM_FAILED {
                self.transform_checked.set(true);
                let error: *mut AnyObject = msg_send![item, error];
                let description: *mut AnyObject = msg_send![error, localizedDescription];
                let text: *const std::ffi::c_char =
                    if description.is_null() { ptr::null() } else { msg_send![description, UTF8String] };
                let text = if text.is_null() {
                    "no description".into()
                } else {
                    std::ffi::CStr::from_ptr(text).to_string_lossy()
                };
                log::warn!("video: the item failed: {text}");
                return;
            }
            if status != ITEM_READY {
                return;
            }
            self.transform_checked.set(true);
            let asset: *mut AnyObject = msg_send![item, asset];
            let tracks: *mut AnyObject = msg_send![asset, tracksWithMediaType: AVMediaTypeVideo];
            let track: *mut AnyObject = msg_send![tracks, firstObject];
            if track.is_null() {
                return;
            }
            let transform: CGAffineTransform = msg_send![track, preferredTransform];
            if (transform.a, transform.b, transform.c, transform.d) != (1.0, 0.0, 0.0, 1.0) {
                let composition: *mut AnyObject =
                    msg_send![class!(AVVideoComposition), videoCompositionWithPropertiesOfAsset: asset];
                let _: () = msg_send![item, setVideoComposition: composition];
                log::info!("video: track turned by {:.0}°", transform.b.atan2(transform.a).to_degrees());
            }
        }
    }
}

/// Wraps an IOSurface-backed BGRA pixel buffer as a wgpu texture.
fn import(cache: &TextureCache, buffer: CFRef, device: &wgpu::Device) -> Result<wgpu::Texture> {
    // SAFETY: a live wgpu device; no raw queue operations happen here.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
        .ok_or_else(|| anyhow!("video frames need the Metal backend"))?;
    let metal_device = Retained::as_ptr(hal.raw_device()).cast::<c_void>();
    let mut cache = cache.borrow_mut();
    if cache.as_ref().is_none_or(|(device, _)| *device != metal_device) {
        let mut created = ptr::null_mut();
        // SAFETY: a live MTLDevice; the cache is owned by `CFRef`.
        let status = unsafe {
            CVMetalTextureCacheCreate(ptr::null(), ptr::null(), metal_device, ptr::null(), &mut created)
        };
        let created = CFRef::new(created).filter(|_| status == 0);
        *cache = Some((metal_device, created.ok_or_else(|| anyhow!("CVMetalTextureCacheCreate: {status}"))?));
    }
    let (_, cache) = cache.as_ref().ok_or_else(|| anyhow!("no texture cache"))?;
    // SAFETY: `buffer` is a live pixel buffer; the CV texture is owned by
    // `CFRef` and keeps the IOSurface, and with it the MTLTexture, valid.
    let (width, height, cv_texture) = unsafe {
        let (width, height) = (CVPixelBufferGetWidth(buffer.0), CVPixelBufferGetHeight(buffer.0));
        let mut cv_texture = ptr::null_mut();
        let status = CVMetalTextureCacheCreateTextureFromImage(
            ptr::null(),
            cache.0,
            buffer.0,
            ptr::null(),
            MTL_PIXEL_FORMAT_BGRA8,
            width,
            height,
            0,
            &mut cv_texture,
        );
        let cv_texture = CFRef::new(cv_texture)
            .filter(|_| status == 0)
            .ok_or_else(|| anyhow!("CVMetalTextureCacheCreateTextureFromImage: {status}"))?;
        (width as u32, height as u32, cv_texture)
    };
    // SAFETY: the CV texture returns its MTLTexture (+0), retained here.
    let raw = unsafe {
        Retained::retain(CVMetalTextureGetTexture(cv_texture.0).cast::<ProtocolObject<dyn MTLTexture>>())
    }
    .ok_or_else(|| anyhow!("CVMetalTexture without an MTLTexture"))?;
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    // SAFETY: a 2D BGRA8 texture of this size from this device; the
    // pixel buffer and CV texture stay alive until wgpu drops it.
    let texture = unsafe {
        let texture = wgpu::hal::metal::Device::texture_from_raw(
            raw,
            wgpu::TextureFormat::Bgra8Unorm,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
            Some(Box::new(move || drop((cv_texture, buffer)))),
        );
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            texture,
            &wgpu::TextureDescriptor {
                label: Some("video frame"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::TextureUses::RESOURCE,
        )
    };
    Ok(texture)
}

/// FFmpeg demuxing, decoding through VideoToolbox (or in software), with the
/// video thread on the shared clock ([`super::player`]).
#[cfg(feature = "ffmpeg")]
pub(super) struct FfmpegPlayer {
    size: Size<DevicePixels>,
    rotation: i32,
    duration_us: i64,
    /// Whether an audio thread was started, which holds the audio session.
    has_audio: bool,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    slot: Arc<Mutex<Option<Picture>>>,
    cache: TextureCache,
    uploader: RefCell<Uploader>,
    frames_shown: Cell<u64>,
    last_import: Cell<Option<Instant>>,
}

#[cfg(feature = "ffmpeg")]
impl FfmpegPlayer {
    fn open(source: Source, software: bool) -> Result<Self> {
        let Source::File(path) = source else {
            // Built without network, as on Android.
            bail!("the FFmpeg backend plays files only, not {source:?}");
        };
        let info = super::ffmpeg::probe(path)?;
        let slot = Arc::new(Mutex::new(None));
        let (mut video, _) = super::ffmpeg::FfmpegVideo::open(path, (!software).then_some(()), slot.clone())?;
        let audio = if info.has_audio {
            super::ffmpeg::FfmpegAudio::open(path)
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
                    if let Err(error) = play_audio::<AudioQueue>(&mut audio, &shared) {
                        log::error!("video: audio stopped: {error:#}");
                    }
                    // The video goes on by the monotonic clock.
                    shared.update(|state| {
                        state.audio = false;
                        state.holding = false;
                    });
                }
            })?);
            audio_session_acquire();
        }
        let (width, height) =
            if info.rotation % 180 == 90 { (info.height, info.width) } else { (info.width, info.height) };
        Ok(Self {
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
            cache: RefCell::new(None),
            uploader: RefCell::new(Uploader::default()),
            frames_shown: Cell::new(0),
            last_import: Cell::new(None),
        })
    }

    fn video_size(&self) -> Size<DevicePixels> {
        self.size
    }

    fn frames_shown(&self) -> u64 {
        self.frames_shown.get()
    }

    fn play(&self) {
        self.shared.play()
    }

    fn pause(&self) {
        self.shared.pause()
    }

    fn is_paused(&self) -> bool {
        self.shared.is_paused()
    }

    fn seek(&self, position: Duration) {
        self.shared.seek((position.as_micros() as i64).clamp(0, self.duration_us))
    }

    fn set_looping(&self, looping: bool) {
        self.shared.set_looping(looping)
    }

    fn position(&self) -> Duration {
        Duration::from_micros(self.shared.position_us().clamp(0, self.duration_us) as u64)
    }

    fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us as u64)
    }

    fn ended(&self) -> bool {
        self.shared.ended()
    }

    fn is_active(&self) -> bool {
        self.shared.is_active(self.last_import.get())
    }

    /// Imports the newest decoded picture, if there is one, and makes it the
    /// texture drawn for `image_id`.
    fn update(&self, image_id: ImageId) -> Result<()> {
        let Some(picture) = self.slot.lock().unwrap_or_else(PoisonError::into_inner).take() else {
            return Ok(());
        };
        let context = crate::ios::gpu_context().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let context = context.borrow();
        let context = context.as_ref().ok_or_else(|| anyhow!("no GPU context yet"))?;
        let (texture, path) = match picture.pixel_buffer() {
            Some(buffer) => {
                // SAFETY: a live pixel buffer; the texture keeps its own reference.
                let buffer = CFRef::new(unsafe { CFRetain(buffer) }).ok_or_else(|| anyhow!("null pixel buffer"))?;
                let texture = import(&self.cache, buffer, &context.device)?;
                let texture = if self.rotation == 0 {
                    Arc::new(texture)
                } else {
                    self.uploader
                        .borrow_mut()
                        .rotate(&texture, self.rotation, &context.device, &context.queue)?
                };
                (texture, "VideoToolbox pixel buffer imported as a Metal texture")
            }
            None => (
                self.uploader
                    .borrow_mut()
                    .convert(&picture, self.rotation, &context.device, &context.queue)?,
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
        Ok(())
    }
}

#[cfg(feature = "ffmpeg")]
impl Drop for FfmpegPlayer {
    fn drop(&mut self) {
        self.shared.update(|state| state.stop = true);
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                log::error!("video: playback thread panicked");
            }
        }
        if self.has_audio {
            audio_session_release();
        }
    }
}

/// Audio output for the FFmpeg backend: an AudioQueue fed 16-bit PCM in
/// small buffers. A buffer's callback means its frames have played, which
/// gives the clock its timestamps.
#[cfg(feature = "ffmpeg")]
struct AudioQueue {
    queue: *mut c_void,
    rate: i32,
    channels: i32,
    capacity_frames: usize,
    /// Boxed so the callback's pointer to it stays put.
    state: Box<QueueState>,
    started: Cell<bool>,
}

#[cfg(feature = "ffmpeg")]
struct QueueState {
    buffers: Mutex<QueueBuffers>,
    freed: Condvar,
    bytes_per_frame: u32,
}

#[cfg(feature = "ffmpeg")]
struct QueueBuffers {
    /// Buffers not enqueued (`AudioQueueBufferRef`s as addresses).
    free: Vec<usize>,
    /// Bumped by a flush; buffers enqueued before it don't count as played.
    epoch: usize,
    written: i64,
    played: i64,
    /// When `played` was last reached, on the monotonic clock.
    played_at: Option<i64>,
}

/// Buffers in the queue, and frames in each: about 85 ms ahead at 48 kHz.
#[cfg(feature = "ffmpeg")]
const QUEUE_BUFFERS: usize = 4;
#[cfg(feature = "ffmpeg")]
const QUEUE_BUFFER_FRAMES: usize = 1024;

#[cfg(feature = "ffmpeg")]
impl AudioQueue {
    fn lock(&self) -> std::sync::MutexGuard<'_, QueueBuffers> {
        self.state.buffers.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(feature = "ffmpeg")]
extern "C" fn queue_buffer_done(user: *mut c_void, _queue: *mut c_void, buffer: *mut AudioQueueBuffer) {
    // SAFETY: `user` is the queue's boxed state, alive until the queue is
    // disposed; the buffer belongs to the queue.
    unsafe {
        let state = &*(user as *const QueueState);
        let mut buffers = state.buffers.lock().unwrap_or_else(PoisonError::into_inner);
        if (*buffer).user_data as usize == buffers.epoch {
            buffers.played += ((*buffer).audio_data_byte_size / state.bytes_per_frame) as i64;
            buffers.played_at = Some(super::player::monotonic_ns());
        }
        buffers.free.push(buffer as usize);
        drop(buffers);
        state.freed.notify_all();
    }
}

#[cfg(feature = "ffmpeg")]
impl AudioSink for AudioQueue {
    fn open(rate: i32, channels: i32) -> Result<Self> {
        let bytes_per_frame = 2 * channels as u32;
        let format = AudioStreamBasicDescription {
            sample_rate: rate as f64,
            format_id: u32::from_be_bytes(*b"lpcm"),
            // Signed integer, packed.
            format_flags: 4 | 8,
            bytes_per_packet: bytes_per_frame,
            frames_per_packet: 1,
            bytes_per_frame,
            channels_per_frame: channels as u32,
            bits_per_channel: 16,
            reserved: 0,
        };
        let state = Box::new(QueueState {
            buffers: Mutex::new(QueueBuffers {
                free: Vec::new(),
                epoch: 0,
                written: 0,
                played: 0,
                played_at: None,
            }),
            freed: Condvar::new(),
            bytes_per_frame,
        });
        let mut queue = ptr::null_mut();
        // SAFETY: the callback runs on the queue's own thread, with the boxed
        // state, which `Drop` keeps until the queue is disposed.
        let status = unsafe {
            AudioQueueNewOutput(
                &format,
                queue_buffer_done,
                &*state as *const QueueState as *mut c_void,
                ptr::null_mut(),
                ptr::null(),
                0,
                &mut queue,
            )
        };
        if status != 0 || queue.is_null() {
            bail!("AudioQueueNewOutput: {status}");
        }
        let sink = Self {
            queue,
            rate,
            channels,
            capacity_frames: QUEUE_BUFFER_FRAMES,
            state,
            started: Cell::new(false),
        };
        for _ in 0..QUEUE_BUFFERS {
            let mut buffer = ptr::null_mut();
            // SAFETY: a live queue; the buffer belongs to it.
            let status =
                unsafe { AudioQueueAllocateBuffer(queue, QUEUE_BUFFER_FRAMES as u32 * bytes_per_frame, &mut buffer) };
            if status != 0 {
                bail!("AudioQueueAllocateBuffer: {status}");
            }
            sink.lock().free.push(buffer as usize);
        }
        log::info!("video: audio through an AudioQueue, {rate} Hz, {channels} channels");
        Ok(sink)
    }

    fn rate(&self) -> i32 {
        self.rate
    }

    fn channels(&self) -> i32 {
        self.channels
    }

    fn follow(&self, running: bool) {
        if running == self.started.get() {
            return;
        }
        // SAFETY: a live queue.
        let status = unsafe {
            if running {
                AudioQueueStart(self.queue, ptr::null())
            } else {
                AudioQueuePause(self.queue)
            }
        };
        if status != 0 {
            log::warn!("video: AudioQueue {}: {status}", if running { "start" } else { "pause" });
        }
        self.started.set(running);
    }

    fn pause_and_flush(&self) {
        self.follow(false);
        let mut buffers = self.lock();
        buffers.epoch += 1;
        buffers.played = buffers.written;
        buffers.played_at = None;
        drop(buffers);
        // SAFETY: a live queue; its callbacks hand back the dropped buffers.
        unsafe { AudioQueueReset(self.queue) };
    }

    fn frames_written(&self) -> i64 {
        self.lock().written
    }

    unsafe fn write(&self, data: *const i16, frames: usize) -> Result<usize> {
        let mut buffers = self.lock();
        if buffers.free.is_empty() {
            buffers = self
                .state
                .freed
                .wait_timeout(buffers, super::player::POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let Some(buffer) = buffers.free.pop() else {
            return Ok(0);
        };
        let buffer = buffer as *mut AudioQueueBuffer;
        let count = frames.min(self.capacity_frames);
        // SAFETY: the buffer is ours until enqueued and holds
        // `capacity_frames` frames; the caller vouches for `data`.
        unsafe {
            ptr::copy_nonoverlapping(data, (*buffer).audio_data.cast::<i16>(), count * self.channels as usize);
            (*buffer).audio_data_byte_size = count as u32 * self.state.bytes_per_frame;
            (*buffer).user_data = buffers.epoch as *mut c_void;
        }
        buffers.written += count as i64;
        drop(buffers);
        // SAFETY: a live queue and one of its buffers.
        let status = unsafe { AudioQueueEnqueueBuffer(self.queue, buffer, 0, ptr::null()) };
        if status != 0 {
            let mut buffers = self.lock();
            buffers.free.push(buffer as usize);
            buffers.written -= count as i64;
            bail!("AudioQueueEnqueueBuffer: {status}");
        }
        Ok(count)
    }

    fn timestamp(&self) -> Option<(i64, i64)> {
        let buffers = self.lock();
        buffers.played_at.map(|time| (buffers.played, time))
    }
}

#[cfg(feature = "ffmpeg")]
impl Drop for AudioQueue {
    fn drop(&mut self) {
        // SAFETY: owned queue, disposed at once (with its buffers) before the
        // state its callback uses drops.
        unsafe { AudioQueueDispose(self.queue, true) };
    }
}

/// `AudioStreamBasicDescription`.
#[cfg(feature = "ffmpeg")]
#[repr(C)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

/// `AudioQueueBuffer`.
#[cfg(feature = "ffmpeg")]
#[repr(C)]
struct AudioQueueBuffer {
    audio_data_bytes_capacity: u32,
    audio_data: *mut c_void,
    audio_data_byte_size: u32,
    user_data: *mut c_void,
    packet_description_capacity: u32,
    packet_descriptions: *mut c_void,
    packet_description_count: u32,
}

#[cfg(feature = "ffmpeg")]
#[link(name = "AudioToolbox", kind = "framework")]
extern "C" {
    fn AudioQueueNewOutput(
        format: *const AudioStreamBasicDescription,
        callback: extern "C" fn(*mut c_void, *mut c_void, *mut AudioQueueBuffer),
        user_data: *mut c_void,
        run_loop: *mut c_void,
        run_loop_mode: *const c_void,
        flags: u32,
        queue_out: *mut *mut c_void,
    ) -> i32;
    fn AudioQueueAllocateBuffer(queue: *mut c_void, byte_size: u32, buffer_out: *mut *mut AudioQueueBuffer) -> i32;
    fn AudioQueueEnqueueBuffer(
        queue: *mut c_void,
        buffer: *mut AudioQueueBuffer,
        packet_descriptions: u32,
        descriptions: *const c_void,
    ) -> i32;
    fn AudioQueueStart(queue: *mut c_void, start_time: *const c_void) -> i32;
    fn AudioQueuePause(queue: *mut c_void) -> i32;
    fn AudioQueueReset(queue: *mut c_void) -> i32;
    fn AudioQueueDispose(queue: *mut c_void, immediate: bool) -> i32;
}

impl Drop for Player {
    fn drop(&mut self) {
        // SAFETY: owned references, released once, on the main thread.
        unsafe {
            let _: () = msg_send![self.player, pause];
            let _: () = msg_send![self.player, release];
            let _: () = msg_send![self.output, release];
        }
        audio_session_release();
    }
}

thread_local! {
    /// Live players, all on the main thread.
    static PLAYERS: Cell<usize> = const { Cell::new(0) };
}

/// The default session category (SoloAmbient) is muted by the silent
/// switch; video apps play through it, so the first player sets Playback.
/// AVPlayer pauses itself when a call or Siri interrupts the session.
fn audio_session_acquire() {
    if PLAYERS.replace(PLAYERS.get() + 1) > 0 {
        return;
    }
    // SAFETY: the shared session and its constant strings, on the main thread.
    unsafe {
        let session: *mut AnyObject = msg_send![class!(AVAudioSession), sharedInstance];
        let set: bool = msg_send![
            session,
            setCategory: AVAudioSessionCategoryPlayback,
            mode: AVAudioSessionModeMoviePlayback,
            options: 0usize,
            error: ptr::null_mut::<*mut AnyObject>()
        ];
        let active: bool = msg_send![session, setActive: true, error: ptr::null_mut::<*mut AnyObject>()];
        if !set || !active {
            log::warn!("video: audio session not set up (category {set}, active {active})");
        }
    }
}

/// After the last player, gives the audio back to other apps.
fn audio_session_release() {
    if PLAYERS.replace(PLAYERS.get().saturating_sub(1)) != 1 {
        return;
    }
    // SAFETY: the shared session, on the main thread.
    unsafe {
        let session: *mut AnyObject = msg_send![class!(AVAudioSession), sharedInstance];
        let _: bool = msg_send![
            session,
            setActive: false,
            withOptions: NOTIFY_OTHERS_ON_DEACTIVATION,
            error: ptr::null_mut::<*mut AnyObject>()
        ];
    }
}

/// An owned Core Foundation reference.
struct CFRef(*mut c_void);

// SAFETY: Core Foundation reference counting is thread-safe, and the pixel
// buffers and textures held are not used after being handed over.
unsafe impl Send for CFRef {}
unsafe impl Sync for CFRef {}

impl CFRef {
    fn new(pointer: *mut c_void) -> Option<Self> {
        (!pointer.is_null()).then_some(Self(pointer))
    }
}

impl Drop for CFRef {
    fn drop(&mut self) {
        // SAFETY: owned (+1) reference, released once.
        unsafe { CFRelease(self.0) };
    }
}

/// `CVPixelBufferRef`'s pointee, for the Objective-C method encoding.
#[repr(C)]
struct CVBuffer {
    _private: [u8; 0],
}

// SAFETY: matches `^{__CVBuffer=}`.
unsafe impl RefEncode for CVBuffer {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__CVBuffer", &[]));
}

/// CMTime as AVFoundation lays it out.
#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

// SAFETY: the field encodings match the C struct, which the runtime names
// `?` (an anonymous struct behind a typedef).
unsafe impl Encode for CMTime {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[Encoding::LongLong, Encoding::Int, Encoding::UInt, Encoding::LongLong],
    );
}

// SAFETY: a pointer to the struct above.
unsafe impl RefEncode for CMTime {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

impl CMTime {
    const VALID: u32 = 1;
    /// Positive or negative infinity, or indefinite.
    const NOT_NUMERIC: u32 = 4 | 8 | 16;

    fn from_micros(micros: i64) -> Self {
        Self {
            value: micros,
            timescale: 1_000_000,
            flags: Self::VALID,
            epoch: 0,
        }
    }

    /// Microseconds, or 0 for an invalid or infinite time.
    fn micros(self) -> i64 {
        if self.flags & Self::VALID == 0 || self.flags & Self::NOT_NUMERIC != 0 || self.timescale <= 0 {
            return 0;
        }
        (self.value as i128 * 1_000_000 / self.timescale as i128) as i64
    }
}

/// CGAffineTransform as Core Graphics lays it out (64-bit).
#[repr(C)]
#[derive(Clone, Copy)]
struct CGAffineTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

// SAFETY: six doubles in a struct the runtime names `CGAffineTransform`.
unsafe impl Encode for CGAffineTransform {
    const ENCODING: Encoding = Encoding::Struct("CGAffineTransform", &[const { Encoding::Double }; 6]);
}

#[link(name = "AVFoundation", kind = "framework")]
#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    static AVMediaTypeVideo: *mut AnyObject;
    static AVAudioSessionCategoryPlayback: *mut AnyObject;
    static AVAudioSessionModeMoviePlayback: *mut AnyObject;
}

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn CACurrentMediaTime() -> f64;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(object: *const c_void);
    #[cfg(feature = "ffmpeg")]
    fn CFRetain(object: *mut c_void) -> *mut c_void;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: *mut AnyObject;
    static kCVPixelBufferMetalCompatibilityKey: *mut AnyObject;
    fn CVPixelBufferGetWidth(buffer: *mut c_void) -> usize;
    fn CVPixelBufferGetHeight(buffer: *mut c_void) -> usize;
    fn CVMetalTextureCacheCreate(
        allocator: *const c_void,
        cache_attributes: *const c_void,
        metal_device: *const c_void,
        texture_attributes: *const c_void,
        cache_out: *mut *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn CVMetalTextureCacheCreateTextureFromImage(
        allocator: *const c_void,
        cache: *mut c_void,
        source_image: *mut c_void,
        texture_attributes: *const c_void,
        pixel_format: usize,
        width: usize,
        height: usize,
        plane_index: usize,
        texture_out: *mut *mut c_void,
    ) -> i32;
    fn CVMetalTextureGetTexture(texture: *mut c_void) -> *mut c_void;
}
