//! iOS: `AVPlayer` plays the file (audio, clock, pause, seek); an
//! `AVPlayerItemVideoOutput` hands out each frame as an IOSurface-backed
//! `CVPixelBuffer` in BGRA, which `CVMetalTextureCache` wraps as an
//! `MTLTexture` and wgpu takes over through `texture_from_raw`. The decoder
//! and the conversion to BGRA run in VideoToolbox on the GPU; no pixels pass
//! through the CPU.

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
use std::path::Path;
use std::ptr;
use std::time::{Duration, Instant};

/// `kCVPixelFormatType_32BGRA`.
const PIXEL_FORMAT_BGRA: u32 = u32::from_be_bytes(*b"BGRA");
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8: usize = 80;
/// `AVPlayerActionAtItemEnd`.
const ACTION_PAUSE: isize = 1;
const ACTION_NONE: isize = 2;

pub(super) struct Decoder {
    player: *mut AnyObject,
    output: *mut AnyObject,
    duration_us: i64,
    looping: Cell<bool>,
    size: Cell<Size<DevicePixels>>,
    /// The texture cache, with the `MTLDevice` it was made for.
    cache: RefCell<Option<(*const c_void, CFRef)>>,
    frames_shown: Cell<u64>,
    last_import: Cell<Option<Instant>>,
}

impl Decoder {
    pub fn open(path: &Path) -> Result<Self> {
        let path = path.to_str().ok_or_else(|| anyhow!("non-UTF-8 path {path:?}"))?;
        // SAFETY: called on the main thread, which has an autorelease pool;
        // the objects kept are retained here and released in `drop`.
        unsafe {
            let url: *mut AnyObject = msg_send![class!(NSURL), fileURLWithPath: crate::ios::util::nsstring(path)];
            let item: *mut AnyObject = msg_send![class!(AVPlayerItem), playerItemWithURL: url];
            if item.is_null() {
                bail!("AVPlayerItem for {path} failed");
            }
            let asset: *mut AnyObject = msg_send![item, asset];
            let duration: CMTime = msg_send![asset, duration];

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
            let _: () = msg_send![player, setActionAtItemEnd: ACTION_PAUSE];
            let _: () = msg_send![player, play];
            log::info!("video: playing {path} through AVPlayer");
            Ok(Self {
                player,
                output,
                duration_us: duration.micros(),
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
        let time = CMTime::from_micros((position.as_micros() as i64).clamp(0, self.duration_us));
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

    fn position_us(&self) -> i64 {
        // SAFETY: a live player, on the main thread.
        let time: CMTime = unsafe { msg_send![self.player, currentTime] };
        time.micros()
    }

    pub fn position(&self) -> Duration {
        Duration::from_micros(self.position_us().clamp(0, self.duration_us) as u64)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us as u64)
    }

    pub fn ended(&self) -> bool {
        !self.looping.get() && self.position_us() >= self.duration_us
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
        if self.looping.get() && self.position_us() >= self.duration_us {
            self.seek(Duration::ZERO);
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
        let texture = self.import(buffer, &context.device)?;
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

    fn import(&self, buffer: CFRef, device: &wgpu::Device) -> Result<wgpu::Texture> {
        // SAFETY: a live wgpu device; no raw queue operations happen here.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
            .ok_or_else(|| anyhow!("video frames need the Metal backend"))?;
        let metal_device = Retained::as_ptr(hal.raw_device()).cast::<c_void>();
        let mut cache = self.cache.borrow_mut();
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
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: owned references, released once, on the main thread.
        unsafe {
            let _: () = msg_send![self.player, pause];
            let _: () = msg_send![self.player, release];
            let _: () = msg_send![self.output, release];
        }
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

// SAFETY: the field encodings match the C struct.
unsafe impl Encode for CMTime {
    const ENCODING: Encoding = Encoding::Struct(
        "CMTime",
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

#[link(name = "AVFoundation", kind = "framework")]
#[link(name = "CoreMedia", kind = "framework")]
extern "C" {}

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn CACurrentMediaTime() -> f64;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(object: *const c_void);
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
