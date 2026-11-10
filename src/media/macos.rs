//! macOS: FFmpeg decodes through VideoToolbox into IOSurface-backed pixel
//! buffers, and GPUI's Metal renderer draws each as a surface
//! (`Window::paint_surface`), sampling its planes where they are. That
//! renderer takes one form only, 8-bit two-plane full-range BT.601, so frames
//! in another (video range, BT.709, 10-bit: most of them) pass through a
//! VideoToolbox pixel transfer first, which converts between surfaces.
//! Software-decoded frames are converted by swscale and copied into a pixel
//! buffer. Audio plays through cpal (CoreAudio).

use super::cpal_sink::Cpal;
use super::ffmpeg::{FfmpegPlayer, HardwareTarget, Picture, Platform};
use anyhow::{anyhow, bail, Result};
use gpui::ImageId;
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_video::{
    kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, kCVPixelBufferHeightKey,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey,
    kCVPixelBufferWidthKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess, CVBuffer,
    CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferPool,
    CVPixelBufferUnlockBaseAddress,
};
use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

pub(super) type Decoder = FfmpegPlayer<MacOs>;

#[derive(Default)]
pub(super) struct MacOs {
    /// The newest frame, as GPUI draws it.
    surface: RefCell<Option<CFRetained<CVPixelBuffer>>>,
    /// Buffers in the form GPUI draws, for frames that arrive in another.
    pool: RefCell<Option<Pool>>,
    transfer: RefCell<Option<Transfer>>,
}

impl MacOs {
    /// What [`super::VideoFrames::paint`] hands to `Window::paint_surface`.
    pub fn surface(&self) -> Option<CFRetained<CVPixelBuffer>> {
        self.surface.borrow().clone()
    }

    /// An empty full-range BT.601 buffer of `size` from the pool.
    fn buffer(&self, size: (usize, usize)) -> Result<CFRetained<CVPixelBuffer>> {
        let mut pool = self.pool.borrow_mut();
        let pool = match &mut *pool {
            Some(pool) if pool.size == size => pool,
            pool => pool.insert(Pool::new(size)?),
        };
        let mut buffer = ptr::null_mut();
        // SAFETY: the pool is live and `buffer` is written on success.
        let status = unsafe { CVPixelBufferPool::create_pixel_buffer(None, &pool.pool, NonNull::from(&mut buffer)) };
        let buffer = NonNull::new(buffer).filter(|_| status == kCVReturnSuccess);
        let buffer = buffer.ok_or_else(|| anyhow!("CVPixelBufferPoolCreatePixelBuffer: {status}"))?;
        // SAFETY: created above, so this takes its one reference.
        Ok(unsafe { CFRetained::from_raw(buffer) })
    }

    /// `source` as GPUI draws it, converted on the GPU by VideoToolbox.
    fn converted(&self, source: &CVPixelBuffer) -> Result<CFRetained<CVPixelBuffer>> {
        let target = self.buffer((CVPixelBufferGetWidth(source), CVPixelBufferGetHeight(source)))?;
        let mut transfer = self.transfer.borrow_mut();
        let transfer = match &mut *transfer {
            Some(transfer) => transfer,
            transfer => transfer.insert(Transfer::new()?),
        };
        // SAFETY: the session and both buffers are live.
        let status = unsafe { VTPixelTransferSessionTransferImage(transfer.0.as_ptr(), source, &target) };
        if status != 0 {
            bail!("VTPixelTransferSessionTransferImage: {status}");
        }
        Ok(target)
    }

    /// A software frame ([`super::ffmpeg::SOFTWARE`]: full-range BT.601 NV12)
    /// copied into a pixel buffer, turned clockwise by `rotation`.
    fn copied(&self, frame: &ffmpeg_next::frame::Video, rotation: i32) -> Result<CFRetained<CVPixelBuffer>> {
        let (width, height) = (frame.width() as usize, frame.height() as usize);
        let turned = if rotation % 180 == 90 { (height, width) } else { (width, height) };
        let target = self.buffer(turned)?;
        // SAFETY: the buffer is locked while its planes are written, each
        // within the rows and bytes a row CoreVideo reports for it.
        unsafe {
            let status = CVPixelBufferLockBaseAddress(&target, CVPixelBufferLockFlags(0));
            if status != kCVReturnSuccess {
                bail!("CVPixelBufferLockBaseAddress: {status}");
            }
            // Y, then CbCr at half the size, two bytes a pixel.
            for (plane, pixel) in [(0, 1), (1, 2)] {
                let size = if plane == 0 { (width, height) } else { (width.div_ceil(2), height.div_ceil(2)) };
                let stride = CVPixelBufferGetBytesPerRowOfPlane(&target, plane);
                let rows = CVPixelBufferGetHeightOfPlane(&target, plane);
                let fits = if rotation % 180 == 90 { (size.1, size.0) } else { size };
                let base = CVPixelBufferGetBaseAddressOfPlane(&target, plane).cast::<u8>();
                if base.is_null() || fits.0 > CVPixelBufferGetWidthOfPlane(&target, plane) || fits.1 > rows {
                    CVPixelBufferUnlockBaseAddress(&target, CVPixelBufferLockFlags(0));
                    bail!("pixel buffer plane {plane} doesn't hold {}x{}", fits.0, fits.1);
                }
                super::copy_turned(
                    frame.data(plane),
                    frame.stride(plane),
                    size,
                    pixel,
                    rotation,
                    std::slice::from_raw_parts_mut(base, stride * rows),
                    stride,
                );
            }
            CVPixelBufferUnlockBaseAddress(&target, CVPixelBufferLockFlags(0));
        }
        Ok(target)
    }
}

impl Platform for MacOs {
    type Sink = Cpal;

    fn hardware() -> Option<HardwareTarget> {
        Some(())
    }

    fn present(&self, picture: Picture, rotation: i32, _image_id: ImageId) -> Result<(&'static str, (u32, u32))> {
        let (buffer, path) = match picture.pixel_buffer() {
            Some(buffer) => {
                // SAFETY: a VideoToolbox frame's pixel buffer, live while the picture is.
                let buffer = unsafe { &*buffer.cast::<CVPixelBuffer>() };
                if drawable(buffer) {
                    // SAFETY: a live buffer; this takes a reference of its own.
                    let buffer = unsafe { CFRetained::retain(NonNull::from(buffer)) };
                    (buffer, "VideoToolbox pixel buffer drawn as a surface")
                } else {
                    (
                        self.converted(buffer)?,
                        "VideoToolbox pixel buffer converted on the GPU to full-range BT.601, drawn as a surface",
                    )
                }
            }
            None => (self.copied(&picture.frame, rotation)?, "software frame copied into a pixel buffer"),
        };
        let size = (CVPixelBufferGetWidth(&buffer) as u32, CVPixelBufferGetHeight(&buffer) as u32);
        *self.surface.borrow_mut() = Some(buffer);
        Ok((path, size))
    }
}

/// Whether GPUI's surface shader shows `buffer` right as it is: 8-bit
/// two-plane 4:2:0, full range, BT.601.
fn drawable(buffer: &CVPixelBuffer) -> bool {
    if CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
        return false;
    }
    // SAFETY: plain reads of a live buffer's attachment and a constant.
    unsafe {
        CVBuffer::attachment(buffer, kCVImageBufferYCbCrMatrixKey, ptr::null_mut())
            .is_some_and(|matrix| *matrix == **kCVImageBufferYCbCrMatrix_ITU_R_601_4)
    }
}

/// Recycled IOSurface-backed buffers of one size in the form GPUI draws.
struct Pool {
    size: (usize, usize),
    pool: CFRetained<CVPixelBufferPool>,
}

impl Pool {
    fn new(size: (usize, usize)) -> Result<Self> {
        let number = |value: usize| CFNumber::new_i64(value as i64);
        let (format, width, height) =
            (number(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange as usize), number(size.0), number(size.1));
        let surface = CFDictionary::<CFType, CFType>::empty();
        // SAFETY: CoreVideo's constant keys.
        let keys: [&CFString; 5] = unsafe {
            [
                kCVPixelBufferPixelFormatTypeKey,
                kCVPixelBufferWidthKey,
                kCVPixelBufferHeightKey,
                kCVPixelBufferIOSurfacePropertiesKey,
                kCVPixelBufferMetalCompatibilityKey,
            ]
        };
        let values: [&CFType; 5] = [&format, &width, &height, &surface, CFBoolean::new(true)];
        let attributes = CFDictionary::<CFString, CFType>::from_slices(&keys, &values);
        let mut pool = ptr::null_mut();
        // SAFETY: the attributes are live and `pool` is written on success.
        let status = unsafe { CVPixelBufferPool::create(None, None, Some(attributes.as_opaque()), NonNull::from(&mut pool)) };
        let pool = NonNull::new(pool).filter(|_| status == kCVReturnSuccess);
        let pool = pool.ok_or_else(|| anyhow!("CVPixelBufferPoolCreate {}x{}: {status}", size.0, size.1))?;
        Ok(Self {
            size,
            // SAFETY: created above, so this takes its one reference.
            pool: unsafe { CFRetained::from_raw(pool) },
        })
    }
}

/// A `VTPixelTransferSession` that writes BT.601.
struct Transfer(NonNull<c_void>);

impl Transfer {
    fn new() -> Result<Self> {
        let mut session = ptr::null_mut();
        // SAFETY: `session` is written on success, and released on drop.
        unsafe {
            let status = VTPixelTransferSessionCreate(ptr::null(), &mut session);
            let session = NonNull::new(session)
                .filter(|_| status == 0)
                .ok_or_else(|| anyhow!("VTPixelTransferSessionCreate: {status}"))?;
            let session = Self(session);
            let status = VTSessionSetProperty(
                session.0.as_ptr(),
                kVTPixelTransferPropertyKey_DestinationYCbCrMatrix,
                kCVImageBufferYCbCrMatrix_ITU_R_601_4,
            );
            if status != 0 {
                bail!("VTSessionSetProperty(DestinationYCbCrMatrix): {status}");
            }
            Ok(session)
        }
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        // SAFETY: the session this owns, invalidated and released once.
        unsafe {
            VTPixelTransferSessionInvalidate(self.0.as_ptr());
            CFRelease(self.0.as_ptr());
        }
    }
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    static kVTPixelTransferPropertyKey_DestinationYCbCrMatrix: &'static CFString;
    fn VTPixelTransferSessionCreate(allocator: *const c_void, session: *mut *mut c_void) -> i32;
    fn VTPixelTransferSessionTransferImage(session: *mut c_void, source: &CVPixelBuffer, target: &CVPixelBuffer) -> i32;
    fn VTPixelTransferSessionInvalidate(session: *mut c_void);
    fn VTSessionSetProperty(session: *mut c_void, key: &CFString, value: &CFString) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(object: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::super::ffmpeg::{probe, Choices, FfmpegVideo};
    use super::super::input::Origin;
    use super::super::player::{VideoOutput, VideoSource as _};
    use super::*;
    use std::sync::{Arc, Mutex};

    /// The `nth` picture of a bundled clip, decoded by VideoToolbox or in
    /// software, as GPUI would draw it: the buffer and its luma rows.
    fn drawn(clip: &str, hardware: bool, nth: usize) -> (CFRetained<CVPixelBuffer>, Vec<u8>) {
        let clip = format!("{}/example/assets/video/{clip}", env!("CARGO_MANIFEST_DIR"));
        let origin = Origin::of(super::super::Source::File(clip.as_ref())).unwrap();
        let info = probe(&origin).unwrap();
        let slot = Arc::new(Mutex::new(None));
        let (mut video, _) =
            FfmpegVideo::open(&origin, hardware.then_some(()), slot.clone(), Choices::new(&info)).unwrap();
        let mut seen = 0;
        let picture = loop {
            match video.next().unwrap() {
                VideoOutput::Frame(_, frame) => {
                    seen += 1;
                    video.release(frame, seen > nth).unwrap();
                    if let Some(picture) = slot.lock().unwrap().take() {
                        break picture;
                    }
                }
                VideoOutput::Pending => {}
                VideoOutput::End => panic!("{clip} ended after {seen} frames"),
            }
        };
        assert_eq!(picture.pixel_buffer().is_some(), hardware, "decoded by VideoToolbox");
        let platform = MacOs::default();
        platform.present(picture, info.rotation, ImageId(0)).unwrap();
        let buffer = platform.surface().unwrap();
        assert_eq!(CVPixelBufferGetPixelFormatType(&buffer), kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        let (width, height) = (CVPixelBufferGetWidth(&buffer), CVPixelBufferGetHeight(&buffer));
        let mut luma = Vec::with_capacity(width * height);
        // SAFETY: the buffer is locked while its first plane's rows are read.
        unsafe {
            let read_only = CVPixelBufferLockFlags::ReadOnly;
            assert_eq!(CVPixelBufferLockBaseAddress(&buffer, read_only), kCVReturnSuccess);
            let base = CVPixelBufferGetBaseAddressOfPlane(&buffer, 0).cast::<u8>();
            let stride = CVPixelBufferGetBytesPerRowOfPlane(&buffer, 0);
            for row in 0..height {
                luma.extend_from_slice(std::slice::from_raw_parts(base.add(row * stride), width));
            }
            CVPixelBufferUnlockBaseAddress(&buffer, read_only);
        }
        (buffer, luma)
    }

    /// VideoToolbox's frames, converted by its pixel transfer, show what
    /// FFmpeg's software decoder and swscale make of the same picture.
    #[test]
    fn videotoolbox_frames_match_software_ones() {
        let (hardware, hardware_luma) = drawn("av-60s-640x360.mp4", true, 60);
        let (software, software_luma) = drawn("av-60s-640x360.mp4", false, 60);
        let size = |buffer: &CVPixelBuffer| (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer));
        assert_eq!(size(&hardware), (640, 360));
        assert_eq!(size(&software), (640, 360));
        let spread = software_luma.iter().max().unwrap() - software_luma.iter().min().unwrap();
        assert!(spread > 64, "the frame has a picture to compare (luma spread {spread})");
        let difference = hardware_luma.iter().zip(&software_luma).map(|(a, b)| a.abs_diff(*b) as u64).sum::<u64>() as f64
            / software_luma.len() as f64;
        assert!(difference < 4.0, "mean luma difference {difference}");
    }

    /// A video recorded on its side decodes in software and is turned as it
    /// is copied.
    #[test]
    fn turned_videos_are_drawn_upright() {
        let clip = format!("{}/example/assets/video/rotated-90.mp4", env!("CARGO_MANIFEST_DIR"));
        let info = probe(&Origin::of(super::super::Source::File(clip.as_ref())).unwrap()).unwrap();
        assert_eq!(info.rotation, 90);
        let (buffer, _) = drawn("rotated-90.mp4", false, 0);
        assert_eq!(
            (CVPixelBufferGetWidth(&buffer), CVPixelBufferGetHeight(&buffer)),
            (info.height as usize, info.width as usize)
        );
    }
}
