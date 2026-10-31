//! Video frames drawn inside the GPUI scene, decoded by the platform and
//! imported on the GPU without CPU copies.
//!
//! Unlike `packages::video_player`, which shows a native view over the GPUI
//! surface, these frames are ordinary scene content: they clip, round, layer
//! and scroll like images, and GPUI can draw on top of them.
//!
//! Each frame is drawn through `Window::paint_image` with a placeholder
//! [`RenderImage`] whose id the patched `gpui-pre-wgpu` atlas answers with the
//! current frame's texture (`gpui_wgpu::set_external_texture`). Applications
//! must therefore patch `gpui-pre-wgpu` with `vendor/gpui-pre-wgpu`, as for
//! `offscreen`.
//!
//! - Android: `AMediaExtractor` + `AMediaCodec` decode into an `AImageReader`;
//!   each frame's `AHardwareBuffer` is imported into Vulkan and converted from
//!   YCbCr to RGBA by one render pass.
//! - iOS: not yet (IOSurface import, see the media track in `PLAN.md`).
//!
//! Feature-gated behind `media`.

#[cfg(target_os = "android")]
mod android;

use gpui::{Bounds, Corners, DevicePixels, Pixels, RenderImage, Size, Window};
use std::path::Path;
use std::sync::Arc;

/// A video file decoded in the background, drawn with [`VideoFrames::paint`].
///
/// Playback starts on [`VideoFrames::open`], follows the file's timestamps and
/// loops at the end. Audio is not played yet.
pub struct VideoFrames {
    image: Arc<RenderImage>,
    #[cfg(target_os = "android")]
    inner: android::Decoder,
}

impl VideoFrames {
    /// Opens a local video file and starts decoding it.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        #[cfg(target_os = "android")]
        {
            Ok(Self {
                image: Arc::new(RenderImage::new(Vec::new())),
                inner: android::Decoder::open(path.as_ref())?,
            })
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = path;
            anyhow::bail!("video frames are only implemented on Android")
        }
    }

    /// The coded picture size, once the decoder knows it.
    pub fn video_size(&self) -> Size<DevicePixels> {
        #[cfg(target_os = "android")]
        return self.inner.video_size();
        #[cfg(not(target_os = "android"))]
        Size::default()
    }

    /// How many decoded frames have been imported into the scene so far.
    pub fn frames_shown(&self) -> u64 {
        #[cfg(target_os = "android")]
        return self.inner.frames_shown();
        #[cfg(not(target_os = "android"))]
        0
    }

    /// Draws the newest decoded frame into `bounds` (stretched; fit the bounds
    /// to [`Self::video_size`] to keep the aspect ratio) and asks for another
    /// frame. Call it from a `canvas` paint callback.
    pub fn paint(&self, bounds: Bounds<Pixels>, window: &mut Window) {
        #[cfg(target_os = "android")]
        if let Err(error) = self.inner.update(self.image.id) {
            log::warn!("video frame import failed: {error:#}");
        }
        window.request_animation_frame();
        // Until a frame is registered the atlas would read the placeholder's
        // (absent) pixels.
        if self.frames_shown() == 0 {
            return;
        }
        if let Err(error) = window.paint_image(
            bounds,
            bounds,
            Corners::default(),
            self.image.clone(),
            0,
            false,
        ) {
            log::warn!("painting video frame failed: {error:#}");
        }
    }
}

impl Drop for VideoFrames {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        gpui_wgpu::remove_external_texture(self.image.id);
    }
}
