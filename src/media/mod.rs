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
//!   YCbCr to RGBA by one render pass. Audio plays through AAudio and drives
//!   the clock the frames follow.
//! - iOS: not yet (IOSurface import, see the media track in `PLAN.md`).
//!
//! Feature-gated behind `media`.

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "android")]
use android::Decoder;

use gpui::{px, size, Bounds, Corners, DevicePixels, Pixels, RenderImage, Size, Window};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// A video file played in the background, drawn with [`VideoFrames::paint`].
///
/// Playback starts on [`VideoFrames::open`] and follows the audio clock (or
/// the monotonic clock for files without audio). It pauses by itself while
/// the app is in the background.
pub struct VideoFrames {
    image: Arc<RenderImage>,
    inner: Decoder,
}

impl VideoFrames {
    /// Opens a local video file and starts playing it.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let inner = Decoder::open(path.as_ref())?;
        // Drawn by the atlas only while no frame is registered (before the
        // first frame, or after a device loss until the next one).
        let placeholder = image::Frame::new(image::RgbaImage::new(1, 1));
        Ok(Self {
            image: Arc::new(RenderImage::new(vec![placeholder])),
            inner,
        })
    }

    /// The displayed picture size (rotation applied).
    pub fn video_size(&self) -> Size<DevicePixels> {
        self.inner.video_size()
    }

    /// How many decoded frames have been imported into the scene so far.
    pub fn frames_shown(&self) -> u64 {
        self.inner.frames_shown()
    }

    /// Resumes playback, from the start if it had ended.
    pub fn play(&self) {
        self.inner.play()
    }

    pub fn pause(&self) {
        self.inner.pause()
    }

    pub fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// Jumps to `position`; the frame there shows even while paused.
    pub fn seek(&self, position: Duration) {
        self.inner.seek(position)
    }

    /// Whether playback starts over at the end instead of stopping.
    pub fn set_looping(&self, looping: bool) {
        self.inner.set_looping(looping)
    }

    pub fn position(&self) -> Duration {
        self.inner.position()
    }

    pub fn duration(&self) -> Duration {
        self.inner.duration()
    }

    /// Whether playback stopped at the end (not looping).
    pub fn ended(&self) -> bool {
        self.inner.ended()
    }

    /// Draws the newest decoded frame, as large as fits in `bounds` with its
    /// aspect ratio and centred, with `corner_radii` rounding the picture.
    /// Call it from a `canvas` paint callback; it asks for further frames
    /// while new ones may arrive.
    pub fn paint(&self, bounds: Bounds<Pixels>, corner_radii: Corners<Pixels>, window: &mut Window) {
        if let Err(error) = self.inner.update(self.image.id) {
            log::warn!("video frame import failed: {error:#}");
        }
        if self.inner.is_active() {
            window.request_animation_frame();
        }
        let bounds = contain(bounds, self.video_size());
        if let Err(error) = window.paint_image(bounds, bounds, corner_radii, self.image.clone(), 0, false) {
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

/// The largest rectangle with the aspect ratio of `content` centred in
/// `bounds`; `bounds` itself while `content` is empty.
fn contain(bounds: Bounds<Pixels>, content: Size<DevicePixels>) -> Bounds<Pixels> {
    if content.width.0 <= 0 || content.height.0 <= 0 {
        return bounds;
    }
    let (width, height) = (content.width.0 as f32, content.height.0 as f32);
    let scale = (bounds.size.width / px(width)).min(bounds.size.height / px(height));
    Bounds::centered_at(bounds.center(), size(px(width * scale), px(height * scale)))
}

/// The conversion shader's push constants: two rows mapping output uv, plus
/// one, to normalised buffer coordinates, so the output shows the `crop`
/// rectangle (x, y, width, height) of a `buffer`-sized image turned clockwise
/// by `rotation` degrees.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn uv_mapping(crop: (i32, i32, u32, u32), buffer: (u32, u32), rotation: i32) -> [f32; 8] {
    // Crop-relative source coordinates (s, t) = (a·u + b·v + c, d·u + e·v + f).
    let [a, b, c, d, e, f] = match rotation {
        90 => [0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
        180 => [-1.0, 0.0, 1.0, 0.0, -1.0, 1.0],
        270 => [0.0, -1.0, 1.0, 1.0, 0.0, 0.0],
        _ => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    let (x, y, w, h) = (crop.0 as f32, crop.1 as f32, crop.2 as f32, crop.3 as f32);
    let (bw, bh) = (buffer.0 as f32, buffer.1 as f32);
    [
        a * w / bw,
        b * w / bw,
        (x + c * w) / bw,
        0.0,
        d * h / bh,
        e * h / bh,
        (y + f * h) / bh,
        0.0,
    ]
}

/// Where no decoder exists yet: [`VideoFrames::open`] fails, so no value of
/// this type is ever made.
#[cfg(not(target_os = "android"))]
enum Decoder {}

#[cfg(not(target_os = "android"))]
#[allow(clippy::unused_self)]
impl Decoder {
    fn open(_path: &Path) -> anyhow::Result<Self> {
        anyhow::bail!("video frames are only implemented on Android")
    }
    fn video_size(&self) -> Size<DevicePixels> {
        match *self {}
    }
    fn frames_shown(&self) -> u64 {
        match *self {}
    }
    fn play(&self) {
        match *self {}
    }
    fn pause(&self) {
        match *self {}
    }
    fn is_paused(&self) -> bool {
        match *self {}
    }
    fn seek(&self, _position: Duration) {
        match *self {}
    }
    fn set_looping(&self, _looping: bool) {
        match *self {}
    }
    fn position(&self) -> Duration {
        match *self {}
    }
    fn duration(&self) -> Duration {
        match *self {}
    }
    fn ended(&self) -> bool {
        match *self {}
    }
    fn is_active(&self) -> bool {
        match *self {}
    }
    fn update(&self, _image_id: gpui::ImageId) -> anyhow::Result<()> {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::point;

    #[test]
    fn contain_letterboxes_and_centres() {
        let bounds = Bounds::new(point(px(10.), px(20.)), size(px(400.), px(400.)));
        let fitted = contain(bounds, size(DevicePixels(640), DevicePixels(360)));
        assert_eq!(fitted.size, size(px(400.), px(225.)));
        assert_eq!(fitted.origin, point(px(10.), px(20. + 87.5)));
        assert_eq!(contain(bounds, Size::default()), bounds);
    }

    /// Asserts where output (u, v) samples, to within rounding.
    fn assert_maps(mapping: [f32; 8], (u, v): (f32, f32), (x, y): (f32, f32)) {
        let mapped = (
            mapping[0] * u + mapping[1] * v + mapping[2],
            mapping[4] * u + mapping[5] * v + mapping[6],
        );
        assert!(
            (mapped.0 - x).abs() < 1e-6 && (mapped.1 - y).abs() < 1e-6,
            "({u}, {v}) maps to {mapped:?}, not ({x}, {y})"
        );
    }

    #[test]
    fn uv_mapping_crops_and_rotates() {
        // A 100x50 crop at (10, 20) of a 200x100 buffer.
        let crop = (10, 20, 100, 50);
        let plain = uv_mapping(crop, (200, 100), 0);
        assert_maps(plain, (0., 0.), (0.05, 0.2));
        assert_maps(plain, (1., 1.), (0.55, 0.7));
        // Turned clockwise, the output's top left shows the crop's bottom
        // left and its top right the crop's top left.
        let quarter = uv_mapping(crop, (200, 100), 90);
        assert_maps(quarter, (0., 0.), (0.05, 0.7));
        assert_maps(quarter, (1., 0.), (0.05, 0.2));
        let half = uv_mapping(crop, (200, 100), 180);
        assert_maps(half, (0., 0.), (0.55, 0.7));
        let three = uv_mapping(crop, (200, 100), 270);
        assert_maps(three, (0., 0.), (0.55, 0.2));
        assert_maps(three, (0., 1.), (0.05, 0.2));
    }
}
