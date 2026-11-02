//! Video frames drawn inside the GPUI scene, decoded by the platform and
//! imported on the GPU without CPU copies.
//!
//! Unlike a native view over the GPUI surface (the removed
//! `packages::video_player`), these frames are ordinary scene content: they clip, round, layer
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
//! - iOS: `AVPlayer` plays the file; an `AVPlayerItemVideoOutput` hands out
//!   IOSurface-backed BGRA pixel buffers, wrapped as Metal textures through
//!   `CVMetalTextureCache`.
//! - FFmpeg ([`Backend::Ffmpeg`], feature `ffmpeg`): FFmpeg demuxes; video
//!   decodes in hardware, or in software with the planes uploaded and
//!   converted on the GPU ([`Backend::FfmpegSoftware`] forces that).
//!   - Android: through FFmpeg's MediaCodec decoders into the same reader and
//!     import; audio decodes in software into AAudio.
//!   - iOS: through VideoToolbox into BGRA pixel buffers, taking the same
//!     Metal import as `AVPlayer`'s; audio into an AudioQueue.
//!   - Linux (the only backend there; [`Backend::Platform`] means FFmpeg too):
//!     through VA-API on the window's GPU, each surface imported into Vulkan
//!     as DMA-BUF planes and converted to RGBA; audio through cpal.
//!
//! Feature-gated behind `media`.

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "android")]
use android::Decoder;
#[cfg(target_os = "ios")]
mod ios;
#[cfg(target_os = "ios")]
use ios::Decoder;
#[cfg(all(feature = "ffmpeg", target_os = "linux"))]
mod linux;
#[cfg(all(feature = "ffmpeg", target_os = "linux"))]
use linux::Decoder;
#[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
mod ffmpeg;
#[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
mod planes;
#[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
mod input;
#[cfg(any(target_os = "android", all(feature = "ffmpeg", any(target_os = "ios", target_os = "linux"))))]
mod player;

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

/// What decodes the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    /// The platform's player: `AMediaExtractor` + `AMediaCodec`, `AVPlayer`.
    #[default]
    Platform,
    /// FFmpeg, decoding in hardware through the platform decoder when it takes
    /// the stream, in software otherwise.
    #[cfg(feature = "ffmpeg")]
    Ffmpeg,
    /// FFmpeg, always decoding in software.
    #[cfg(feature = "ffmpeg")]
    FfmpegSoftware,
}

impl VideoFrames {
    /// Opens a local video file and starts playing it with the platform's player.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::open_source(Source::File(path.as_ref()), Backend::Platform)
    }

    /// Opens a local video file and starts playing it, decoded by `backend`.
    pub fn open_with(path: impl AsRef<Path>, backend: Backend) -> anyhow::Result<Self> {
        Self::open_source(Source::File(path.as_ref()), backend)
    }

    /// Opens an `http(s)` URL and starts playing it as it streams in.
    ///
    /// On Android this connects before it returns, so call it off a
    /// latency-sensitive path; on iOS it returns at once and the duration and
    /// size are zero until the stream is ready.
    pub fn open_url(url: &str) -> anyhow::Result<Self> {
        Self::open_source(Source::Url(url), Backend::Platform)
    }

    /// Plays bytes Rust supplies, through FFmpeg ([`Backend::Platform`] means
    /// [`Backend::Ffmpeg`] here). `open` makes a fresh reader over the same
    /// bytes, called once per playback thread (video, audio) and once to
    /// probe. FFmpeg seeks it as it needs; `SeekFrom::End(0)` gives the size,
    /// and a reader that can't answer that plays with the size unknown.
    #[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
    pub fn open_reader(
        open: impl Fn() -> std::io::Result<Box<dyn ReadSeek>> + Send + Sync + 'static,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        Self::open_source(Source::Ffmpeg(&input::Origin::Reader(Arc::new(open))), ffmpeg_backend(backend))
    }

    /// Plays an HLS stream whose every playlist, key and segment Rust fetches:
    /// `fetch` gets each URI as FFmpeg's `hls` demuxer resolves it against
    /// `playlist` and returns its bytes whole. AES-128 segments are decrypted
    /// here. URIs must be relative or plain paths: FFmpeg has no network
    /// protocols, so it refuses `http://` URIs before `fetch` sees them.
    /// Each playback thread demuxes on its own, so `fetch` sees each segment
    /// once per thread; cache in `fetch` to download it once.
    #[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
    pub fn open_hls(
        playlist: &str,
        fetch: impl Fn(&str) -> std::io::Result<Vec<u8>> + Send + Sync + 'static,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        let origin = input::Origin::Hls {
            playlist: playlist.to_owned(),
            fetch: Arc::new(fetch),
        };
        Self::open_source(Source::Ffmpeg(&origin), ffmpeg_backend(backend))
    }

    fn open_source(source: Source, backend: Backend) -> anyhow::Result<Self> {
        let inner = Decoder::open(source, backend)?;
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
        #[cfg(any(target_os = "android", target_os = "ios", all(feature = "ffmpeg", target_os = "linux")))]
        gpui_wgpu::remove_external_texture(self.image.id);
    }
}

/// A byte source for [`VideoFrames::open_reader`].
#[cfg(feature = "ffmpeg")]
pub trait ReadSeek: std::io::Read + std::io::Seek + Send {}
#[cfg(feature = "ffmpeg")]
impl<T: std::io::Read + std::io::Seek + Send> ReadSeek for T {}

/// Rust-supplied bytes play only through FFmpeg.
#[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
fn ffmpeg_backend(backend: Backend) -> Backend {
    if backend == Backend::Platform { Backend::Ffmpeg } else { backend }
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
#[cfg_attr(
    not(any(target_os = "android", all(feature = "ffmpeg", any(target_os = "ios", target_os = "linux")))),
    allow(dead_code)
)]
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

/// The YCbCr → RGB conversion for 8-bit video, as the planes shader takes
/// it: (Y offset, Y scale, chroma offset, chroma scale), then the R, G and B
/// rows applied to (Y, Cb, Cr, 0). BT.709 or BT.601, limited or full range.
#[cfg_attr(
    not(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux"))),
    allow(dead_code)
)]
fn yuv_to_rgb(bt709: bool, full_range: bool) -> [f32; 16] {
    let (kr, kb) = if bt709 { (0.2126, 0.0722) } else { (0.299, 0.114) };
    let kg = 1.0 - kr - kb;
    let range = if full_range {
        [0.0, 1.0, 128.0 / 255.0, 1.0]
    } else {
        [16.0 / 255.0, 255.0 / 219.0, 128.0 / 255.0, 255.0 / 224.0]
    };
    [
        range[0],
        range[1],
        range[2],
        range[3],
        1.0,
        0.0,
        2.0 - 2.0 * kr,
        0.0,
        1.0,
        -(2.0 - 2.0 * kb) * kb / kg,
        -(2.0 - 2.0 * kr) * kr / kg,
        0.0,
        1.0,
        2.0 - 2.0 * kb,
        0.0,
        0.0,
    ]
}

/// What a decoder plays.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(any(target_os = "android", target_os = "ios", all(feature = "ffmpeg", target_os = "linux"))),
    allow(dead_code)
)]
enum Source<'a> {
    File(&'a Path),
    /// A Rust reader or Rust-fed HLS, for the FFmpeg backends.
    #[cfg(all(feature = "ffmpeg", any(target_os = "android", target_os = "ios", target_os = "linux")))]
    Ffmpeg(&'a input::Origin),
    // The FFmpeg player, all Linux has, opens files only.
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Url(&'a str),
}

/// Where no decoder exists yet: [`VideoFrames::open`] fails, so no value of
/// this type is ever made.
#[cfg(not(any(target_os = "android", target_os = "ios", all(feature = "ffmpeg", target_os = "linux"))))]
enum Decoder {}

#[cfg(not(any(target_os = "android", target_os = "ios", all(feature = "ffmpeg", target_os = "linux"))))]
#[allow(clippy::unused_self)]
impl Decoder {
    fn open(_source: Source, _backend: Backend) -> anyhow::Result<Self> {
        anyhow::bail!("video frames are only implemented on Android, iOS and Linux (with `ffmpeg`)")
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

    /// Converts 8-bit (Y, Cb, Cr) the way the planes shader does.
    fn convert(m: [f32; 16], (y, cb, cr): (f32, f32, f32)) -> [f32; 3] {
        let y = (y / 255.0 - m[0]) * m[1];
        let (cb, cr) = ((cb / 255.0 - m[2]) * m[3], (cr / 255.0 - m[2]) * m[3]);
        [4, 8, 12].map(|row| (m[row] * y + m[row + 1] * cb + m[row + 2] * cr).clamp(0.0, 1.0))
    }

    #[test]
    fn yuv_to_rgb_matches_reference_colours() {
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 0.01);
        // Limited-range black and white.
        assert!(close(convert(yuv_to_rgb(true, false), (16., 128., 128.)), [0., 0., 0.]));
        assert!(close(convert(yuv_to_rgb(false, false), (235., 128., 128.)), [1., 1., 1.]));
        // BT.709 limited red (63, 102, 240) and BT.601 full-range green (150, 44, 21).
        assert!(close(convert(yuv_to_rgb(true, false), (63., 102., 240.)), [1., 0., 0.]));
        assert!(close(convert(yuv_to_rgb(false, true), (150., 44., 21.)), [0., 1., 0.]));
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
