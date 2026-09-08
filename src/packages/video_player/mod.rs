//! Video playback composed in the GPUI GPU scene.
//!
//! Native decoders deliver owned GPU frames; controls and fullscreen remain GPUI elements.

#[cfg(any(target_os = "android", target_os = "ios"))]
pub use lumina_video_gpui::{GpuiVideoPlayer, GpuiVideoPlayerConfig};
