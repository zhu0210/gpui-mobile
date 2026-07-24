//! GPUI-native video playback powered by `lumina-video`.
//!
//! Frames are decoded by the platform-native Lumina backend and composited
//! directly by GPUI's wgpu renderer. Android uses MediaCodec/ExoPlayer and
//! AHardwareBuffer; iOS uses AVFoundation/VideoToolbox and IOSurface.

pub use lumina_video::{
    GpuVideoFrame, GpuVideoFrameTextures, GpuiVideoPlayer as VideoPlayer,
    GpuiVideoPlayerConfig as VideoPlayerConfig, RealizedVideoPath, VideoState,
};
