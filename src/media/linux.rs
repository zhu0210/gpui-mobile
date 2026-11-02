//! Linux: FFmpeg decodes through VA-API on the GPU the window renders with;
//! each surface is mapped to DRM PRIME and its Y and CbCr planes imported into
//! Vulkan as DMA-BUFs, then converted to RGBA by one render pass. Audio plays
//! through cpal (PipeWire, PulseAudio or ALSA).

use super::ffmpeg::{FfmpegPlayer, HardwareTarget, Platform};
use super::planes::{Picture, Uploader};
use super::player::{monotonic_ns, AudioSink, POLL};
use anyhow::{anyhow, bail, Context as _, Result};
use cpal::traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _};
use ffmpeg_next::ffi;
use gpui_wgpu::wgpu;
use std::cell::Cell;
use std::collections::VecDeque;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

pub(super) type Decoder = FfmpegPlayer<Linux>;

#[derive(Default)]
pub(super) struct Linux;

impl Platform for Linux {
    type Sink = Cpal;
    const IMPORT: &str = "VA-API surface imported as DMA-BUF planes, converted on the GPU";

    fn context() -> Option<gpui_wgpu::WgpuContext> {
        gpui_wgpu::WgpuContext::window_context()
    }

    fn hardware(context: Option<&gpui_wgpu::WgpuContext>) -> Option<HardwareTarget> {
        let Some(context) = context else {
            log::warn!("video: no window GPU yet, decoding in software");
            return None;
        };
        let info = context.adapter.get_info();
        if info.backend != wgpu::Backend::Vulkan
            || !context.device.features().contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        {
            log::warn!("video: {} cannot import DMA-BUFs, decoding in software", info.name);
            return None;
        }
        // VA-API must decode on the GPU that draws: its render node by PCI bus.
        let node = PathBuf::from(format!("/dev/dri/by-path/pci-{}-render", info.device_pci_bus_id));
        if !node.exists() {
            log::warn!("video: no render node {node:?} for {}, decoding in software", info.name);
            return None;
        }
        Some(node)
    }

    fn import(
        &self,
        picture: &Picture,
        rotation: i32,
        uploader: &mut Uploader,
        context: &gpui_wgpu::WgpuContext,
    ) -> Result<Option<Arc<wgpu::Texture>>> {
        let Some(frame) = picture.vaapi() else {
            return Ok(None);
        };
        let mapped = Mapped::new(frame)?;
        let (width, height) = (frame.width(), frame.height());
        let planes = mapped.planes();
        if planes.len() != 2 {
            bail!("expected a two-plane NV12/P010 surface, got {} planes", planes.len());
        }
        // SAFETY: a VA-API frame has its frames context.
        let ten_bit = unsafe {
            let frames = &*((*(*frame.as_ptr()).hw_frames_ctx).data as *const ffi::AVHWFramesContext);
            frames.sw_format == ffi::AVPixelFormat::AV_PIX_FMT_P010LE
        };
        let (luma_format, chroma_format) = if ten_bit {
            (wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm)
        } else {
            (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm)
        };
        // ponytail: two VkImages imported per frame (~0.5 ms); cache them by
        // VA surface if profiling says so.
        let luma = import_plane(&context.device, &planes[0], luma_format, (width, height))?;
        let chroma = import_plane(
            &context.device,
            &planes[1],
            chroma_format,
            (width.div_ceil(2), height.div_ceil(2)),
        )?;
        let texture = uploader.convert_biplanar(
            frame,
            &luma,
            &chroma,
            ten_bit,
            rotation,
            &context.device,
            &context.queue,
        )?;
        // The surface goes back to FFmpeg's pool once the pass has read it.
        context.queue.on_submitted_work_done(move || drop(mapped));
        Ok(Some(texture))
    }
}

/// A VA-API frame mapped to DRM PRIME; holds the surface until dropped.
struct Mapped(*mut ffi::AVFrame);

// SAFETY: the mapping is only freed, from whichever thread drops it.
unsafe impl Send for Mapped {}

/// One plane of a mapped frame, as its DRM descriptor gives it.
struct Plane {
    fd: i32,
    modifier: u64,
    offset: u64,
    pitch: u64,
}

impl Mapped {
    fn new(frame: &ffmpeg_next::frame::Video) -> Result<Self> {
        // SAFETY: a live VA-API frame; the mapping keeps a reference to it.
        unsafe {
            let mapped = Self(ffi::av_frame_alloc());
            if mapped.0.is_null() {
                bail!("av_frame_alloc failed");
            }
            (*mapped.0).format = ffi::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            // Syncs the surface: decoding has finished when this returns.
            let ret = ffi::av_hwframe_map(mapped.0, frame.as_ptr(), ffi::AV_HWFRAME_MAP_READ as i32);
            if ret < 0 {
                bail!("av_hwframe_map to DRM PRIME: {}", ffmpeg_next::Error::from(ret));
            }
            Ok(mapped)
        }
    }

    fn planes(&self) -> Vec<Plane> {
        // SAFETY: a DRM PRIME frame keeps its descriptor in data[0].
        let descriptor = unsafe { &*((*self.0).data[0] as *const ffi::AVDRMFrameDescriptor) };
        descriptor.layers[..descriptor.nb_layers as usize]
            .iter()
            .flat_map(|layer| &layer.planes[..layer.nb_planes as usize])
            .map(|plane| {
                let object = &descriptor.objects[plane.object_index as usize];
                Plane {
                    fd: object.fd,
                    modifier: object.format_modifier,
                    offset: plane.offset as u64,
                    pitch: plane.pitch as u64,
                }
            })
            .collect()
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: owned; frees the mapping and its reference to the surface.
        unsafe { ffi::av_frame_free(&mut self.0) };
    }
}

/// Imports one plane as a sampled texture of `format` and `size`.
fn import_plane(
    device: &wgpu::Device,
    plane: &Plane,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
) -> Result<wgpu::Texture> {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    // SAFETY: the fd is open while the mapping lives; Vulkan takes a duplicate.
    let fd: OwnedFd = unsafe { BorrowedFd::borrow_raw(plane.fd) }.try_clone_to_owned()?;
    // SAFETY: the descriptor matches the plane FFmpeg exported; wgpu takes the
    // image and starts it from UNDEFINED, fine for uncompressed modifiers.
    unsafe {
        let hal = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| anyhow!("not a Vulkan device"))?;
        let texture = hal
            .texture_from_dmabuf_fd(
                fd,
                &wgpu::hal::TextureDescriptor {
                    label: Some("video_dmabuf_plane"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUses::RESOURCE,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                plane.modifier,
                plane.pitch,
                plane.offset,
            )
            .with_context(|| format!("importing a DMA-BUF plane (modifier {:#018x})", plane.modifier))?;
        drop(hal);
        Ok(device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            texture,
            &wgpu::TextureDescriptor {
                label: Some("video_dmabuf_plane"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::TextureUses::UNINITIALIZED,
        ))
    }
}

/// How much audio the ring holds ahead of the device.
const RING_MS: usize = 100;

/// The default output device through cpal, fed from a ring of 16-bit frames
/// that the device callback drains.
pub(super) struct Cpal {
    stream: cpal::Stream,
    rate: i32,
    channels: i32,
    ring: Arc<Ring>,
    playing: Cell<bool>,
}

struct Ring {
    state: Mutex<RingState>,
    drained: Condvar,
    capacity: usize,
}

struct RingState {
    samples: VecDeque<i16>,
    written: i64,
    /// Frames the device has taken.
    played: i64,
    /// A frame position and when it plays (monotonic ns), from the last callback.
    stamp: Option<(i64, i64)>,
}

impl Ring {
    fn lock(&self) -> MutexGuard<'_, RingState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl AudioSink for Cpal {
    fn open(rate: i32, channels: i32) -> Result<Self> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| anyhow!("no audio output device"))?;
        let samples_per_frame = channels as usize;
        let ring = Arc::new(Ring {
            state: Mutex::new(RingState {
                samples: VecDeque::new(),
                written: 0,
                played: 0,
                stamp: None,
            }),
            drained: Condvar::new(),
            capacity: rate as usize * RING_MS / 1000 * samples_per_frame,
        });
        let config = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(rate as u32),
            buffer_size: cpal::BufferSize::Default,
        };
        let stream = device.build_output_stream(
            &config,
            {
                let ring = ring.clone();
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let timestamp = info.timestamp();
                    let latency = timestamp.playback.duration_since(&timestamp.callback).unwrap_or_default();
                    let mut state = ring.lock();
                    // The first frame of this buffer plays after the latency.
                    state.stamp = Some((state.played, monotonic_ns() + latency.as_nanos() as i64));
                    let available = state.samples.len().min(data.len());
                    for (out, sample) in data.iter_mut().zip(state.samples.drain(..available)) {
                        *out = f32::from(sample) / 32768.0;
                    }
                    data[available..].fill(0.0);
                    state.played += (available / samples_per_frame) as i64;
                    drop(state);
                    ring.drained.notify_all();
                }
            },
            |error| log::warn!("video: audio output error: {error}"),
            None,
        )?;
        stream.pause()?;
        let name = device.name().unwrap_or_default();
        log::info!("video: audio through cpal on {name:?}, {rate} Hz, {channels} channels");
        Ok(Self {
            stream,
            rate,
            channels,
            ring,
            playing: Cell::new(false),
        })
    }

    fn rate(&self) -> i32 {
        self.rate
    }

    fn channels(&self) -> i32 {
        self.channels
    }

    fn follow(&self, running: bool) {
        if running == self.playing.get() {
            return;
        }
        let result = if running {
            self.stream.play().map_err(|error| error.to_string())
        } else {
            self.stream.pause().map_err(|error| error.to_string())
        };
        if let Err(error) = result {
            log::warn!("video: audio {}: {error}", if running { "play" } else { "pause" });
        }
        if !running {
            self.ring.lock().stamp = None;
        }
        self.playing.set(running);
    }

    fn pause_and_flush(&self) {
        self.follow(false);
        let mut state = self.ring.lock();
        state.samples.clear();
        state.played = state.written;
        state.stamp = None;
    }

    fn frames_written(&self) -> i64 {
        self.ring.lock().written
    }

    unsafe fn write(&self, data: *const i16, frames: usize) -> Result<usize> {
        let channels = self.channels as usize;
        let mut state = self.ring.lock();
        if state.samples.len() + channels > self.ring.capacity {
            state = self
                .ring
                .drained
                .wait_timeout(state, POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let count = ((self.ring.capacity - state.samples.len().min(self.ring.capacity)) / channels).min(frames);
        // SAFETY: the caller vouches for `frames` frames at `data`.
        let samples = unsafe { std::slice::from_raw_parts(data, count * channels) };
        state.samples.extend(samples);
        state.written += count as i64;
        Ok(count)
    }

    fn timestamp(&self) -> Option<(i64, i64)> {
        self.ring.lock().stamp
    }
}
