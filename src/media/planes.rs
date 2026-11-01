//! Software-decoded frames: the Y, U and V planes are uploaded as they are
//! (1.5 bytes a pixel, not 4) and converted to RGBA by one render pass, with
//! the stream's colour matrix and range and the display rotation.

use anyhow::{anyhow, Result};
use gpui_wgpu::wgpu;
use std::sync::Arc;

/// An 8-bit 4:2:0 frame waiting to be uploaded.
pub(super) struct SoftwareFrame {
    frame: ffmpeg_next::frame::Video,
    pub timestamp_us: i64,
}

// SAFETY: the frame owns its buffers and is only read after it moved.
unsafe impl Send for SoftwareFrame {}

impl SoftwareFrame {
    pub fn new(frame: ffmpeg_next::frame::Video, timestamp_us: i64) -> Self {
        Self { frame, timestamp_us }
    }
}

const SHADER: &str = r#"
struct Params { map_s: vec4<f32>, map_t: vec4<f32>, range: vec4<f32>, r: vec4<f32>, g: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var y_plane: texture_2d<f32>;
@group(0) @binding(1) var u_plane: texture_2d<f32>;
@group(0) @binding(2) var v_plane: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var<uniform> p: Params;

struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}

@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    let st = vec2<f32>(dot(p.map_s.xy, in.uv) + p.map_s.z, dot(p.map_t.xy, in.uv) + p.map_t.z);
    let y = (textureSample(y_plane, smp, st).r - p.range.x) * p.range.y;
    let c = (vec2<f32>(textureSample(u_plane, smp, st).r, textureSample(v_plane, smp, st).r) - p.range.z) * p.range.w;
    let v = vec4<f32>(y, c, 0.0);
    return vec4<f32>(clamp(vec3<f32>(dot(p.r, v), dot(p.g, v), dot(p.b, v)), vec3(0.0), vec3(1.0)), 1.0);
}
"#;

struct Pipeline {
    device: wgpu::Device,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

/// Reused plane textures, while the frame size stays.
struct Planes {
    size: (u32, u32),
    textures: [wgpu::Texture; 3],
}

#[derive(Default)]
pub(super) struct Uploader {
    pipeline: Option<Pipeline>,
    planes: Option<Planes>,
}

impl Uploader {
    /// Uploads `frame` and converts it into a new RGBA8 texture, turned
    /// clockwise by `rotation` degrees. Submits on `queue`.
    pub fn convert(
        &mut self,
        frame: &SoftwareFrame,
        rotation: i32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Arc<wgpu::Texture>> {
        let frame_ref = &frame.frame;
        let (width, height) = (frame_ref.width(), frame_ref.height());
        if width == 0 || height == 0 {
            return Err(anyhow!("empty software frame"));
        }
        if self.pipeline.as_ref().is_none_or(|p| &p.device != device) {
            self.pipeline = Some(Pipeline::new(device));
            self.planes = None;
        }
        let pipeline = self.pipeline.as_ref().ok_or_else(|| anyhow!("no pipeline"))?;
        let chroma = (width.div_ceil(2), height.div_ceil(2));
        if self.planes.as_ref().is_none_or(|p| p.size != (width, height)) {
            let plane = |(w, h): (u32, u32)| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("video_plane"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            self.planes = Some(Planes {
                size: (width, height),
                textures: [plane((width, height)), plane(chroma), plane(chroma)],
            });
        }
        let planes = self.planes.as_ref().ok_or_else(|| anyhow!("no planes"))?;
        for (index, texture) in planes.textures.iter().enumerate() {
            let (w, h) = if index == 0 { (width, height) } else { chroma };
            queue.write_texture(
                texture.as_image_copy(),
                frame_ref.data(index),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(frame_ref.stride(index) as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        let (out_w, out_h) = if rotation % 180 == 90 { (height, width) } else { (width, height) };
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("video_frame"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // SAFETY: reads plain colour fields of a live frame.
        let (colorspace, range) = unsafe {
            let raw = &*frame_ref.as_ptr();
            (raw.colorspace, raw.color_range)
        };
        let mapping = super::uv_mapping((0, 0, width, height), (width, height), rotation);
        let colour = super::yuv_to_rgb(
            colorspace == ffmpeg_next::ffi::AVColorSpace::AVCOL_SPC_BT709
                || (colorspace == ffmpeg_next::ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED && height >= 720),
            range == ffmpeg_next::ffi::AVColorRange::AVCOL_RANGE_JPEG,
        );
        let mut params = [0f32; 24];
        params[..8].copy_from_slice(&mapping);
        params[8..].copy_from_slice(&colour);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video_params"),
            size: std::mem::size_of_val(&params) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bytes: Vec<u8> = params.iter().flat_map(|value| value.to_ne_bytes()).collect();
        queue.write_buffer(&uniform, 0, &bytes);
        let views: Vec<_> = planes.textures.iter().map(|t| t.create_view(&Default::default())).collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_planes"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&views[0]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&views[1]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&pipeline.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let view = output.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("video_planes_to_rgba"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Writes and the pass go in before the frame that samples the output.
        queue.submit([encoder.finish()]);
        Ok(Arc::new(output))
    }
}

impl Pipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video_planes_to_rgba"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("video_planes_to_rgba"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            device: device.clone(),
            pipeline,
            sampler,
        }
    }
}
