//! Windows: FFmpeg decodes through Direct3D 11 on the device GPUI's renderer
//! draws with, so each frame is one of that device's textures (a slice of the
//! decoder's NV12 or P010 array). The device's video processor converts it to
//! a BGRA texture on the GPU, cropped to the picture, with the stream's
//! matrix and range; the patched `gpui-pre-windows` atlas draws that texture
//! wherever the frame's image is painted. Software-decoded frames are
//! converted to BGRA by swscale and copied into a texture. Audio plays through
//! cpal (WASAPI).
//!
//! shortcut: a lost device (driver update, GPU removed) ends hardware
//! decoding for the videos open then: their decoder's device is gone. Open
//! them again; reopen by itself if that ever matters.

use super::cpal_sink::Cpal;
use super::ffmpeg::{is_bt709, FfmpegPlayer, HardwareTarget, Picture, Platform};
use anyhow::{anyhow, bail, Context as _, Result};
use ffmpeg_next::{ffi, format::Pixel};
use gpui::{DevicePixels, ImageId, Size};
use std::cell::RefCell;
use std::mem::ManuallyDrop;
use windows::core::Interface as _;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11ShaderResourceView, ID3D11Texture2D,
    ID3D11VideoContext, ID3D11VideoDevice, ID3D11VideoProcessor, ID3D11VideoProcessorEnumerator,
    ID3D11VideoProcessorInputView, ID3D11VideoProcessorOutputView, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_WRITE_DISCARD,
    D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC,
    D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE, D3D11_VIDEO_PROCESSOR_COLOR_SPACE, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0,
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_STREAM,
    D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_RATIONAL, DXGI_SAMPLE_DESC};

pub(super) type Decoder = FfmpegPlayer<Direct3D>;

#[derive(Default)]
pub(super) struct Direct3D {
    target: RefCell<Option<Target>>,
}

impl Platform for Direct3D {
    type Sink = Cpal;

    fn hardware() -> Option<HardwareTarget> {
        let Some(device) = gpui_windows::directx_device() else {
            log::warn!("video: no Direct3D device yet, decoding in software");
            return None;
        };
        if device.cast::<ID3D11VideoDevice>().is_err() {
            log::warn!("video: the Direct3D device has no video support, decoding in software");
            return None;
        }
        // FFmpeg's decoding thread and GPUI's renderer both call the device's
        // one immediate context from here on.
        // SAFETY: a live device; the switch itself is thread-safe.
        let protected = unsafe {
            device.GetImmediateContext().and_then(|context| context.cast::<ID3D11Multithread>()).map(|multithread| {
                let _was = multithread.SetMultithreadProtected(true);
            })
        };
        if let Err(error) = protected {
            log::warn!("video: the Direct3D context can't be shared with a decoder ({error}), decoding in software");
            return None;
        }
        Some(device)
    }

    fn present(&self, picture: Picture, rotation: i32, image_id: ImageId) -> Result<(&'static str, (u32, u32))> {
        let device = gpui_windows::directx_device().ok_or_else(|| anyhow!("no Direct3D device yet"))?;
        let frame = &picture.frame;
        let mut target = self.target.borrow_mut();
        let (target, path) = if frame.format() == Pixel::D3D11 {
            // SAFETY: a Direct3D 11 frame keeps its texture in data[0] and the
            // slice of it in data[1], both live while the picture is.
            let (texture, slice) = unsafe {
                let raw = &*frame.as_ptr();
                let texture = raw.data[0].cast::<std::ffi::c_void>();
                let texture = ID3D11Texture2D::from_raw_borrowed(&texture).context("a frame without its texture")?;
                (texture.clone(), raw.data[1] as usize as u32)
            };
            // SAFETY: plain reads of a live texture.
            let coded = unsafe {
                if texture.GetDevice()?.as_raw() != device.as_raw() {
                    bail!("the frame belongs to a Direct3D device the renderer no longer uses");
                }
                let mut desc = D3D11_TEXTURE2D_DESC::default();
                texture.GetDesc(&mut desc);
                (desc.Width, desc.Height)
            };
            let target = Target::fit(&mut target, &device, (frame.width(), frame.height()), Some(coded))?;
            target.convert(&texture, slice, frame)?;
            (target, "Direct3D 11 texture converted on the GPU by the video processor")
        } else {
            let size = (frame.width(), frame.height());
            let turned = if rotation % 180 == 90 { (size.1, size.0) } else { size };
            let target = Target::fit(&mut target, &device, turned, None)?;
            target.upload(frame, rotation)?;
            (target, "software frame copied into a texture")
        };
        let (width, height) = target.size;
        gpui_windows::set_external_texture(
            image_id,
            &target.view,
            Size {
                width: DevicePixels(width as i32),
                height: DevicePixels(height as i32),
            },
        );
        Ok((path, target.size))
    }
}

/// The BGRA texture GPUI draws for the video, and what fills it.
struct Target {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    size: (u32, u32),
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    /// For hardware frames: the video processor writing into `texture`.
    converter: Option<Converter>,
}

struct Converter {
    /// The decoder textures' size: the picture's, rounded up to whole blocks.
    coded: (u32, u32),
    video_device: ID3D11VideoDevice,
    video_context: ID3D11VideoContext,
    enumerator: ID3D11VideoProcessorEnumerator,
    processor: ID3D11VideoProcessor,
    output: ID3D11VideoProcessorOutputView,
}

impl Target {
    /// The target in `slot` when it suits, a new one otherwise: on `device`,
    /// `size` pixels, filled from decoder textures of `coded` size or (`None`)
    /// from software frames.
    fn fit<'a>(
        slot: &'a mut Option<Self>,
        device: &ID3D11Device,
        size: (u32, u32),
        coded: Option<(u32, u32)>,
    ) -> Result<&'a mut Self> {
        let fits = slot.as_ref().is_some_and(|target| {
            target.device.as_raw() == device.as_raw()
                && target.size == size
                && target.converter.as_ref().map(|converter| converter.coded) == coded
        });
        if !fits {
            *slot = Some(Self::new(device, size, coded)?);
        }
        slot.as_mut().context("no video target")
    }

    fn new(device: &ID3D11Device, size: (u32, u32), coded: Option<(u32, u32)>) -> Result<Self> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: size.0,
            Height: size.1,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            // The video processor renders into it; software frames are mapped.
            Usage: if coded.is_some() { D3D11_USAGE_DEFAULT } else { D3D11_USAGE_DYNAMIC },
            BindFlags: if coded.is_some() {
                (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32
            } else {
                D3D11_BIND_SHADER_RESOURCE.0 as u32
            },
            CPUAccessFlags: if coded.is_some() { 0 } else { D3D11_CPU_ACCESS_WRITE.0 as u32 },
            MiscFlags: 0,
        };
        // SAFETY: plain creation calls on a live device, each result checked.
        unsafe {
            let context = device.GetImmediateContext()?;
            let mut texture = None;
            device.CreateTexture2D(&desc, None, Some(&mut texture)).context("creating the video texture")?;
            let texture = texture.context("no video texture")?;
            let mut view = None;
            device.CreateShaderResourceView(&texture, None, Some(&mut view))?;
            let view = view.context("no view of the video texture")?;
            let converter = match coded {
                Some(coded) => Some(Converter::new(device, &context, &texture, size, coded)?),
                None => None,
            };
            Ok(Self {
                device: device.clone(),
                context,
                size,
                texture,
                view,
                converter,
            })
        }
    }

    /// Converts slice `slice` of the decoder's `texture` into the target.
    fn convert(&self, texture: &ID3D11Texture2D, slice: u32, frame: &ffmpeg_next::frame::Video) -> Result<()> {
        let converter = self.converter.as_ref().context("a target for software frames")?;
        // SAFETY: reads plain colour fields of a live frame.
        let (bt709, full) = unsafe {
            let raw = &*frame.as_ptr();
            (is_bt709(raw.colorspace, frame.height()), raw.color_range == ffi::AVColorRange::AVCOL_RANGE_JPEG)
        };
        // `D3D11_VIDEO_PROCESSOR_COLOR_SPACE`'s bits: Usage (playback: 0),
        // RGB_Range (0–255: 0), YCbCr_Matrix (BT.709: 1), YCbCr_xvYCC, then
        // two of Nominal_Range (16–235: 1, 0–255: 2).
        let stream_colours = D3D11_VIDEO_PROCESSOR_COLOR_SPACE {
            _bitfield: (bt709 as u32) << 2 | (if full { 2 } else { 1 }) << 4,
        };
        let output_colours = D3D11_VIDEO_PROCESSOR_COLOR_SPACE { _bitfield: 0 };
        let picture = RECT {
            left: 0,
            top: 0,
            right: self.size.0 as i32,
            bottom: self.size.1 as i32,
        };
        let input_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: slice,
                },
            },
        };
        // SAFETY: the views, processor and textures are live and of one
        // device; the stream's reference to the input view is released below.
        unsafe {
            // ponytail: one input view made per frame; keep them by slice if
            // profiling says so.
            let mut input: Option<ID3D11VideoProcessorInputView> = None;
            converter
                .video_device
                .CreateVideoProcessorInputView(texture, &converter.enumerator, &input_desc, Some(&mut input))
                .context("viewing the decoder's texture")?;
            let (context, processor) = (&converter.video_context, &converter.processor);
            context.VideoProcessorSetStreamSourceRect(processor, 0, true, Some(&picture));
            context.VideoProcessorSetStreamColorSpace(processor, 0, &stream_colours);
            context.VideoProcessorSetOutputColorSpace(processor, &output_colours);
            let mut stream = D3D11_VIDEO_PROCESSOR_STREAM {
                Enable: true.into(),
                pInputSurface: ManuallyDrop::new(input),
                ..Default::default()
            };
            let result = context.VideoProcessorBlt(processor, &converter.output, 0, std::slice::from_ref(&stream));
            ManuallyDrop::drop(&mut stream.pInputSurface);
            result.context("converting the frame")
        }
    }

    /// Copies a software frame ([`super::ffmpeg::SOFTWARE`]: BGRA) into the
    /// target, turned clockwise by `rotation`.
    fn upload(&self, frame: &ffmpeg_next::frame::Video, rotation: i32) -> Result<()> {
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: the texture is mapped while its `RowPitch`-wide rows are
        // written, `size.1` of them.
        unsafe {
            self.context.Map(&self.texture, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))?;
            let pitch = mapped.RowPitch as usize;
            super::copy_turned(
                frame.data(0),
                frame.stride(0),
                (frame.width() as usize, frame.height() as usize),
                4,
                rotation,
                std::slice::from_raw_parts_mut(mapped.pData.cast::<u8>(), pitch * self.size.1 as usize),
                pitch,
            );
            self.context.Unmap(&self.texture, 0);
        }
        Ok(())
    }
}

impl Converter {
    /// A video processor from `coded`-sized decoder textures into `texture`.
    ///
    /// # Safety
    /// `texture` must be a render target of `device`, whose context is `context`.
    unsafe fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        texture: &ID3D11Texture2D,
        size: (u32, u32),
        coded: (u32, u32),
    ) -> Result<Self> {
        let video_device: ID3D11VideoDevice = device.cast()?;
        let video_context: ID3D11VideoContext = context.cast()?;
        // The rates only steer deinterlacing, which progressive frames skip.
        let rate = DXGI_RATIONAL {
            Numerator: 60,
            Denominator: 1,
        };
        let content = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: rate,
            InputWidth: coded.0,
            InputHeight: coded.1,
            OutputFrameRate: rate,
            OutputWidth: size.0,
            OutputHeight: size.1,
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };
        let output_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
            },
        };
        // SAFETY: plain creation calls, each result checked.
        unsafe {
            let enumerator = video_device.CreateVideoProcessorEnumerator(&content).context("no video processor")?;
            let processor = video_device.CreateVideoProcessor(&enumerator, 0)?;
            let mut output = None;
            video_device.CreateVideoProcessorOutputView(texture, &enumerator, &output_desc, Some(&mut output))?;
            // The picture as decoded: none of the driver's "enhancements".
            video_context.VideoProcessorSetStreamAutoProcessingMode(&processor, 0, false);
            Ok(Self {
                coded,
                video_device,
                video_context,
                enumerator,
                processor,
                output: output.context("no view of the video texture for the processor")?,
            })
        }
    }
}
