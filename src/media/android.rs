//! Android: `AMediaExtractor` + `AMediaCodec` decoding into an `AImageReader`
//! on a background thread, and the GPU import of each frame on the main
//! thread.
//!
//! The import (ported from lumina-video's `android_import.rs`) wraps the
//! frame's `AHardwareBuffer` in a `VkImage` with the buffer's external format
//! and draws it once through a YCbCr-converting sampler into an RGBA8 texture.
//! No pixels pass through the CPU. It needs the Vulkan device extensions that
//! the vendored `gpui-pre-wgpu` enables on Android.

use anyhow::{anyhow, bail, Context as _, Result};
use ash::vk;
use gpui::{DevicePixels, ImageId, Size};
use gpui_wgpu::wgpu;
use ndk_sys as ffi;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Images the app may hold at once: the newest, the one still being drawn,
/// and one in flight on the GPU.
const MAX_IMAGES: i32 = 4;
const DEQUEUE_TIMEOUT_US: i64 = 10_000;

pub(super) struct Decoder {
    reader: Reader,
    size: Size<DevicePixels>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    importer: RefCell<Importer>,
    frames_shown: Cell<u64>,
}

impl Decoder {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
        let length = file.metadata()?.len() as i64;
        let mut extractor = Extractor::new()?;
        // SAFETY: the descriptor is open for the call; the extractor dups what it keeps.
        status(unsafe {
            ffi::AMediaExtractor_setDataSourceFd(extractor.0.as_ptr(), file.as_raw_fd(), 0, length)
        })
        .context("AMediaExtractor_setDataSourceFd")?;
        let (track, format, mime, width, height) = extractor.video_track()?;

        let reader = Reader::new(width, height)?;
        let mime_c = CString::new(mime.clone())?;
        // SAFETY: `mime_c` is a valid C string for the call.
        let codec = NonNull::new(unsafe { ffi::AMediaCodec_createDecoderByType(mime_c.as_ptr()) })
            .map(Codec)
            .ok_or_else(|| anyhow!("no decoder for {mime}"))?;
        // SAFETY: the format and the reader's window are live; the window stays
        // valid until the reader is deleted, which `Decoder::drop` does after
        // the decoding thread has deleted the codec.
        unsafe {
            status(ffi::AMediaCodec_configure(
                codec.0.as_ptr(),
                format.0.as_ptr(),
                reader.window()?,
                ptr::null_mut(),
                0,
            ))
            .context("AMediaCodec_configure")?;
            status(ffi::AMediaExtractor_selectTrack(extractor.0.as_ptr(), track))
                .context("AMediaExtractor_selectTrack")?;
            status(ffi::AMediaCodec_start(codec.0.as_ptr())).context("AMediaCodec_start")?;
        }
        log::info!("video: decoding {mime} {width}x{height} from {path:?}");

        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new()
            .name("gpui-video-decode".into())
            .spawn({
                let stop = stop.clone();
                let codec = SendPtr(codec);
                let extractor = SendPtr(extractor);
                move || {
                    let (codec, extractor) = (codec, extractor);
                    if let Err(error) = decode_loop(&codec.0, &extractor.0, &stop) {
                        log::error!("video: decoding stopped: {error:#}");
                    }
                }
            })?;
        Ok(Self {
            reader,
            size: Size {
                width: DevicePixels(width),
                height: DevicePixels(height),
            },
            stop,
            thread: Some(thread),
            importer: RefCell::new(Importer::default()),
            frames_shown: Cell::new(0),
        })
    }

    pub fn video_size(&self) -> Size<DevicePixels> {
        self.size
    }

    pub fn frames_shown(&self) -> u64 {
        self.frames_shown.get()
    }

    /// Imports the newest decoded image, if there is one, and makes it the
    /// texture drawn for `image_id`.
    pub fn update(&self, image_id: ImageId) -> Result<()> {
        let Some(image) = self.reader.acquire_latest()? else {
            return Ok(());
        };
        let platform =
            crate::android::jni::platform().ok_or_else(|| anyhow!("platform not initialised"))?;
        let context = platform.gpu_context();
        let context = context.borrow();
        let context = context.as_ref().ok_or_else(|| anyhow!("no GPU context yet"))?;
        // SAFETY: the image came from our GPU-sampleable reader and was
        // acquired with `acquireLatestImage`, which waits for the decoder's fence.
        let prepared = unsafe { self.importer.borrow_mut().prepare(image, &context.device)? };
        // Submitted before the frame that samples the texture, on the same queue.
        context.queue.submit(prepared.commands);
        gpui_wgpu::set_external_texture(image_id, &prepared.texture);
        let shown = self.frames_shown.get() + 1;
        self.frames_shown.set(shown);
        if shown == 1 || shown % 120 == 0 {
            log::info!(
                "video: AHardwareBuffer frame {shown} imported as a {}x{} Vulkan texture",
                prepared.texture.width(),
                prepared.texture.height()
            );
        }
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                log::error!("video: decoding thread panicked");
            }
        }
    }
}

/// Feeds samples to the codec and renders each output buffer into the reader
/// when its timestamp is due, looping at the end of the stream.
fn decode_loop(codec: &Codec, extractor: &Extractor, stop: &AtomicBool) -> Result<()> {
    let codec = codec.0.as_ptr();
    let extractor = extractor.0.as_ptr();
    let mut input_done = false;
    let mut clock: Option<(Instant, i64)> = None;
    while !stop.load(Ordering::Relaxed) {
        // SAFETY: codec and extractor are owned by this thread for the loop;
        // buffer indices and pointers come from the codec just before use.
        unsafe {
            if !input_done {
                let index = ffi::AMediaCodec_dequeueInputBuffer(codec, DEQUEUE_TIMEOUT_US);
                if index >= 0 {
                    let index = index as usize;
                    let mut capacity = 0;
                    let buffer = ffi::AMediaCodec_getInputBuffer(codec, index, &mut capacity);
                    let size = ffi::AMediaExtractor_readSampleData(extractor, buffer, capacity);
                    if size < 0 {
                        input_done = true;
                        status(ffi::AMediaCodec_queueInputBuffer(
                            codec,
                            index,
                            0,
                            0,
                            0,
                            ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32,
                        ))?;
                    } else {
                        let time = ffi::AMediaExtractor_getSampleTime(extractor).max(0);
                        status(ffi::AMediaCodec_queueInputBuffer(
                            codec,
                            index,
                            0,
                            size as usize,
                            time as u64,
                            0,
                        ))?;
                        ffi::AMediaExtractor_advance(extractor);
                    }
                }
            }

            let mut info = std::mem::zeroed::<ffi::AMediaCodecBufferInfo>();
            let index = ffi::AMediaCodec_dequeueOutputBuffer(codec, &mut info, DEQUEUE_TIMEOUT_US);
            if index < 0 {
                // Try-again, format and buffer changes need nothing here.
                continue;
            }
            let end = info.flags & ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32 != 0;
            if !end && info.size > 0 {
                let (start, first) = *clock.get_or_insert((Instant::now(), info.presentationTimeUs));
                let due = start + Duration::from_micros((info.presentationTimeUs - first).max(0) as u64);
                // ponytail: presents on a wall clock started at the first frame; an
                // audio clock replaces it in the playback step.
                while let Some(wait) = due.checked_duration_since(Instant::now()) {
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    std::thread::sleep(wait.min(Duration::from_millis(20)));
                }
            }
            status(ffi::AMediaCodec_releaseOutputBuffer(codec, index as usize, !end && info.size > 0))?;
            if end {
                status(ffi::AMediaExtractor_seekTo(
                    extractor,
                    0,
                    ffi::SeekMode::AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC,
                ))?;
                status(ffi::AMediaCodec_flush(codec))?;
                input_done = false;
                clock = None;
            }
        }
    }
    Ok(())
}

fn status(status: ffi::media_status_t) -> Result<()> {
    if status == ffi::media_status_t::AMEDIA_OK {
        Ok(())
    } else {
        bail!("media status {}", status.0)
    }
}

struct SendPtr<T>(T);
// SAFETY: the codec and extractor move to the decoding thread and are only
// used there from then on.
unsafe impl<T> Send for SendPtr<T> {}

struct Extractor(NonNull<ffi::AMediaExtractor>);

impl Extractor {
    fn new() -> Result<Self> {
        // SAFETY: plain constructor.
        NonNull::new(unsafe { ffi::AMediaExtractor_new() })
            .map(Self)
            .ok_or_else(|| anyhow!("AMediaExtractor_new failed"))
    }

    /// The first video track: its index, format, MIME type and size.
    fn video_track(&mut self) -> Result<(usize, Format, String, i32, i32)> {
        // SAFETY: the extractor is live; formats are owned by `Format` and the
        // strings they return stay valid while the format lives.
        unsafe {
            for track in 0..ffi::AMediaExtractor_getTrackCount(self.0.as_ptr()) {
                let Some(format) =
                    NonNull::new(ffi::AMediaExtractor_getTrackFormat(self.0.as_ptr(), track))
                        .map(Format)
                else {
                    continue;
                };
                let mut mime = ptr::null();
                if !ffi::AMediaFormat_getString(format.0.as_ptr(), ffi::AMEDIAFORMAT_KEY_MIME, &mut mime)
                    || mime.is_null()
                {
                    continue;
                }
                let mime = CStr::from_ptr(mime).to_string_lossy().into_owned();
                if !mime.starts_with("video/") {
                    continue;
                }
                let (mut width, mut height) = (0, 0);
                ffi::AMediaFormat_getInt32(format.0.as_ptr(), ffi::AMEDIAFORMAT_KEY_WIDTH, &mut width);
                ffi::AMediaFormat_getInt32(format.0.as_ptr(), ffi::AMEDIAFORMAT_KEY_HEIGHT, &mut height);
                if width <= 0 || height <= 0 {
                    bail!("video track without a size");
                }
                return Ok((track, format, mime, width, height));
            }
        }
        bail!("no video track")
    }
}

impl Drop for Extractor {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once.
        unsafe { ffi::AMediaExtractor_delete(self.0.as_ptr()) };
    }
}

struct Format(NonNull<ffi::AMediaFormat>);

impl Drop for Format {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once.
        unsafe { ffi::AMediaFormat_delete(self.0.as_ptr()) };
    }
}

struct Codec(NonNull<ffi::AMediaCodec>);

impl Drop for Codec {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once; stopping first returns its buffers.
        unsafe {
            ffi::AMediaCodec_stop(self.0.as_ptr());
            ffi::AMediaCodec_delete(self.0.as_ptr());
        }
    }
}

struct Reader(NonNull<ffi::AImageReader>);

impl Reader {
    fn new(width: i32, height: i32) -> Result<Self> {
        let mut reader = ptr::null_mut();
        // SAFETY: output pointer to local storage.
        status(unsafe {
            ffi::AImageReader_newWithUsage(
                width,
                height,
                ffi::AIMAGE_FORMATS::AIMAGE_FORMAT_PRIVATE.0 as i32,
                ffi::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0 as u64,
                MAX_IMAGES,
                &mut reader,
            )
        })
        .context("AImageReader_newWithUsage")?;
        NonNull::new(reader)
            .map(Self)
            .ok_or_else(|| anyhow!("AImageReader_newWithUsage returned null"))
    }

    fn window(&self) -> Result<*mut ffi::ANativeWindow> {
        let mut window = ptr::null_mut();
        // SAFETY: the reader is live; the window it returns is owned by it.
        status(unsafe { ffi::AImageReader_getWindow(self.0.as_ptr(), &mut window) })?;
        Ok(window)
    }

    fn acquire_latest(&self) -> Result<Option<Arc<AcquiredImage>>> {
        let mut image = ptr::null_mut();
        // SAFETY: the reader is live and `image` is local output storage.
        let result = unsafe { ffi::AImageReader_acquireLatestImage(self.0.as_ptr(), &mut image) };
        if result == ffi::media_status_t::AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE
            || result == ffi::media_status_t::AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED
        {
            return Ok(None);
        }
        status(result).context("AImageReader_acquireLatestImage")?;
        let image = NonNull::new(image).ok_or_else(|| anyhow!("null AImage"))?;
        // SAFETY: the image is live; `AcquiredImage` owns it (and the buffer
        // reference taken here) from here on, so error paths release both.
        unsafe {
            let mut acquired = AcquiredImage {
                image,
                buffer: ptr::null_mut(),
                buffer_size: (0, 0),
                crop: vk::Rect2D::default(),
            };
            let mut buffer = ptr::null_mut();
            status(ffi::AImage_getHardwareBuffer(image.as_ptr(), &mut buffer))?;
            if buffer.is_null() {
                bail!("AImage without a hardware buffer");
            }
            ffi::AHardwareBuffer_acquire(buffer);
            acquired.buffer = buffer;
            let mut description = std::mem::zeroed::<ffi::AHardwareBuffer_Desc>();
            ffi::AHardwareBuffer_describe(buffer, &mut description);
            acquired.buffer_size = (description.width, description.height);
            let mut crop = std::mem::zeroed::<ffi::AImageCropRect>();
            status(ffi::AImage_getCropRect(image.as_ptr(), &mut crop))?;
            acquired.crop = vk::Rect2D {
                offset: vk::Offset2D {
                    x: crop.left,
                    y: crop.top,
                },
                extent: vk::Extent2D {
                    width: (crop.right - crop.left).max(1) as u32,
                    height: (crop.bottom - crop.top).max(1) as u32,
                },
            };
            Ok(Some(Arc::new(acquired)))
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        // SAFETY: owned pointer, deleted once, after the codec writing into it.
        unsafe { ffi::AImageReader_delete(self.0.as_ptr()) };
    }
}

/// A decoded image held until the GPU has finished reading its buffer, so the
/// codec cannot reuse the buffer underneath the conversion.
struct AcquiredImage {
    image: NonNull<ffi::AImage>,
    buffer: *mut ffi::AHardwareBuffer,
    buffer_size: (u32, u32),
    crop: vk::Rect2D,
}

// SAFETY: AImage and AHardwareBuffer handles may be released from any thread;
// nothing reads through them after acquisition.
unsafe impl Send for AcquiredImage {}
unsafe impl Sync for AcquiredImage {}

impl Drop for AcquiredImage {
    fn drop(&mut self) {
        // SAFETY: the buffer reference was taken in `acquire_latest`; the
        // image is owned and deleted once.
        unsafe {
            if !self.buffer.is_null() {
                ffi::AHardwareBuffer_release(self.buffer);
            }
            ffi::AImage_delete(self.image.as_ptr());
        }
    }
}

struct PreparedFrame {
    commands: Vec<wgpu::CommandBuffer>,
    texture: Arc<wgpu::Texture>,
}

#[derive(Default)]
struct Importer {
    pipeline: Option<Arc<ConversionPipeline>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ConversionKey {
    external_format: u64,
    model: vk::SamplerYcbcrModelConversion,
    range: vk::SamplerYcbcrRange,
    x: vk::ChromaLocation,
    y: vk::ChromaLocation,
    components: [vk::ComponentSwizzle; 4],
}

struct ConversionPipeline {
    device: ash::Device,
    key: ConversionKey,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

impl Drop for ConversionPipeline {
    fn drop(&mut self) {
        // SAFETY: every frame retains this pipeline until its GPU work is done;
        // the handles belong to this device and are destroyed in dependency order.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_sampler_ycbcr_conversion(self.conversion, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

struct FrameResources {
    pipeline: Arc<ConversionPipeline>,
    producer: Arc<AcquiredImage>,
    input: vk::Image,
    input_memory: vk::DeviceMemory,
    input_view: vk::ImageView,
    output: vk::Image,
    output_memory: vk::DeviceMemory,
    output_view: vk::ImageView,
    descriptor_pool: vk::DescriptorPool,
    framebuffer: vk::Framebuffer,
}

impl Drop for FrameResources {
    fn drop(&mut self) {
        // SAFETY: before submission this object owns every handle; afterwards
        // the output texture's HAL drop callback runs only once the GPU is done
        // with it. Views and the framebuffer go before images and memory.
        unsafe {
            let device = &self.pipeline.device;
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_image_view(self.input_view, None);
            device.destroy_image_view(self.output_view, None);
            device.destroy_image(self.input, None);
            device.destroy_image(self.output, None);
            device.free_memory(self.input_memory, None);
            device.free_memory(self.output_memory, None);
        }
    }
}

fn memory_type(bits: u32) -> Result<u32> {
    if bits == 0 {
        bail!("no compatible Vulkan memory type");
    }
    Ok(bits.trailing_zeros())
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

impl Importer {
    /// Records the conversion of `frame` into a new RGBA8 texture of its crop
    /// size. The caller submits `commands` before sampling `texture`.
    ///
    /// # Safety
    /// `frame` must come from a `GPU_SAMPLED_IMAGE` reader with its producer
    /// fence already waited on, and `device` must be GPUI's Vulkan device.
    unsafe fn prepare(
        &mut self,
        frame: Arc<AcquiredImage>,
        device: &wgpu::Device,
    ) -> Result<PreparedFrame> {
        // SAFETY: a live wgpu device; no raw queue operations happen here.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| anyhow!("video frames need the Vulkan backend"))?;
        for extension in [
            ash::android::external_memory_android_hardware_buffer::NAME,
            ash::ext::queue_family_foreign::NAME,
        ] {
            if !hal.enabled_device_extensions().contains(&extension) {
                bail!("Vulkan device lacks {extension:?}");
            }
        }
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let android = ash::android::external_memory_android_hardware_buffer::Device::new(instance, raw);
        let mut format = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
        let (allocation_size, memory_bits) = {
            let mut properties =
                vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format);
            // SAFETY: the buffer is referenced by `frame`; the output chain is local.
            unsafe { android.get_android_hardware_buffer_properties(frame.buffer.cast(), &mut properties) }?;
            (properties.allocation_size, properties.memory_type_bits)
        };
        if !format.format_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            bail!("AHardwareBuffer cannot be sampled");
        }
        // Always import through the external format: the buffer's reported
        // features and suggested conversion describe it, even when a matching
        // VkFormat exists.
        let key = ConversionKey {
            external_format: format.external_format,
            model: format.suggested_ycbcr_model,
            range: format.suggested_ycbcr_range,
            x: format.suggested_x_chroma_offset,
            y: format.suggested_y_chroma_offset,
            components: [
                format.sampler_ycbcr_conversion_components.r,
                format.sampler_ycbcr_conversion_components.g,
                format.sampler_ycbcr_conversion_components.b,
                format.sampler_ycbcr_conversion_components.a,
            ],
        };
        if key.external_format == 0 {
            bail!("AHardwareBuffer has no external format");
        }
        if self.pipeline.as_ref().is_none_or(|pipeline| pipeline.key != key) {
            // SAFETY: the device enabled samplerYcbcrConversion (vendored
            // gpui-pre-wgpu) and the key comes from this device's AHB properties.
            self.pipeline = Some(Arc::new(unsafe { ConversionPipeline::new(raw.clone(), key) }?));
        }
        let pipeline = self.pipeline.clone().ok_or_else(|| anyhow!("missing pipeline"))?;
        let (buffer_width, buffer_height) = frame.buffer_size;
        let crop = frame.crop;
        let (width, height) = (crop.extent.width, crop.extent.height);
        let mut resources = FrameResources {
            pipeline,
            producer: frame,
            input: vk::Image::null(),
            input_memory: vk::DeviceMemory::null(),
            input_view: vk::ImageView::null(),
            output: vk::Image::null(),
            output_memory: vk::DeviceMemory::null(),
            output_view: vk::ImageView::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            framebuffer: vk::Framebuffer::null(),
        };
        let mut external_memory = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
        let mut external_format = vk::ExternalFormatANDROID::default().external_format(key.external_format);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::UNDEFINED)
            .extent(vk::Extent3D {
                width: buffer_width,
                height: buffer_height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory)
            .push_next(&mut external_format);
        // SAFETY: the descriptors match the queried buffer; `resources` owns
        // every handle created here and frees partial work on any error.
        unsafe {
            resources.input = raw.create_image(&image_info, None)?;
            let mut import =
                vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(resources.producer.buffer.cast());
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(resources.input);
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(memory_type(memory_bits)?)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            resources.input_memory = raw.allocate_memory(&allocation, None)?;
            raw.bind_image_memory(resources.input, resources.input_memory, 0)?;
            let mut conversion =
                vk::SamplerYcbcrConversionInfo::default().conversion(resources.pipeline.conversion);
            resources.input_view = raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(resources.input)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::UNDEFINED)
                    .subresource_range(color_range())
                    .push_next(&mut conversion),
                None,
            )?;

            resources.output = raw.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let requirements = raw.get_image_memory_requirements(resources.output);
            resources.output_memory = raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type(requirements.memory_type_bits)?),
                None,
            )?;
            raw.bind_image_memory(resources.output, resources.output_memory, 0)?;
            resources.output_view = raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(resources.output)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(color_range()),
                None,
            )?;
            // The spec bounds combined YCbCr descriptors by the plane count;
            // four covers external formats with alpha.
            let pool_size = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4,
            }];
            resources.descriptor_pool = raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_size),
                None,
            )?;
            let layouts = [resources.pipeline.descriptor_layout];
            let descriptor_set = raw
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool)
                        .set_layouts(&layouts),
                )?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no descriptor set returned"))?;
            let image = [vk::DescriptorImageInfo::default()
                .image_view(resources.input_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            raw.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image)],
                &[],
            );
            let attachments = [resources.output_view];
            resources.framebuffer = raw.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(resources.pipeline.render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?;

            let resources = Arc::new(resources);
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let retained = Arc::clone(&resources);
            let texture = hal.texture_from_raw(
                resources.output,
                &wgpu::hal::TextureDescriptor {
                    label: Some("video frame"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COLOR_TARGET,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: Vec::new(),
                },
                Some(Box::new(move || drop(retained))),
                wgpu::hal::vulkan::TextureMemory::External,
            );
            let texture = device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                texture,
                &wgpu::TextureDescriptor {
                    label: Some("video frame"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
                wgpu::TextureUses::UNINITIALIZED,
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("video frame initialization"),
            });
            // Clear through wgpu so its init tracker won't clear over the raw
            // conversion later; this also ties the output's lifetime to the
            // submission.
            let output_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("video frame initialization"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            encoder.transition_resources(
                std::iter::empty(),
                std::iter::once(wgpu::TextureTransition {
                    texture: &texture,
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                }),
            );
            let initialize = encoder.finish();
            // wgpu 30 doesn't allow mixing wgpu and raw encoding on one encoder.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("video frame YCbCr conversion"),
            });
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                let encoder = encoder.ok_or_else(|| anyhow!("missing Vulkan command encoder"))?;
                resources.record(
                    encoder.raw_handle(),
                    hal.queue_family_index(),
                    descriptor_set,
                    crop,
                    (buffer_width, buffer_height),
                );
                anyhow::Ok(())
            })?;
            let texture = Arc::new(texture);
            // Raw encoding bypasses wgpu's tracker: keep the texture (and the
            // decoded image behind it) alive until this buffer has run.
            let retained = Arc::clone(&texture);
            encoder.on_submitted_work_done(move || drop(retained));
            Ok(PreparedFrame {
                commands: vec![initialize, encoder.finish()],
                texture,
            })
        }
    }
}

impl FrameResources {
    /// Draws the crop rectangle of the input into the whole output. The
    /// viewport spans the full buffer, offset so the crop lands at the origin.
    unsafe fn record(
        &self,
        command: vk::CommandBuffer,
        family: u32,
        descriptor: vk::DescriptorSet,
        crop: vk::Rect2D,
        (buffer_width, buffer_height): (u32, u32),
    ) {
        let raw = &self.pipeline.device;
        let acquire = vk::ImageMemoryBarrier::default()
            .image(self.input)
            .subresource_range(color_range())
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(family);
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: crop.extent,
        };
        // SAFETY: an active, exclusively borrowed wgpu Vulkan command buffer;
        // every resource and the decoded image outlive its submission.
        unsafe {
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[acquire],
            );
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            }];
            raw.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pipeline.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(area)
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout,
                0,
                &[descriptor],
                &[],
            );
            raw.cmd_set_viewport(
                command,
                0,
                &[vk::Viewport {
                    x: -crop.offset.x as f32,
                    y: -crop.offset.y as f32,
                    width: buffer_width as f32,
                    height: buffer_height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            raw.cmd_set_scissor(command, 0, &[area]);
            raw.cmd_draw(command, 3, 1, 0, 0);
            raw.cmd_end_render_pass(command);
            let release = vk::ImageMemoryBarrier::default()
                .image(self.input)
                .subresource_range(color_range())
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT);
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[release],
            );
        }
    }
}

impl ConversionPipeline {
    unsafe fn new(device: ash::Device, key: ConversionKey) -> Result<Self> {
        let mut state = Self {
            device,
            key,
            conversion: vk::SamplerYcbcrConversion::null(),
            sampler: vk::Sampler::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
        };
        // SAFETY: the device enabled samplerYcbcrConversion, the key is from
        // its AHB properties, and `state` frees partial work on errors.
        unsafe {
            let mut external = vk::ExternalFormatANDROID::default().external_format(key.external_format);
            let [r, g, b, a] = key.components;
            state.conversion = state.device.create_sampler_ycbcr_conversion(
                &vk::SamplerYcbcrConversionCreateInfo::default()
                    .format(vk::Format::UNDEFINED)
                    .ycbcr_model(key.model)
                    .ycbcr_range(key.range)
                    .components(vk::ComponentMapping { r, g, b, a })
                    .x_chroma_offset(key.x)
                    .y_chroma_offset(key.y)
                    .chroma_filter(vk::Filter::NEAREST)
                    .push_next(&mut external),
                None,
            )?;
            let mut conversion = vk::SamplerYcbcrConversionInfo::default().conversion(state.conversion);
            state.sampler = state.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0)
                    .push_next(&mut conversion),
                None,
            )?;
            let samplers = [state.sampler];
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&samplers)];
            state.descriptor_layout = state.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?;
            state.pipeline_layout = state.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[state.descriptor_layout]),
                None,
            )?;
            let attachment = [vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let color = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let subpass = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color)];
            let dependencies = [
                // The wgpu initialization pass runs first on the same queue.
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::SubpassDependency::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            ];
            state.render_pass = state.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachment)
                    .subpasses(&subpass)
                    .dependencies(&dependencies),
                None,
            )?;
            // Built from the GLSL next to them with
            // `glslc -o shaders/<name>.spv shaders/<name>`.
            let vertex = shader(&state.device, include_bytes!("shaders/blit.vert.spv"))?;
            let fragment = match shader(&state.device, include_bytes!("shaders/ycbcr_blit.frag.spv")) {
                Ok(shader) => shader,
                Err(error) => {
                    state.device.destroy_shader_module(vertex, None);
                    return Err(error);
                }
            };
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment)
                    .name(c"main"),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend =
                [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
            let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .dynamic_state(&dynamic_state)
                .layout(state.pipeline_layout)
                .render_pass(state.render_pass);
            let result = state
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
            state.device.destroy_shader_module(fragment, None);
            state.device.destroy_shader_module(vertex, None);
            match result {
                Ok(pipelines) => {
                    state.pipeline = pipelines
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("no conversion pipeline returned"))?
                }
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        state.device.destroy_pipeline(pipeline, None);
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(state)
    }
}

fn shader(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    // SAFETY: shipped SPIR-V; Vulkan copies the words.
    Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }?)
}
