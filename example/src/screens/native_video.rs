//! Video decoded by the platform and drawn inside the GPUI scene
//! (`gpui_mobile::media`), with GPUI content layered over the frames.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use gpui::{canvas, div, prelude::*, px, rgb, rgba, AnyElement, App, Corners, MouseDownEvent, Window};
use gpui_mobile::media::{Backend, VideoFrames};

use super::{BLUE, GREEN, SUBTEXT, SURFACE0, TEXT};

enum Clip {
    /// A bundled file: its name and contents.
    Bundled(&'static str, &'static [u8]),
    Stream(&'static str),
    /// Bundled bytes played through `VideoFrames::open_reader`.
    #[cfg(feature = "ffmpeg")]
    Reader(&'static [u8]),
    /// A bundled AES-128 HLS stream whose playlist, key and segments Rust
    /// serves to FFmpeg (`VideoFrames::open_hls`): name and contents.
    #[cfg(feature = "ffmpeg")]
    Hls(&'static [(&'static str, &'static [u8])]),
}

/// The sample clips, by button label.
const CLIPS: &[(&str, Clip)] = &[
    (
        "60 s with audio",
        Clip::Bundled("av-60s-640x360.mp4", include_bytes!("../../assets/video/av-60s-640x360.mp4")),
    ),
    (
        "Rotated 90°",
        Clip::Bundled("rotated-90.mp4", include_bytes!("../../assets/video/rotated-90.mp4")),
    ),
    (
        "Stream",
        // The bundled clip again, served over HTTPS by jsDelivr (as video/mp4,
        // with ranges; AVPlayer won't play GitHub raw's octet-stream).
        Clip::Stream("https://cdn.jsdelivr.net/gh/zhu0210/gpui-mobile@2a29d54/example/assets/video/av-60s-640x360.mp4"),
    ),
    #[cfg(feature = "ffmpeg")]
    (
        "Rust reader",
        Clip::Reader(include_bytes!("../../assets/video/av-60s-640x360.mp4")),
    ),
    #[cfg(feature = "ffmpeg")]
    (
        "HLS from Rust",
        Clip::Hls(&[
            ("clip.m3u8", include_bytes!("../../assets/video/hls/clip.m3u8")),
            ("clip.key", include_bytes!("../../assets/video/hls/clip.key")),
            ("clip0.ts", include_bytes!("../../assets/video/hls/clip0.ts")),
            ("clip1.ts", include_bytes!("../../assets/video/hls/clip1.ts")),
            ("clip2.ts", include_bytes!("../../assets/video/hls/clip2.ts")),
        ]),
    ),
];

/// The decoders to choose from: name and backend.
const BACKENDS: &[(&str, Backend)] = &[
    ("Platform", Backend::Platform),
    #[cfg(feature = "ffmpeg")]
    ("FFmpeg", Backend::Ffmpeg),
    #[cfg(feature = "ffmpeg")]
    ("FFmpeg software", Backend::FfmpegSoftware),
];

thread_local! {
    static VIDEO: RefCell<Option<Result<VideoFrames, String>>> = const { RefCell::new(None) };
    // Smoke tests pick the first clip with GPUI_VIDEO_CLIP, e.g. 1 for the
    // rotated one.
    static CLIP: Cell<usize> = Cell::new(
        std::env::var("GPUI_VIDEO_CLIP")
            .ok()
            .and_then(|clip| clip.parse().ok())
            .filter(|&clip| clip < CLIPS.len())
            .unwrap_or(0),
    );
    // And the decoder with GPUI_VIDEO_BACKEND, an index into BACKENDS.
    static BACKEND: Cell<usize> = Cell::new(
        std::env::var("GPUI_VIDEO_BACKEND")
            .ok()
            .and_then(|backend| backend.parse().ok())
            .filter(|&backend| backend < BACKENDS.len())
            .unwrap_or(0),
    );
}

fn open_clip(index: usize) -> Result<VideoFrames, String> {
    let (_, backend) = BACKENDS[BACKEND.with(Cell::get)];
    let video = match CLIPS[index].1 {
        Clip::Bundled(name, bytes) => {
            let path = gpui_mobile::packages::path_provider::cache_directory()?.join(name);
            std::fs::write(&path, bytes).map_err(|error| error.to_string())?;
            VideoFrames::open_with(&path, backend)
        }
        // ponytail: streams always use the platform player; FFmpeg here is
        // built without network (Cirrus will feed it through its own IO).
        Clip::Stream(url) => VideoFrames::open_url(url),
        #[cfg(feature = "ffmpeg")]
        Clip::Reader(bytes) => VideoFrames::open_reader(
            move || Ok(Box::new(std::io::Cursor::new(bytes)) as Box<dyn gpui_mobile::media::ReadSeek>),
            backend,
        ),
        #[cfg(feature = "ffmpeg")]
        Clip::Hls(files) => VideoFrames::open_hls(
            "clip.m3u8",
            move |uri| {
                let (_, bytes) = files
                    .iter()
                    .find(|(name, _)| *name == uri)
                    .ok_or(std::io::ErrorKind::NotFound)?;
                Ok(bytes.to_vec())
            },
            backend,
        ),
    }
    .map_err(|error| format!("{error:#}"))?;
    video.set_looping(true);
    Ok(video)
}

/// Stops playback when the screen is left.
pub fn reset_state() {
    VIDEO.with(|video| video.borrow_mut().take());
}

/// Runs `action` on the open video and redraws.
fn with_video(action: impl Fn(&VideoFrames) + 'static) -> impl Fn(&MouseDownEvent, &mut Window, &mut App) {
    move |_, window, _| {
        VIDEO.with(|video| {
            if let Some(Ok(video)) = video.borrow().as_ref() {
                action(video);
            }
        });
        window.refresh();
    }
}

fn button(
    label: impl Into<String>,
    color: u32,
    handler: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .px_4()
        .py_2()
        .rounded_full()
        .bg(rgb(SURFACE0))
        .border_1()
        .border_color(rgb(color))
        .text_sm()
        .text_color(rgb(color))
        .child(label.into())
        .on_mouse_down(gpui::MouseButton::Left, handler)
}

fn clock(time: Duration) -> String {
    let seconds = time.as_secs();
    format!("{}:{:02}.{}", seconds / 60, seconds % 60, time.subsec_millis() / 100)
}

pub fn render(_cx: &App) -> AnyElement {
    let clip = CLIP.with(Cell::get);
    let (status, size, paused) = VIDEO.with(|video| {
        let mut video = video.borrow_mut();
        match video.get_or_insert_with(|| open_clip(clip)) {
            Ok(video) => (
                format!(
                    "{} / {} · frames imported: {}{}",
                    clock(video.position()),
                    clock(video.duration()),
                    video.frames_shown(),
                    if video.ended() { " · ended" } else { "" }
                ),
                video.video_size(),
                video.is_paused(),
            ),
            Err(error) => (format!("Error: {error}"), Default::default(), true),
        }
    });
    let aspect = if size.width.0 > 0 {
        size.width.0 as f32 / size.height.0 as f32
    } else {
        16.0 / 9.0
    };

    let clip_buttons = CLIPS.iter().enumerate().map(|(index, (name, _))| {
        button(*name, if index == clip { GREEN } else { SUBTEXT }, move |_, window, _| {
            CLIP.with(|clip| clip.set(index));
            reset_state();
            window.refresh();
        })
    });

    let backend = BACKEND.with(Cell::get);
    let backend_buttons = BACKENDS.iter().enumerate().map(|(index, (name, _))| {
        button(*name, if index == backend { GREEN } else { SUBTEXT }, move |_, window, _| {
            BACKEND.with(|backend| backend.set(index));
            reset_state();
            window.refresh();
        })
    });

    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .child(
            div()
                .text_sm()
                .text_color(rgb(SUBTEXT))
                .child(match (cfg!(target_os = "ios"), BACKENDS[backend].1 == Backend::Platform) {
                    (true, true) => "Played by AVPlayer, each frame an IOSurface wrapped as a Metal texture and drawn as a GPUI image. No CPU copies.",
                    (true, false) => "Demuxed by FFmpeg. Video through VideoToolbox into BGRA IOSurfaces, the same Metal import as AVPlayer's, or decoded in software with the planes converted on the GPU. No sound yet.",
                    (false, true) => "Decoded by MediaCodec into an AHardwareBuffer, imported into Vulkan and drawn as a GPUI image, paced by the AAudio clock. No CPU copies.",
                    (false, false) => "Demuxed by FFmpeg. Video through FFmpeg's MediaCodec decoder into the same AHardwareBuffer import, or decoded in software with the planes converted on the GPU; audio decoded by FFmpeg into AAudio.",
                }),
        )
        .child(div().flex().flex_wrap().gap_2().children(clip_buttons))
        .when(BACKENDS.len() > 1, |this| {
            this.child(div().flex().flex_wrap().gap_2().children(backend_buttons))
        })
        .child(
            div()
                .relative()
                .w_full()
                .child(
                    canvas(
                        |_, _, _| {},
                        |bounds, _, window, _| {
                            VIDEO.with(|video| {
                                if let Some(Ok(video)) = video.borrow().as_ref() {
                                    video.paint(bounds, Corners::all(px(16.)), window);
                                }
                            })
                        },
                    )
                    .w_full()
                    // Rotated clips are portrait; cap their height.
                    .aspect_ratio(aspect.max(0.75)),
                )
                // GPUI content drawn over the video frames.
                .child(
                    div()
                        .absolute()
                        .top_2()
                        .left_2()
                        .px_2()
                        .py_1()
                        .rounded_md()
                        .bg(rgba(0x000000aa))
                        .text_xs()
                        .text_color(rgb(TEXT))
                        .child(format!("GPUI overlay · {}x{}", size.width.0, size.height.0)),
                ),
        )
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap_2()
                .child(button(
                    if paused { "Play" } else { "Pause" },
                    BLUE,
                    with_video(|video| {
                        if video.is_paused() {
                            video.play();
                        } else {
                            video.pause();
                        }
                    }),
                ))
                .child(button(
                    "−10 s",
                    BLUE,
                    with_video(|video| video.seek(video.position().saturating_sub(Duration::from_secs(10)))),
                ))
                .child(button(
                    "+10 s",
                    BLUE,
                    with_video(|video| video.seek(video.position() + Duration::from_secs(10))),
                ))
                .child(button("Restart", BLUE, with_video(|video| video.seek(Duration::ZERO)))),
        )
        .child(div().text_sm().text_color(rgb(TEXT)).child(status))
        .into_any_element()
}
