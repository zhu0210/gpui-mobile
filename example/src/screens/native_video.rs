//! Video decoded by the platform and drawn inside the GPUI scene
//! (`gpui_mobile::media`), with GPUI content layered over the frames.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use gpui::{canvas, div, prelude::*, px, rgb, rgba, AnyElement, App, Corners, MouseDownEvent, Window};
use gpui_mobile::media::VideoFrames;

use super::{BLUE, GREEN, SUBTEXT, SURFACE0, TEXT};

/// The sample clips: name, file name and contents.
const CLIPS: [(&str, &str, &[u8]); 2] = [
    (
        "60 s with audio",
        "av-60s-640x360.mp4",
        include_bytes!("../../assets/video/av-60s-640x360.mp4"),
    ),
    (
        "Rotated 90°",
        "rotated-90.mp4",
        include_bytes!("../../assets/video/rotated-90.mp4"),
    ),
];

thread_local! {
    static VIDEO: RefCell<Option<Result<VideoFrames, String>>> = const { RefCell::new(None) };
    static CLIP: Cell<usize> = const { Cell::new(0) };
}

fn open_clip(index: usize) -> Result<VideoFrames, String> {
    let (_, name, bytes) = CLIPS[index];
    let directory = gpui_mobile::packages::path_provider::cache_directory()?;
    let path = directory.join(name);
    std::fs::write(&path, bytes).map_err(|error| error.to_string())?;
    let video = VideoFrames::open(&path).map_err(|error| format!("{error:#}"))?;
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

    let clip_buttons = CLIPS.iter().enumerate().map(|(index, (name, _, _))| {
        button(*name, if index == clip { GREEN } else { SUBTEXT }, move |_, window, _| {
            CLIP.with(|clip| clip.set(index));
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
                .child(if cfg!(target_os = "ios") {
                    "Played by AVPlayer, each frame an IOSurface wrapped as a Metal texture and drawn as a GPUI image. No CPU copies."
                } else {
                    "Decoded by MediaCodec into an AHardwareBuffer, imported into Vulkan and drawn as a GPUI image, paced by the AAudio clock. No CPU copies."
                }),
        )
        .child(div().flex().flex_wrap().gap_2().children(clip_buttons))
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
