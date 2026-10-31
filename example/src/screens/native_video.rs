//! Video decoded by the platform and drawn inside the GPUI scene
//! (`gpui_mobile::media`), with GPUI content layered over the frames.

use std::cell::RefCell;

use gpui::{canvas, div, prelude::*, rgb, rgba, AnyElement, App};
use gpui_mobile::media::VideoFrames;

use super::{SURFACE0, SUBTEXT, TEXT};

const SAMPLE: &[u8] = include_bytes!("../../assets/video/testsrc-640x360.mp4");

thread_local! {
    static VIDEO: RefCell<Option<Result<VideoFrames, String>>> = const { RefCell::new(None) };
}

fn open_sample() -> Result<VideoFrames, String> {
    let directory = gpui_mobile::packages::path_provider::cache_directory()?;
    let path = directory.join("testsrc-640x360.mp4");
    std::fs::write(&path, SAMPLE).map_err(|error| error.to_string())?;
    VideoFrames::open(&path).map_err(|error| format!("{error:#}"))
}

/// Stops decoding when the screen is left.
pub fn reset_state() {
    VIDEO.with(|video| video.borrow_mut().take());
}

pub fn render(_cx: &App) -> AnyElement {
    let (frames, size, error) = VIDEO.with(|video| {
        let mut video = video.borrow_mut();
        let video = video.get_or_insert_with(open_sample);
        match video {
            Ok(frames) => (frames.frames_shown(), frames.video_size(), None),
            Err(error) => (0, Default::default(), Some(error.clone())),
        }
    });
    let aspect = if size.width.0 > 0 {
        size.height.0 as f32 / size.width.0 as f32
    } else {
        9.0 / 16.0
    };

    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .child(
            div()
                .text_sm()
                .text_color(rgb(SUBTEXT))
                .child("Decoded by MediaCodec into an AHardwareBuffer, imported into Vulkan and drawn as a GPUI image. No CPU copies."),
        )
        .child(
            div()
                .relative()
                .w_full()
                .rounded_xl()
                .overflow_hidden()
                .bg(rgb(SURFACE0))
                .child(
                    canvas(
                        |_, _, _| {},
                        |bounds, _, window, _| {
                            VIDEO.with(|video| {
                                if let Some(Ok(frames)) = video.borrow().as_ref() {
                                    frames.paint(bounds, window);
                                }
                            })
                        },
                    )
                    .w_full()
                    .aspect_ratio(1.0 / aspect),
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
        .child(div().text_sm().text_color(rgb(TEXT)).child(match error {
            Some(error) => format!("Error: {error}"),
            None => format!("Frames imported: {frames}"),
        }))
        .into_any_element()
}
