//! `media::VideoFrames` in a desktop GPUI window (Linux): FFmpeg through
//! VA-API, DMA-BUF imports, sound through cpal, GPUI drawn over the video.
//!
//! ```sh
//! cargo run --example desktop_video --features ffmpeg -- [clip] [software]
//! ```
//!
//! `clip` defaults to the bundled 60 s clip; `software` forces software
//! decoding. `ZED_DEVICE_ID=0x<pci id>` picks the GPU.

use gpui::{
    canvas, div, prelude::*, px, rgb, rgba, size, App, Bounds, Context, Corners, Window,
    WindowBounds, WindowOptions,
};
use gpui_mobile::media::{Backend, VideoFrames};
use std::time::Duration;

struct Player {
    video: VideoFrames,
}

impl Render for Player {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let video = &self.video;
        let size = video.video_size();
        div()
            .size_full()
            .bg(rgb(0x101010))
            .child(
                canvas(|_, _, _| {}, {
                    let entity = cx.entity();
                    move |bounds, _, window, cx| entity.read(cx).video.paint(bounds, Corners::all(px(12.)), window)
                })
                .size_full(),
            )
            // GPUI content drawn over the video frames.
            .child(
                div()
                    .absolute()
                    .top_4()
                    .left_4()
                    .px_3()
                    .py_2()
                    .rounded_md()
                    .bg(rgba(0x000000aa))
                    .text_color(rgb(0xffffff))
                    .child(format!(
                        "GPUI overlay · {}x{} · {:.1}/{:.1} s · {} frames{}",
                        size.width.0,
                        size.height.0,
                        video.position().as_secs_f32(),
                        video.duration().as_secs_f32(),
                        video.frames_shown(),
                        if video.is_paused() { " · paused" } else { "" },
                    )),
            )
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    let clip = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/example/assets/video/av-60s-640x360.mp4").into()
    });
    let backend = match args.next().as_deref() {
        Some("software") => Backend::FfmpegSoftware,
        _ => Backend::Ffmpeg,
    };
    gpui_platform::application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(960.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                // Opened once the window has its GPU, so VA-API decodes on it.
                let video = VideoFrames::open_with(&clip, backend).expect("opening the clip");
                video.set_looping(true);
                let player = cx.new(|_| Player { video });
                // The overlay's text changes as the video plays.
                cx.spawn({
                    let player = player.downgrade();
                    async move |cx| loop {
                        cx.background_executor().timer(Duration::from_millis(250)).await;
                        if player.update(cx, |_, cx| cx.notify()).is_err() {
                            break;
                        }
                    }
                })
                .detach();
                player
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
