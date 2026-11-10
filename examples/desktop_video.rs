//! `media::VideoFrames` in a desktop GPUI window, GPUI drawn over the video,
//! sound through cpal. Linux: FFmpeg through VA-API, DMA-BUF imports. macOS:
//! through VideoToolbox, pixel buffers drawn as surfaces. Windows: through
//! Direct3D 11 on the renderer's device.
//!
//! ```sh
//! cargo run --example desktop_video --features ffmpeg -- [clip|hls] [software|reader]
//! ```
//!
//! `clip` defaults to the bundled 60 s clip; `hls` plays the bundled AES-128
//! HLS fixture through `open_hls`, every byte fetched by Rust. `software`
//! forces software decoding; `reader` feeds the clip through `open_reader`. `ZED_DEVICE_ID=0x<pci id>` picks the GPU.
//! `GPUI_VIDEO_CONTROLS=1` plays at 1.5× with the last audio track and the
//! first subtitles (try `example/assets/video/tracks.mp4`).
//! `GPUI_VIDEO_QUIT_AFTER=<seconds>` quits then, failing if no frame was
//! shown (CI runs it so); `GPUI_VIDEO_MUTED=1` plays without sound.

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
                        "GPUI overlay · {}x{} · {:.1}/{:.1} s (buffered {:.1}) · {}× · audio {:?} · {} frames{}",
                        size.width.0,
                        size.height.0,
                        video.position().as_secs_f32(),
                        video.duration().as_secs_f32(),
                        video.buffered().as_secs_f32(),
                        video.rate(),
                        video.audio_track(),
                        video.frames_shown(),
                        if video.is_paused() { " · paused" } else { "" },
                    )),
            )
            .when_some(video.subtitle_text(), |this, text| {
                this.child(
                    div()
                        .absolute()
                        .bottom_8()
                        .left_0()
                        .right_0()
                        .flex()
                        .flex_col()
                        .items_center()
                        .text_xl()
                        .text_color(rgb(0xffffff))
                        .children(text.lines().map(|line| div().px_2().bg(rgba(0x000000cc)).child(line.to_owned())).collect::<Vec<_>>()),
                )
            })
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    let clip = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/example/assets/video/av-60s-640x360.mp4").into()
    });
    let mode = args.next();
    let backend = match mode.as_deref() {
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
                let video = match (clip.as_str(), mode.as_deref()) {
                    ("hls", _) => {
                        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/example/assets/video/hls");
                        VideoFrames::open_hls(
                            "clip.m3u8",
                            move |uri| std::fs::read(std::path::Path::new(dir).join(uri)),
                            backend,
                        )
                    }
                    (_, Some("reader")) => {
                        let clip = clip.clone();
                        VideoFrames::open_reader(
                            move || Ok(Box::new(std::fs::File::open(&clip)?) as Box<dyn gpui_mobile::media::ReadSeek>),
                            backend,
                        )
                    }
                    _ => VideoFrames::open_with(&clip, backend),
                }
                .expect("opening the clip");
                video.set_looping(true);
                video.set_muted(std::env::var("GPUI_VIDEO_MUTED").is_ok_and(|value| value == "1"));
                if std::env::var("GPUI_VIDEO_CONTROLS").is_ok_and(|value| value == "1") {
                    video.set_rate(1.5);
                    if let Some(track) = video.audio_tracks().last() {
                        video.set_audio_track(track.id);
                    }
                    video.set_subtitle_track(video.subtitle_tracks().first().map(|track| track.id));
                }
                let player = cx.new(|_| Player { video });
                if let Some(seconds) = std::env::var("GPUI_VIDEO_QUIT_AFTER").ok().and_then(|value| value.parse().ok()) {
                    cx.spawn({
                        let player = player.downgrade();
                        async move |cx| {
                            cx.background_executor().timer(Duration::from_secs_f32(seconds)).await;
                            let shown = player.read_with(cx, |player, _| player.video.frames_shown()).unwrap_or(0);
                            println!("frames shown: {shown}");
                            std::process::exit(if shown > 0 { 0 } else { 1 });
                        }
                    })
                    .detach();
                }
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
