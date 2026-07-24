//! Lumina Video demo: native decode into GPUI wgpu surfaces.

use std::cell::RefCell;
use std::time::Duration;

use gpui::{div, prelude::*, px, rgb, AnyElement};
use lumina_video::GpuiVideoPlayer;

use super::{Router, BLUE, LIGHT_CARD_BG, LIGHT_SUBTEXT, LIGHT_TEXT, SUBTEXT, SURFACE0, TEXT};

const VIDEOS: &[(&str, &str)] = &[
    ("H.264 MP4 — 720p", "https://lorem.video/720p_h264_10s"),
    ("AV1 MP4 — 720p", "https://lorem.video/720p_av1"),
    (
        "VP9/Opus WebM — 1080p",
        "https://lorem.video/test_1080p_vp9_30fps_30s_25crf_opus_128kbps.webm",
    ),
    ("Live HLS — cat", "https://lorem.video/hls/cat/"),
];

#[derive(Default)]
struct VideoState {
    player: Option<GpuiVideoPlayer>,
    selected: usize,
}

thread_local! {
    static VIDEO_STATE: RefCell<VideoState> = RefCell::new(VideoState::default());
}

pub fn reset_state() {
    VIDEO_STATE.with(|state| *state.borrow_mut() = VideoState::default());
}

pub fn dismiss() {
    VIDEO_STATE.with(|state| {
        if let Some(player) = state.borrow_mut().player.as_mut() {
            player.pause();
        }
    });
}

fn format_time(value: Duration) -> String {
    let seconds = value.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

fn load_selected() {
    VIDEO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let url = VIDEOS[state.selected].1;
        state.player = Some(
            GpuiVideoPlayer::new(url)
                .with_autoplay(true)
                .with_controls(false)
                .with_looping(false),
        );
    });
}

pub fn render(
    router: &Router,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<Router>,
) -> impl IntoElement {
    let needs_initial_player = VIDEO_STATE.with(|state| state.borrow().player.is_none());
    if needs_initial_player {
        load_selected();
    }

    VIDEO_STATE.with(|state| {
        if let Some(player) = state.borrow_mut().player.as_mut() {
            player.update(window, cx);
        }
    });

    let (has_player, playing, position, duration, progress, selected, surface): (
        bool,
        bool,
        Duration,
        Option<Duration>,
        f32,
        usize,
        AnyElement,
    ) = VIDEO_STATE.with(|state| {
        let state = state.borrow();
        match state.player.as_ref() {
            Some(player) => (
                true,
                player.is_playing(),
                player.position(),
                player.duration(),
                player.seek_progress(),
                state.selected,
                player.surface_element().into_any_element(),
            ),
            None => (
                false,
                false,
                Duration::ZERO,
                None,
                0.0,
                state.selected,
                div()
                    .size_full()
                    .bg(rgb(0x000000))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(rgb(0xaaaaaa))
                    .child("Choose a video")
                    .into_any_element(),
            ),
        }
    });

    let dark = router.dark_mode;
    let text = if dark { TEXT } else { LIGHT_TEXT };
    let subtext = if dark { SUBTEXT } else { LIGHT_SUBTEXT };
    let card = if dark { SURFACE0 } else { LIGHT_CARD_BG };
    let duration_text = duration.map(format_time).unwrap_or_else(|| "--:--".into());

    div()
        .flex()
        .flex_col()
        .size_full()
        .gap_4()
        .p_4()
        .text_color(rgb(text))
        .child(
            div()
                .w_full()
                .h(px(230.0))
                .rounded_lg()
                .overflow_hidden()
                .bg(rgb(0x000000))
                .child(surface),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_3()
                .child(
                    div()
                        .id("lumina-play")
                        .px_4()
                        .py_2()
                        .rounded_md()
                        .bg(rgb(BLUE))
                        .cursor_pointer()
                        .child(if playing { "Pause" } else { "Play" })
                        .on_click(cx.listener(|_, _, _, cx| {
                            let needs_load = VIDEO_STATE.with(|state| {
                                let mut state = state.borrow_mut();
                                match state.player.as_mut() {
                                    Some(player) if player.is_playing() => {
                                        player.pause();
                                        false
                                    }
                                    Some(player) => {
                                        player.play();
                                        false
                                    }
                                    None => true,
                                }
                            });
                            if needs_load {
                                load_selected();
                            }
                            cx.notify();
                        })),
                )
                .child(format!("{} / {}", format_time(position), duration_text)),
        )
        .child(
            div()
                .id("lumina-seek")
                .relative()
                .w_full()
                .h(px(24.0))
                .cursor_pointer()
                .child(
                    div()
                        .absolute()
                        .top(px(9.0))
                        .w_full()
                        .h(px(6.0))
                        .rounded_full()
                        .bg(rgb(0x45475a)),
                )
                .child(
                    div()
                        .absolute()
                        .top(px(9.0))
                        .w(gpui::relative(progress))
                        .h(px(6.0))
                        .rounded_full()
                        .bg(rgb(BLUE)),
                )
                .on_click(cx.listener(|_, event: &gpui::ClickEvent, window, cx| {
                    let bounds = window.mouse_position();
                    let _ = (event, bounds);
                    // GPUI Mobile currently exposes click events without element bounds;
                    // the Lumina desktop demo contains the full drag implementation.
                    cx.notify();
                })),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .children(VIDEOS.iter().enumerate().map(|(index, (name, _))| {
                    div()
                        .id(("lumina-video", index))
                        .p_3()
                        .rounded_md()
                        .bg(rgb(if index == selected { BLUE } else { card }))
                        .cursor_pointer()
                        .text_color(rgb(if index == selected { 0xffffff } else { text }))
                        .child(*name)
                        .on_click(cx.listener(move |_, _, _, cx| {
                            VIDEO_STATE.with(|state| state.borrow_mut().selected = index);
                            load_selected();
                            cx.notify();
                        }))
                })),
        )
        .when(!has_player, |root| {
            root.child(
                div()
                    .text_sm()
                    .text_color(rgb(subtext))
                    .child("MediaCodec/ExoPlayer frames are imported into the GPUI wgpu surface."),
            )
        })
}
