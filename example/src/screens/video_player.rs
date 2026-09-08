//! Video and controls share GPUI's scene on both mobile platforms.

use super::Router;
use gpui::{div, prelude::*};

#[derive(Default)]
pub(crate) struct VideoState {
    pub fullscreen: bool,
    #[cfg(any(target_os = "android", target_os = "ios"))]
    player: Option<gpui_mobile::packages::video_player::GpuiVideoPlayer>,
    #[cfg(any(target_os = "android", target_os = "ios"))]
    selected: usize,
    #[cfg(any(target_os = "android", target_os = "ios"))]
    retire_pending: bool,

    #[cfg(any(target_os = "android", target_os = "ios"))]
    refresh_until: Option<std::time::Instant>,
}

impl VideoState {
    pub fn dismiss(&mut self) {
        self.fullscreen = false;
        #[cfg(any(target_os = "android", target_os = "ios"))]
        if let Some(player) = self.player.as_mut() {
            player.pause();
            self.retire_pending = true;
        }
    }

    pub fn retire_if_needed(&mut self, _window: &mut gpui::Window) {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        if self.retire_pending {
            if let Some(player) = self.player.as_mut() {
                player.retire_external_frame(_window);
            }
            self.retire_pending = false;
        }
    }
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub fn render(
    _router: &mut Router,
    _window: &mut gpui::Window,
    _cx: &mut gpui::Context<Router>,
) -> impl IntoElement {
    div()
        .size_full()
        .flex()
        .items_center()
        .justify_center()
        .child("Video playback is available in the Android and iOS app.")
}

#[cfg(any(target_os = "android", target_os = "ios"))]
pub use mobile::render;

#[cfg(any(target_os = "android", target_os = "ios"))]
mod mobile {
    use super::*;
    use gpui::{px, rgb, SharedString};
    use gpui_mobile::packages::video_player::{GpuiVideoPlayer, GpuiVideoPlayerConfig};
    use std::time::{Duration, Instant};

    const VIDEOS: &[(&str, &str)] = &[
        ("Big Buck Bunny", "https://lorem.video/bunny_720p"),
        (
            "Big Buck Bunny · HLS",
            "https://lorem.video/hls/bunny/480p/media.m3u8",
        ),
    ];

    impl VideoState {
        fn load(&mut self, index: usize, cx: &gpui::App) {
            let Some((_, url)) = VIDEOS.get(index) else {
                return;
            };
            self.selected = index;
            if let Some(player) = self.player.as_mut() {
                player.open(*url, cx);
            } else {
                self.player = Some(GpuiVideoPlayer::with_config(
                    *url,
                    GpuiVideoPlayerConfig {
                        autoplay: true,
                        show_controls: false,
                        ..Default::default()
                    },
                    cx,
                ));
            }
            self.refresh_until = Some(Instant::now() + Duration::from_secs(4));
        }
    }

    fn time(position: Duration) -> String {
        format!("{}:{:02}", position.as_secs() / 60, position.as_secs() % 60)
    }

    fn control(
        label: impl Into<SharedString>,
        action: impl Fn(&mut VideoState) + 'static,
        cx: &mut gpui::Context<Router>,
    ) -> impl IntoElement {
        div()
            .px_3()
            .py_2()
            .rounded_lg()
            .bg(rgb(0x313244))
            .text_color(rgb(0xcdd6f4))
            .child(label.into())
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(move |router, _, _, cx| {
                    action(&mut router.video_state);
                    // Keep polling briefly after paused seeks so their new preview
                    // reaches the GPUI scene; this never advances the media clock.
                    router.video_state.refresh_until =
                        Some(Instant::now() + Duration::from_secs(4));
                    cx.notify();
                }),
            )
    }

    pub fn render(
        router: &mut Router,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Router>,
    ) -> impl IntoElement {
        let state = &mut router.video_state;
        if state.player.is_none() {
            state.load(state.selected, cx);
        }
        let fullscreen = state.fullscreen;
        let selected = state.selected;
        let Some(player) = state.player.as_mut() else {
            return div().into_any_element();
        };
        player.update(window, cx);
        if !player.is_error()
            && (player.is_playing()
                || player.current_textures().is_none()
                || state
                    .refresh_until
                    .is_some_and(|deadline| Instant::now() < deadline))
        {
            window.request_animation_frame();
        }
        let position = player.position();
        let duration = player.duration();
        let progress = duration.filter(|d| !d.is_zero()).map_or(0.0, |d| {
            (position.as_secs_f32() / d.as_secs_f32()).clamp(0.0, 1.0)
        });
        let playing = player.is_playing();
        let muted = player.is_muted();
        let volume = player.volume();
        let has_subtitles = player.has_subtitles();
        let mut picture = div()
            .relative()
            .w_full()
            .min_h_0()
            .bg(rgb(0))
            .overflow_hidden()
            .when(fullscreen, |d| d.flex_1())
            .when(!fullscreen, |d| d.h(px(220.0)))
            .child(player.surface_element());
        if player.is_error() {
            picture = picture.child(player.error_overlay());
        } else if player.current_textures().is_none() {
            picture = picture.child(player.loading_overlay());
        }
        if let Some(overlay) = player.buffering_overlay() {
            picture = picture.child(overlay);
        }
        if let Some(overlay) = player.subtitle_overlay() {
            picture = picture.child(overlay);
        }

        let mut controls = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .bg(rgb(0x181825))
            .text_color(rgb(0xcdd6f4))
            .child(div().text_sm().child(format!(
                "{} / {}",
                time(position),
                duration.map(time).unwrap_or_else(|| "Live".into())
            )))
            .child(
                div()
                    .h(px(4.0))
                    .w_full()
                    .bg(rgb(0x45475a))
                    .child(div().h_full().w(gpui::relative(progress)).bg(rgb(0x89b4fa))),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .child(control(
                        "−10 s",
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                player.seek(
                                    player.position().saturating_sub(Duration::from_secs(10)),
                                );
                            }
                        },
                        cx,
                    ))
                    .child(control(
                        if playing { "Pause" } else { "Play" },
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                player.toggle_playback();
                            }
                        },
                        cx,
                    ))
                    .child(control(
                        "+10 s",
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                let target =
                                    player.position().saturating_add(Duration::from_secs(10));
                                player.seek(
                                    player
                                        .duration()
                                        .map_or(target, |duration| target.min(duration)),
                                );
                            }
                        },
                        cx,
                    ))
                    .child(control(
                        if fullscreen {
                            "Exit fullscreen"
                        } else {
                            "Fullscreen"
                        },
                        |state| {
                            state.fullscreen = !state.fullscreen;
                        },
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_2()
                    .child(control(
                        if muted { "Unmute" } else { "Mute" },
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                player.toggle_mute();
                            }
                        },
                        cx,
                    ))
                    .child(control(
                        "Volume −",
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                player.set_volume((player.volume() - 0.1).max(0.0));
                            }
                        },
                        cx,
                    ))
                    .child(div().text_sm().child(format!("{:.0}%", volume * 100.0)))
                    .child(control(
                        "Volume +",
                        |state| {
                            if let Some(player) = state.player.as_mut() {
                                player.set_volume((player.volume() + 0.1).min(1.0));
                            }
                        },
                        cx,
                    )),
            );
        if has_subtitles {
            controls = controls.child(control(
                "Subtitles",
                |state| {
                    if let Some(player) = state.player.as_mut() {
                        player.toggle_subtitles();
                    }
                },
                cx,
            ));
        }
        if !fullscreen {
            let mut sources = div().flex().flex_col().gap_2();
            for (index, (title, _)) in VIDEOS.iter().enumerate() {
                sources = sources.child(
                    div()
                        .p_3()
                        .rounded_lg()
                        .bg(rgb(if selected == index {
                            0x45475a
                        } else {
                            0x313244
                        }))
                        .child(*title)
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(move |router, _, window, cx| {
                                if let Some(player) = router.video_state.player.as_mut() {
                                    player.retire_external_frame(window);
                                }
                                router.video_state.load(index, cx);
                                cx.notify();
                            }),
                        ),
                );
            }
            controls = controls.child(sources);
        }
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h_0()
            .bg(rgb(0))
            .child(picture)
            .child(
                div()
                    .id("video-controls")
                    .max_h(px(if fullscreen { 160.0 } else { 420.0 }))
                    .overflow_y_scroll()
                    .child(controls),
            )
            .into_any_element()
    }
}
