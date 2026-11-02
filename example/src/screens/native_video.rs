//! Video decoded by the platform and drawn inside the GPUI scene
//! (`gpui_mobile::media`), with GPUI content layered over the frames.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use gpui::{canvas, div, prelude::*, px, rgb, rgba, AnyElement, App, Corners, MouseDownEvent, Window};
use gpui_mobile::media::{Backend, Track, VideoFrames};

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
    (
        // Two audio tracks and two MP4 text subtitle tracks; the platform's
        // player plays the first audio track only.
        "Tracks & subtitles",
        Clip::Bundled("tracks.mp4", include_bytes!("../../assets/video/tracks.mp4")),
    ),
];

/// The speeds the speed button steps through.
const RATES: &[f32] = &[1.0, 1.5, 2.0, 0.5];
/// The heights the size cap button steps through.
const MAX_HEIGHTS: &[Option<u32>] = &[None, Some(120), Some(90)];

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
    // The index into MAX_HEIGHTS (kept across clips, unlike the rest).
    static MAX_HEIGHT: Cell<usize> = const { Cell::new(0) };
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
    // Smoke tests try the controls with GPUI_VIDEO_CONTROLS=1: 1.5×, the last
    // audio track, the first subtitles, frames capped at 120 px.
    if std::env::var("GPUI_VIDEO_CONTROLS").is_ok_and(|value| value == "1") {
        video.set_rate(1.5);
        if let Some(track) = video.audio_tracks().last() {
            video.set_audio_track(track.id);
        }
        video.set_subtitle_track(video.subtitle_tracks().first().map(|track| track.id));
        MAX_HEIGHT.with(|height| height.set(1));
    }
    video.set_max_height(MAX_HEIGHTS[MAX_HEIGHT.with(Cell::get)]);
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

/// A track as a button shows it: its title and language, or its id.
fn track_name(track: &Track) -> String {
    match (&track.title, &track.language) {
        (Some(title), Some(language)) => format!("{title} ({language})"),
        (Some(name), None) | (None, Some(name)) => name.clone(),
        (None, None) => format!("#{}", track.id),
    }
}

/// The track after `current` in `tracks`, wrapping round.
fn next_track(tracks: &[Track], current: Option<usize>) -> Option<&Track> {
    let index = tracks.iter().position(|track| Some(track.id) == current);
    tracks.get(index.map_or(0, |index| (index + 1) % tracks.len()))
}

fn clock(time: Duration) -> String {
    let seconds = time.as_secs();
    format!("{}:{:02}.{}", seconds / 60, seconds % 60, time.subsec_millis() / 100)
}

pub fn render(_cx: &App) -> AnyElement {
    let clip = CLIP.with(Cell::get);
    let (status, size, paused, controls) = VIDEO.with(|video| {
        let mut video = video.borrow_mut();
        match video.get_or_insert_with(|| open_clip(clip)) {
            Ok(video) => (
                format!(
                    "{} / {} · buffered {} · frames imported: {}{}",
                    clock(video.position()),
                    clock(video.duration()),
                    clock(video.buffered()),
                    video.frames_shown(),
                    if video.ended() { " · ended" } else { "" }
                ),
                video.video_size(),
                video.is_paused(),
                Some((
                    video.rate(),
                    video.is_muted(),
                    video.volume(),
                    video.audio_tracks(),
                    video.audio_track(),
                    video.subtitle_tracks(),
                    video.subtitle_track(),
                    video.subtitle_text(),
                )),
            ),
            Err(error) => (format!("Error: {error}"), Default::default(), true, None),
        }
    });
    let (rate, muted, volume, audio_tracks, audio_track, subtitle_tracks, subtitle_track, subtitle) =
        controls.unwrap_or((1.0, false, 1.0, Vec::new(), None, Vec::new(), None, None));
    let max_height = MAX_HEIGHT.with(Cell::get);
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
                    (true, false) => "Demuxed by FFmpeg. Video through VideoToolbox into BGRA IOSurfaces, the same Metal import as AVPlayer's, or decoded in software with the planes converted on the GPU; audio decoded by FFmpeg into an AudioQueue.",
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
                )
                // The chosen subtitles, drawn by the app.
                .when_some(subtitle, |this, text| {
                    this.child(
                        div().absolute().bottom_3().left_0().right_0().flex().justify_center().child(
                            div()
                                .px_2()
                                .py_1()
                                .rounded_md()
                                .bg(rgba(0x000000cc))
                                .text_sm()
                                .text_color(rgb(0xffffff))
                                .text_center()
                                .children(text.lines().map(|line| div().child(line.to_owned())).collect::<Vec<_>>()),
                        ),
                    )
                }),
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
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap_2()
                .child(button(
                    format!("Speed {rate}×"),
                    BLUE,
                    with_video(|video| {
                        let index = RATES.iter().position(|&rate| rate == video.rate()).unwrap_or(0);
                        video.set_rate(RATES[(index + 1) % RATES.len()]);
                    }),
                ))
                .child(button(
                    if muted { "Unmute" } else { "Mute" },
                    BLUE,
                    with_video(|video| video.set_muted(!video.is_muted())),
                ))
                .child(button(
                    format!("Volume {:.0}%", volume * 100.0),
                    BLUE,
                    with_video(|video| video.set_volume(if video.volume() > 0.3 { video.volume() / 2.0 } else { 1.0 })),
                ))
                .child(button(
                    match MAX_HEIGHTS[max_height] {
                        Some(height) => format!("Max {height}p"),
                        None => "Full size".into(),
                    },
                    BLUE,
                    with_video(|video| {
                        let next = (MAX_HEIGHT.with(Cell::get) + 1) % MAX_HEIGHTS.len();
                        MAX_HEIGHT.with(|height| height.set(next));
                        video.set_max_height(MAX_HEIGHTS[next]);
                    }),
                ))
                .when(audio_tracks.len() > 1, |this| {
                    let name = audio_tracks
                        .iter()
                        .find(|track| Some(track.id) == audio_track)
                        .map_or_else(|| "?".into(), track_name);
                    this.child(button(
                        format!("Audio: {name}"),
                        BLUE,
                        with_video(|video| {
                            if let Some(track) = next_track(&video.audio_tracks(), video.audio_track()) {
                                video.set_audio_track(track.id);
                            }
                        }),
                    ))
                })
                .when(!subtitle_tracks.is_empty(), |this| {
                    let name = subtitle_tracks
                        .iter()
                        .find(|track| Some(track.id) == subtitle_track)
                        .map_or_else(|| "off".into(), track_name);
                    this.child(button(
                        format!("Subtitles: {name}"),
                        BLUE,
                        with_video(|video| {
                            // Off after the last track.
                            let tracks = video.subtitle_tracks();
                            let next = match video.subtitle_track() {
                                Some(id) if tracks.last().is_some_and(|track| track.id == id) => None,
                                current => next_track(&tracks, current).map(|track| track.id),
                            };
                            video.set_subtitle_track(next);
                        }),
                    ))
                }),
        )
        .child(div().text_sm().text_color(rgb(TEXT)).child(status))
        .into_any_element()
}
