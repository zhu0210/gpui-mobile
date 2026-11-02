//! The clock and the video loop of the threaded players: Android's (both
//! sources) and FFmpeg on iOS.
//!
//! Video and audio run on their own threads and follow one clock ([`State`]):
//! the audio stream's presentation timestamps when there is sound, the
//! monotonic clock otherwise. The video thread shows each frame once the clock
//! reaches its timestamp.

use anyhow::Result;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Video frames later than this are dropped instead of rendered.
pub(super) const LATE_US: i64 = 50_000;
/// How long the playback threads wait before looking at the state again.
pub(super) const POLL: Duration = Duration::from_millis(20);

pub(super) fn monotonic_ns() -> i64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes into local storage.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    time.tv_sec * 1_000_000_000 + time.tv_nsec
}

/// Playback state shared by the UI, video and audio threads.
pub(super) struct Shared {
    pub state: Mutex<State>,
    pub changed: Condvar,
}

#[derive(Default)]
pub(super) struct State {
    pub playing: bool,
    pub looping: bool,
    pub stop: bool,
    pub ended: bool,
    /// Whether the audio thread is playing and owns the clock.
    pub audio: bool,
    /// Set by a seek until its first output; the clock stands at the target.
    pub holding: bool,
    pub seek_generation: u64,
    pub seek_target_us: i64,
    /// The clock: media time `anchor_us` at monotonic time `anchor_ns`,
    /// advancing while `running`.
    pub anchor_us: i64,
    pub anchor_ns: i64,
    pub running: bool,
    /// When the clock last started; audio timestamps from before are stale.
    pub started_ns: i64,
    /// Where each track's output ended, once it has, since the last seek.
    pub video_end_us: Option<i64>,
    pub audio_end_us: Option<i64>,
}

impl State {
    pub fn should_run(&self) -> bool {
        self.wants_sound() && !self.holding
    }

    /// Whether the audio stream should play: also while holding after a seek,
    /// so that its first timestamp can start the clock.
    pub fn wants_sound(&self) -> bool {
        #[cfg(target_os = "android")]
        let visible = crate::android::jni::app_visible();
        // ponytail: iOS plays on in the background; pause it there once
        // the FFmpeg backend has sound.
        #[cfg(not(target_os = "android"))]
        let visible = true;
        self.playing && !self.ended && visible
    }

    pub fn position_at(&self, now_ns: i64) -> i64 {
        if self.running {
            self.anchor_us + (now_ns - self.anchor_ns) / 1000
        } else {
            self.anchor_us
        }
    }

    /// Starts or stops the clock when `should_run` changed: play, pause,
    /// seek, end, or the app going to or coming back from the background.
    pub fn settle(&mut self) {
        let run = self.should_run();
        if run != self.running {
            let now = monotonic_ns();
            self.anchor_us = self.position_at(now);
            self.anchor_ns = now;
            self.running = run;
            if run {
                self.started_ns = now;
            }
        }
    }

    pub fn seek(&mut self, target_us: i64) {
        self.seek_generation += 1;
        self.seek_target_us = target_us;
        self.anchor_us = target_us;
        self.anchor_ns = monotonic_ns();
        self.running = false;
        self.holding = true;
        self.ended = false;
        self.video_end_us = None;
        self.audio_end_us = None;
    }
}

impl Shared {
    pub fn new(audio: bool) -> Arc<Self> {
        let mut state = State {
            playing: true,
            audio,
            ..State::default()
        };
        state.seek(0);
        Arc::new(Self {
            state: Mutex::new(state),
            changed: Condvar::new(),
        })
    }

    pub fn play(&self) {
        self.update(|state| {
            if state.ended {
                state.seek(0);
            }
            state.playing = true;
        });
    }

    pub fn pause(&self) {
        self.update(|state| state.playing = false);
    }

    pub fn is_paused(&self) -> bool {
        !self.lock().playing
    }

    pub fn seek(&self, target_us: i64) {
        self.update(|state| state.seek(target_us));
    }

    pub fn set_looping(&self, looping: bool) {
        self.update(|state| state.looping = looping);
    }

    pub fn position_us(&self) -> i64 {
        self.lock().position_at(monotonic_ns())
    }

    pub fn ended(&self) -> bool {
        self.lock().ended
    }

    /// Whether new frames may still arrive without anything else changing:
    /// while playing, during a seek, or shortly after the last frame shown.
    pub fn is_active(&self, last_shown: Option<Instant>) -> bool {
        let state = self.lock();
        (state.playing && !state.ended)
            || state.holding
            || last_shown.is_some_and(|time| time.elapsed() < Duration::from_millis(500))
    }

    /// The state, with the clock settled.
    pub fn lock(&self) -> MutexGuard<'_, State> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.settle();
        state
    }

    pub fn update(&self, change: impl FnOnce(&mut State)) {
        let mut state = self.lock();
        change(&mut state);
        state.settle();
        drop(state);
        self.changed.notify_all();
    }

    /// Waits for a change or `timeout`, then returns the settled state.
    pub fn wait<'a>(&self, state: MutexGuard<'a, State>, timeout: Duration) -> MutexGuard<'a, State> {
        let (mut state, _) = self
            .changed
            .wait_timeout(state, timeout)
            .unwrap_or_else(PoisonError::into_inner);
        state.settle();
        state
    }

    /// Moves the clock to the audio stream's timestamp: `media_us` was heard
    /// at `time_ns`.
    /// The first one after a seek ends the hold and starts the clock there.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub fn sync(&self, generation: u64, media_us: i64, time_ns: i64) {
        let mut state = self.lock();
        if state.seek_generation != generation {
            return;
        }
        if state.holding {
            state.holding = false;
            state.settle();
        } else if !state.running || time_ns <= state.started_ns {
            return;
        }
        state.anchor_us = media_us;
        state.anchor_ns = time_ns;
        drop(state);
        self.changed.notify_all();
    }

    /// Waits until the clock reaches `pts`; returns how late that is, or
    /// `None` when a seek or stop came first.
    pub fn wait_until(&self, generation: u64, pts: i64) -> Option<i64> {
        let mut state = self.lock();
        loop {
            if state.stop || state.seek_generation != generation {
                return None;
            }
            let position = state.position_at(monotonic_ns());
            if position >= pts {
                return Some(position - pts);
            }
            let wait = if state.running {
                Duration::from_micros((pts - position) as u64).min(POLL)
            } else {
                POLL
            };
            state = self.wait(state, wait);
        }
    }
}

/// What a video source produced.
pub(super) enum VideoOutput<F> {
    /// Nothing yet; try again.
    Pending,
    End,
    /// A frame and its presentation time.
    Frame(i64, F),
}

/// A video decoder the playback loop drives: `AMediaCodec`, or FFmpeg.
pub(super) trait VideoSource {
    type Frame;
    /// Moves to the sync sample at or before `target_us`, dropping what the
    /// decoder holds.
    fn seek(&mut self, target_us: i64) -> Result<()>;
    fn next(&mut self) -> Result<VideoOutput<Self::Frame>>;
    /// Hands the frame back, shown (rendered into the reader, or queued for
    /// upload) when `render`.
    fn release(&mut self, frame: Self::Frame, render: bool) -> Result<()>;
}

/// Decodes video and renders each frame into the reader when the clock
/// reaches it; drops late frames and, after a seek, those before the target.
pub(super) fn play_video<S: VideoSource>(source: &mut S, shared: &Shared) -> Result<()> {
    let (mut generation, mut target, mut first, mut last_us) = (0, 0, false, 0);
    loop {
        let mut state = shared.lock();
        if state.stop {
            return Ok(());
        }
        if state.seek_generation != generation {
            generation = state.seek_generation;
            target = state.seek_target_us;
            drop(state);
            source.seek(target)?;
            first = true;
            continue;
        }
        if let Some(video_end) = state.video_end_us {
            // Ends (or loops) once both tracks ran out and the clock passed them.
            let audio_end = if state.audio { state.audio_end_us } else { Some(video_end) };
            if let Some(audio_end) = audio_end {
                if !state.ended && state.position_at(monotonic_ns()) >= video_end.max(audio_end) {
                    if state.looping {
                        state.seek(0);
                    } else {
                        state.ended = true;
                    }
                    state.settle();
                    drop(state);
                    shared.changed.notify_all();
                    continue;
                }
            }
            drop(shared.wait(state, POLL));
            continue;
        }
        drop(state);

        let (pts, frame) = match source.next()? {
            VideoOutput::Pending => continue,
            VideoOutput::End => {
                shared.update(|state| {
                    if state.seek_generation == generation {
                        state.video_end_us = Some(last_us.max(target));
                        state.holding &= state.audio;
                    }
                });
                continue;
            }
            VideoOutput::Frame(pts, frame) => (pts, frame),
        };
        last_us = pts;
        let render = if pts < target {
            false
        } else if first {
            // The first frame after a seek shows at once, even when paused.
            first = false;
            shared.update(|state| {
                if state.seek_generation == generation && !state.audio {
                    state.holding = false;
                }
            });
            true
        } else {
            shared.wait_until(generation, pts).is_some_and(|late| late <= LATE_US)
        };
        source.release(frame, render)?;
    }
}

