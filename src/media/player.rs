//! The clock and the video loop of the threaded players: Android's (both
//! sources) and FFmpeg on iOS.
//!
//! Video and audio run on their own threads and follow one clock ([`State`]):
//! the audio stream's presentation timestamps when there is sound, the
//! monotonic clock otherwise. The video thread shows each frame once the clock
//! reaches its timestamp.

use anyhow::{anyhow, Result};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Video frames later than this are dropped instead of rendered.
pub(super) const LATE_US: i64 = 50_000;
/// How long the playback threads wait before looking at the state again.
pub(super) const POLL: Duration = Duration::from_millis(20);
/// Imported frames per drift report in the log.
const DRIFT_REPORT_FRAMES: u32 = 300;
/// How long a seek waits for the audio stream's first timestamp.
const HOLD_TIMEOUT: Duration = Duration::from_millis(500);
/// How far the clock runs on past the sound last heard: when the sound stops
/// coming (its input stalled), the picture waits for it instead of running
/// ahead and jumping back.
const HEARD_SLACK_NS: i64 = 200_000_000;

#[cfg(not(target_os = "windows"))]
pub(super) fn monotonic_ns() -> i64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes into local storage.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    time.tv_sec * 1_000_000_000 + time.tv_nsec
}

/// Nanoseconds on a monotonic clock, as [`Instant`] counts them.
#[cfg(target_os = "windows")]
pub(super) fn monotonic_ns() -> i64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as i64
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
    /// Whether the anchor is the sound's timestamp, which the clock doesn't
    /// run far past.
    pub heard: bool,
    /// When the clock last started; audio timestamps from before are stale.
    pub started_ns: i64,
    /// Where each track's output ended, once it has, since the last seek.
    pub video_end_us: Option<i64>,
    pub audio_end_us: Option<i64>,
    /// How fast the clock runs: 1 is normal speed.
    pub rate: f64,
    /// The gain applied to the sound, 0 while muted.
    pub volume: f32,
    pub muted: bool,
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
            let mut elapsed = now_ns - self.anchor_ns;
            if self.heard {
                elapsed = elapsed.min(HEARD_SLACK_NS);
            }
            self.anchor_us + (elapsed as f64 / 1000.0 * self.rate) as i64
        } else {
            self.anchor_us
        }
    }

    /// Starts or stops the clock when `should_run` changed: play, pause,
    /// seek, end, or the app going to or coming back from the background.
    pub fn settle(&mut self) {
        let run = self.should_run();
        if run != self.running {
            self.reanchor();
            self.running = run;
            if run {
                self.started_ns = self.anchor_ns;
            }
        }
    }

    /// Anchors the clock where it stands now.
    pub fn reanchor(&mut self) {
        let now = monotonic_ns();
        self.anchor_us = self.position_at(now);
        self.anchor_ns = now;
        self.heard = false;
    }

    pub fn seek(&mut self, target_us: i64) {
        self.seek_generation += 1;
        self.seek_target_us = target_us;
        self.anchor_us = target_us;
        self.anchor_ns = monotonic_ns();
        self.heard = false;
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
            rate: 1.0,
            volume: 1.0,
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

    /// Changes the speed from now on: the clock is re-anchored here first.
    pub fn set_rate(&self, rate: f64) {
        self.update(|state| {
            state.reanchor();
            state.rate = rate;
        });
    }

    pub fn rate(&self) -> f64 {
        self.lock().rate
    }

    pub fn set_volume(&self, volume: f32) {
        self.update(|state| state.volume = volume);
    }

    pub fn volume(&self) -> f32 {
        self.lock().volume
    }

    pub fn set_muted(&self, muted: bool) {
        self.update(|state| state.muted = muted);
    }

    pub fn is_muted(&self) -> bool {
        self.lock().muted
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
        state.heard = true;
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
                Duration::from_micros(((pts - position) as f64 / state.rate) as u64).min(POLL)
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


/// What an audio source produced.
pub(super) enum AudioOutput {
    Pending,
    End,
    /// The output switched to this sample rate and channel count.
    Format((i32, i32)),
    /// Interleaved 16-bit samples, valid until `release`.
    Samples { pts: i64, data: *const i16, frames: usize },
}

pub(super) trait AudioSource {
    /// The sample rate and channel count before any `Format` output.
    fn format(&self) -> Result<(i32, i32)>;
    /// Moves to `target_us`, switching first to the track chosen since (a
    /// `Format` output follows when that changes the format).
    fn seek(&mut self, target_us: i64) -> Result<()>;
    fn next(&mut self) -> Result<AudioOutput>;
    /// Gives back the samples of the last `Samples` output.
    fn release(&mut self) -> Result<()>;
}

/// Where `play_audio` writes: AAudio on Android, an AudioQueue on iOS.
pub(super) trait AudioSink: Sized {
    /// Opens 16-bit interleaved PCM output at `rate` with `channels` channels.
    fn open(rate: i32, channels: i32) -> Result<Self>;
    fn rate(&self) -> i32;
    fn channels(&self) -> i32;
    /// Starts or pauses the output to match the clock.
    fn follow(&self, running: bool);
    /// Pauses and drops what has not played.
    fn pause_and_flush(&self);
    fn frames_written(&self) -> i64;
    /// Writes up to `frames` frames from `data`, waiting at most about
    /// [`POLL`] for room; returns how many it took.
    ///
    /// # Safety
    /// `data` must hold `frames` frames.
    unsafe fn write(&self, data: *const i16, frames: usize) -> Result<usize>;
    /// The frame position last presented, and when, on the monotonic clock.
    fn timestamp(&self) -> Option<(i64, i64)>;
}

/// Decodes audio and writes it to the sink, which paces the thread; the
/// sink's timestamps drive the clock.
pub(super) fn play_audio<S: AudioSink>(source: &mut dyn AudioSource, shared: &Shared) -> Result<()> {
    let mut stream: Option<S> = None;
    let (mut generation, mut target) = (0, 0);
    // When the first samples since the last seek went to the stream.
    let mut first_write: Option<Instant> = None;
    // Frames written before the first sample since the last seek or speed
    // change, its time, and the speed since.
    let mut base: Option<(i64, i64, f64)> = None;
    let mut end_us = 0;
    // Samples resampled for speed or scaled for volume, and where in the
    // input the next output frame falls.
    let mut scratch = Vec::new();
    let mut phase = 0.0;
    loop {
        let state = shared.lock();
        if state.stop {
            return Ok(());
        }
        if state.seek_generation != generation {
            generation = state.seek_generation;
            target = state.seek_target_us;
            drop(state);
            source.seek(target)?;
            if let Some(stream) = &stream {
                stream.pause_and_flush();
            }
            (first_write, base, phase) = (None, None, 0.0);
            continue;
        }
        let (running, holding, finished) = (state.running, state.holding, state.audio_end_us.is_some());
        if let Some(stream) = &stream {
            stream.follow(state.wants_sound());
        }
        if (!running && !holding) || finished {
            drop(shared.wait(state, POLL));
            continue;
        }
        drop(state);

        let (mut pts, data, frames) = match source.next()? {
            AudioOutput::Pending => continue,
            AudioOutput::Format((rate, channels)) => {
                stream = Some(S::open(rate, channels)?);
                continue;
            }
            AudioOutput::End => {
                shared.update(|state| {
                    if state.seek_generation == generation {
                        state.audio_end_us = Some(end_us.max(target));
                        state.holding = false;
                        // A longer picture plays on without sound.
                        state.reanchor();
                    }
                });
                continue;
            }
            AudioOutput::Samples { pts, data, frames } => (pts, data, frames),
        };
        if stream.is_none() {
            let (rate, channels) = source.format()?;
            stream = Some(S::open(rate, channels)?);
        }
        let stream = stream.as_ref().ok_or_else(|| anyhow!("no audio stream"))?;
        let (rate, channels) = (stream.rate() as i64, stream.channels() as usize);
        let mut offset = 0;
        if pts < target {
            offset = (((target - pts) * rate / 1_000_000) as usize).min(frames);
            pts += offset as i64 * 1_000_000 / rate;
        }
        if offset < frames {
            end_us = pts + (frames - offset) as i64 * 1_000_000 / rate;
            let state = shared.lock();
            let (speed, gain) = (state.rate, if state.muted { 0.0 } else { state.volume });
            drop(state);
            // SAFETY: the source keeps `frames` frames at `data` until `release`.
            let input = unsafe { std::slice::from_raw_parts(data.add(offset * channels), (frames - offset) * channels) };
            let output = if speed == 1.0 && gain == 1.0 {
                phase = 0.0;
                input
            } else {
                scratch.clear();
                if speed == 1.0 {
                    scratch.extend_from_slice(input);
                } else {
                    stretch(input, channels, speed, &mut phase, &mut scratch);
                }
                if gain != 1.0 {
                    scratch.iter_mut().for_each(|sample| *sample = (*sample as f32 * gain) as i16);
                }
                &scratch[..]
            };
            let (base_frames, base_us, base_speed) = match base {
                Some(base) if base.2 == speed => base,
                _ => *base.insert((stream.frames_written(), pts, speed)),
            };
            let (data, frames) = (output.as_ptr(), output.len() / channels);
            let mut offset = 0;
            while offset < frames {
                let state = shared.lock();
                if state.stop || state.seek_generation != generation {
                    break;
                }
                let sound = state.wants_sound();
                drop(state);
                stream.follow(sound);
                // Blocks while the stream's buffer is full, which paces this thread.
                // SAFETY: `output` holds `frames` frames.
                let written = unsafe { stream.write(data.add(offset * channels), frames - offset)? };
                offset += written;
                if written > 0 {
                    first_write.get_or_insert_with(Instant::now);
                }
                // Without timestamps the clock starts on its own.
                if first_write.is_some_and(|time| time.elapsed() > HOLD_TIMEOUT) {
                    shared.update(|state| {
                        if state.seek_generation == generation {
                            state.holding = false;
                        }
                    });
                }
                if let Some((frame, time_ns)) = stream.timestamp() {
                    if frame >= base_frames {
                        let played_us = ((frame - base_frames) as f64 * 1e6 / rate as f64 * base_speed) as i64;
                        shared.sync(generation, base_us + played_us, time_ns);
                    }
                }
            }
        }
        source.release()?;
    }
}

/// Resamples interleaved `input` to play `speed` times as fast, starting
/// `phase` frames in and carrying the remainder over to the next call.
// ponytail: plain resampling, so the pitch follows the speed; a WSOLA
// time-stretch keeps it if users mind.
fn stretch(input: &[i16], channels: usize, speed: f64, phase: &mut f64, output: &mut Vec<i16>) {
    let frames = input.len() / channels;
    let mut position = *phase;
    while (position as usize) < frames {
        let (index, fraction) = (position as usize, position.fract());
        let next = (index + 1).min(frames - 1);
        for channel in 0..channels {
            let (a, b) = (input[index * channels + channel] as f64, input[next * channels + channel] as f64);
            output.push((a + (b - a) * fraction) as i16);
        }
        position += speed;
    }
    *phase = position - frames as f64;
}

/// How far imported frames were from the clock, between log reports.
#[derive(Clone, Copy, Default)]
pub(super) struct Drift {
    frames: u32,
    total_us: i64,
    max_us: i64,
}

impl Shared {
    /// Logs how far imported frames trail or lead the clock while playing.
    pub fn record_drift(&self, drift: &std::cell::Cell<Drift>, timestamp_us: i64) {
        let state = self.lock();
        if !state.running {
            return;
        }
        let drift_us = state.position_at(monotonic_ns()) - timestamp_us;
        drop(state);
        let mut current = drift.get();
        current.frames += 1;
        current.total_us += drift_us;
        current.max_us = current.max_us.max(drift_us.abs());
        if current.frames == DRIFT_REPORT_FRAMES {
            log::info!(
                "video: A/V drift over {} frames: mean {:.1} ms, max {:.1} ms",
                current.frames,
                current.total_us as f64 / current.frames as f64 / 1000.0,
                current.max_us as f64 / 1000.0,
            );
            current = Drift::default();
        }
        drift.set(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_keeps_the_count_across_calls() {
        let input: Vec<i16> = (0..200).flat_map(|i| [i, -i]).collect();
        let (mut phase, mut output) = (0.0, Vec::new());
        for chunk in input.chunks(2 * 30) {
            stretch(chunk, 2, 1.5, &mut phase, &mut output);
        }
        assert!((output.len() / 2).abs_diff(134) <= 1, "{} frames", output.len() / 2);
        assert_eq!(&output[..6], &[0, 0, 1, -1, 3, -3]);
        let (mut phase, mut slow) = (0.0, Vec::new());
        stretch(&input, 2, 0.5, &mut phase, &mut slow);
        assert_eq!(slow.len(), 800);
        assert_eq!(&slow[..6], &[0, 0, 0, 0, 1, -1]);
    }

    #[test]
    fn the_clock_waits_for_sound_that_stopped_coming() {
        let shared = Shared::new(true);
        let generation = shared.lock().seek_generation;
        shared.sync(generation, 0, monotonic_ns());
        std::thread::sleep(Duration::from_millis(400));
        let stalled = shared.position_us();
        assert!((HEARD_SLACK_NS / 1000..HEARD_SLACK_NS / 1000 + 20_000).contains(&stalled), "at {stalled} µs");
        // The sound comes again: the clock follows it from there.
        shared.sync(generation, 250_000, monotonic_ns());
        std::thread::sleep(Duration::from_millis(50));
        let resumed = shared.position_us();
        assert!((290_000..330_000).contains(&resumed), "at {resumed} µs");
    }

    #[test]
    fn the_clock_runs_at_the_rate() {
        let shared = Shared::new(false);
        shared.update(|state| state.holding = false);
        shared.set_rate(2.0);
        let start = shared.position_us();
        std::thread::sleep(Duration::from_millis(100));
        let moved = shared.position_us() - start;
        assert!((190_000..260_000).contains(&moved), "moved {moved} µs");
    }
}
