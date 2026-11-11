//! Sound for the desktop FFmpeg players (Linux, macOS, Windows), through cpal:
//! PipeWire, PulseAudio or ALSA; CoreAudio; WASAPI.

use super::player::{monotonic_ns, AudioSink, POLL};
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _};
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// How much audio the ring holds ahead of the device.
const RING_MS: usize = 100;

/// The default output device through cpal, fed from a ring of 16-bit frames
/// that the device callback drains.
pub(super) struct Cpal {
    stream: cpal::Stream,
    rate: i32,
    channels: i32,
    ring: Arc<Ring>,
    playing: Cell<bool>,
}

struct Ring {
    state: Mutex<RingState>,
    drained: Condvar,
    capacity: usize,
}

struct RingState {
    samples: VecDeque<i16>,
    written: i64,
    /// Frames the device has taken.
    played: i64,
    /// A frame position and when it plays (monotonic ns), from the last callback.
    stamp: Option<(i64, i64)>,
}

impl Ring {
    fn lock(&self) -> MutexGuard<'_, RingState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl AudioSink for Cpal {
    fn open(rate: i32, channels: i32) -> Result<Self> {
        // cpal keeps one device enumerator for the process, but COM only for each thread that
        // uses it: when the last of those ends, Windows unloads the audio library from under
        // the enumerator, and the next player to ask for a device crashes. This keeps COM,
        // and the library with it, for the process.
        #[cfg(target_os = "windows")]
        {
            static KEPT: std::sync::Once = std::sync::Once::new();
            KEPT.call_once(|| {
                let _ = unsafe { windows::Win32::System::Com::CoIncrementMTAUsage() };
            });
        }
        let device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| anyhow!("no audio output device"))?;
        let samples_per_frame = channels as usize;
        let ring = Arc::new(Ring {
            state: Mutex::new(RingState {
                samples: VecDeque::new(),
                written: 0,
                played: 0,
                stamp: None,
            }),
            drained: Condvar::new(),
            capacity: rate as usize * RING_MS / 1000 * samples_per_frame,
        });
        let config = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(rate as u32),
            buffer_size: cpal::BufferSize::Default,
        };
        let stream = device.build_output_stream(
            &config,
            {
                let ring = ring.clone();
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let timestamp = info.timestamp();
                    let latency = timestamp.playback.duration_since(&timestamp.callback).unwrap_or_default();
                    let mut state = ring.lock();
                    // The first frame of this buffer plays after the latency.
                    state.stamp = Some((state.played, monotonic_ns() + latency.as_nanos() as i64));
                    let available = state.samples.len().min(data.len());
                    for (out, sample) in data.iter_mut().zip(state.samples.drain(..available)) {
                        *out = f32::from(sample) / 32768.0;
                    }
                    data[available..].fill(0.0);
                    state.played += (available / samples_per_frame) as i64;
                    drop(state);
                    ring.drained.notify_all();
                }
            },
            |error| log::warn!("video: audio output error: {error}"),
            None,
        )?;
        stream.pause()?;
        let name = device.name().unwrap_or_default();
        log::info!("video: audio through cpal on {name:?}, {rate} Hz, {channels} channels");
        Ok(Self {
            stream,
            rate,
            channels,
            ring,
            playing: Cell::new(false),
        })
    }

    fn rate(&self) -> i32 {
        self.rate
    }

    fn channels(&self) -> i32 {
        self.channels
    }

    fn follow(&self, running: bool) {
        if running == self.playing.get() {
            return;
        }
        let result = if running {
            self.stream.play().map_err(|error| error.to_string())
        } else {
            self.stream.pause().map_err(|error| error.to_string())
        };
        if let Err(error) = result {
            log::warn!("video: audio {}: {error}", if running { "play" } else { "pause" });
        }
        if !running {
            self.ring.lock().stamp = None;
        }
        self.playing.set(running);
    }

    fn pause_and_flush(&self) {
        self.follow(false);
        let mut state = self.ring.lock();
        state.samples.clear();
        state.played = state.written;
        state.stamp = None;
    }

    fn frames_written(&self) -> i64 {
        self.ring.lock().written
    }

    unsafe fn write(&self, data: *const i16, frames: usize) -> Result<usize> {
        let channels = self.channels as usize;
        let mut state = self.ring.lock();
        if state.samples.len() + channels > self.ring.capacity {
            state = self
                .ring
                .drained
                .wait_timeout(state, POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let count = ((self.ring.capacity - state.samples.len().min(self.ring.capacity)) / channels).min(frames);
        // SAFETY: the caller vouches for `frames` frames at `data`.
        let samples = unsafe { std::slice::from_raw_parts(data, count * channels) };
        state.samples.extend(samples);
        state.written += count as i64;
        Ok(count)
    }

    fn timestamp(&self) -> Option<(i64, i64)> {
        self.ring.lock().stamp
    }
}
