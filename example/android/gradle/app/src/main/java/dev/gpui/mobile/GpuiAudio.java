package dev.gpui.mobile;

import android.app.Activity;
import android.net.Uri;
import android.util.Log;
import android.util.SparseArray;

import androidx.annotation.OptIn;
import androidx.media3.common.MediaItem;
import androidx.media3.common.MimeTypes;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.PlaybackParameters;
import androidx.media3.common.Player;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Audio playback helper for the GPUI audio package using AndroidX Media3 (ExoPlayer).
 *
 * <p>Supports HLS and other streaming formats. All public methods are static
 * and called from Rust via JNI.</p>
 */
@OptIn(markerClass = UnstableApi.class)
public final class GpuiAudio {

    private static final String TAG = "GpuiAudio";
    private static final SparseArray<ExoPlayer> sPlayers = new SparseArray<>();
    private static int sNextId = 1;
    private static final Object sLock = new Object();

    /**
     * Create a new audio player.
     *
     * @return Player ID, or -1 on failure.
     */
    public static int create(Activity activity) {
        final CountDownLatch latch = new CountDownLatch(1);
        final int[] result = new int[]{-1};

        activity.runOnUiThread(() -> {
            try {
                ExoPlayer player = new ExoPlayer.Builder(activity).build();
                synchronized (sLock) {
                    int id = sNextId++;
                    sPlayers.put(id, player);
                    result[0] = id;
                }
            } catch (Exception e) {
                Log.e(TAG, "create failed", e);
            } finally {
                latch.countDown();
            }
        });

        try {
            latch.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {}

        return result[0];
    }

    /**
     * Set the audio source from a URL or file path.
     *
     * @return Duration in milliseconds, or -1 if unknown/error.
     */
    public static long setUrl(Activity activity, int id, String url) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player == null) return -1;

        final ExoPlayer fPlayer = player;
        final CountDownLatch latch = new CountDownLatch(1);

        activity.runOnUiThread(() -> {
            try {
                Uri uri = Uri.parse(url);
                MediaItem.Builder mediaItemBuilder = new MediaItem.Builder().setUri(uri);
                if (url.toLowerCase().contains(".m3u8")) {
                    mediaItemBuilder.setMimeType(MimeTypes.APPLICATION_M3U8);
                }

                fPlayer.addListener(new Player.Listener() {
                    @Override
                    public void onPlaybackStateChanged(int state) {
                        if (state == Player.STATE_READY) {
                            latch.countDown();
                        }
                    }

                    @Override
                    public void onPlayerError(PlaybackException error) {
                        Log.e(TAG, "playback error for url: " + url, error);
                        latch.countDown();
                    }
                });

                fPlayer.setMediaItem(mediaItemBuilder.build());
                fPlayer.prepare();
            } catch (Exception e) {
                Log.e(TAG, "setUrl failed: " + url, e);
                latch.countDown();
            }
        });

        try {
            latch.await(5, TimeUnit.SECONDS);
            long duration = fPlayer.getDuration();
            return duration > 0 ? duration : 0;
        } catch (InterruptedException e) {
            return -1;
        }
    }

    /**
     * Start or resume playback.
     */
    public static void play(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(player::play);
        }
    }

    /**
     * Pause playback.
     */
    public static void pause(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(player::pause);
        }
    }

    /**
     * Stop playback and reset to the beginning.
     */
    public static void stop(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(() -> {
                player.stop();
                player.seekTo(0);
            });
        }
    }

    /**
     * Seek to position in milliseconds.
     */
    public static void seek(int id, long positionMs) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(() -> player.seekTo(positionMs));
        }
    }

    /**
     * Set volume (0.0 to 1.0).
     */
    public static void setVolume(int id, float volume) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            float v = Math.max(0.0f, Math.min(1.0f, volume));
            player.getClock().createHandler(player.getApplicationLooper(), null).post(() -> player.setVolume(v));
        }
    }

    /**
     * Set playback speed.
     */
    public static void setSpeed(int id, float speed) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(() -> {
                PlaybackParameters params = new PlaybackParameters(speed);
                player.setPlaybackParameters(params);
            });
        }
    }

    /**
     * Set looping mode.
     */
    public static void setLooping(int id, boolean looping) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(() -> {
                player.setRepeatMode(looping ? Player.REPEAT_MODE_ALL : Player.REPEAT_MODE_OFF);
            });
        }
    }

    /**
     * Get current playback position in milliseconds.
     */
    public static long getPosition(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            return player.getCurrentPosition();
        }
        return -1;
    }

    /**
     * Get total duration in milliseconds.
     */
    public static long getDuration(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            long dur = player.getDuration();
            return dur > 0 ? dur : 0;
        }
        return -1;
    }

    /**
     * Check if currently playing.
     */
    public static boolean isPlaying(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
        }
        if (player != null) {
            return player.isPlaying();
        }
        return false;
    }

    /**
     * Release the player and free resources.
     */
    public static void dispose(int id) {
        ExoPlayer player;
        synchronized (sLock) {
            player = sPlayers.get(id);
            sPlayers.remove(id);
        }
        if (player != null) {
            player.getClock().createHandler(player.getApplicationLooper(), null).post(player::release);
        }
    }

    private GpuiAudio() {}
}
