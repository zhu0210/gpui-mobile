package dev.gpui.mobile;

import android.app.Activity;
import android.net.Uri;
import android.util.SparseArray;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import androidx.annotation.OptIn;
import androidx.media3.common.MediaItem;
import androidx.media3.common.MimeTypes;
import androidx.media3.common.Player;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

@OptIn(markerClass = UnstableApi.class)
public class GpuiVideoPlayer {
    private static final SparseArray<ExoPlayer> sPlayers = new SparseArray<>();
    private static final SparseArray<Activity> sActivities = new SparseArray<>();
    private static final SparseArray<SurfaceView> sSurfaces = new SparseArray<>();
    private static int sNextId = 1;

    // Create a new player instance
    public static int create(Activity activity) {
        final int playerId = sNextId++;

        activity.runOnUiThread(() -> {
            ExoPlayer player = new ExoPlayer.Builder(activity).build();
            sPlayers.put(playerId, player);
            sActivities.put(playerId, activity);
        });

        return playerId;
    }

    // Set video source from URL
    public static String setUrl(Activity activity, int playerId, String url) {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("");

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player == null) {
                result.set("0|0|0");
                latch.countDown();
                return;
            }

            // Create MediaItem with HLS support
            MediaItem.Builder builder = new MediaItem.Builder().setUri(Uri.parse(url));
            if (url.endsWith(".m3u8") || url.contains(".m3u8?")) {
                builder.setMimeType(MimeTypes.APPLICATION_M3U8);
            }
            MediaItem mediaItem = builder.build();

            // Set up listener to get metadata after player is ready
            Player.Listener listener = new Player.Listener() {
                @Override
                public void onPlaybackStateChanged(int state) {
                    if (state == Player.STATE_READY) {
                        long duration = player.getDuration();
                        int width = player.getVideoSize().width;
                        int height = player.getVideoSize().height;
                        result.set(duration + "|" + width + "|" + height);

                        // Remove listener after use to prevent memory leaks
                        player.removeListener(this);
                        latch.countDown();
                    } else if (state == Player.STATE_IDLE || state == Player.STATE_ENDED) {
                        // Handle error cases
                        if (latch.getCount() > 0) {
                            result.set("0|0|0");
                            player.removeListener(this);
                            latch.countDown();
                        }
                    }
                }
            };

            player.addListener(listener);
            player.setMediaItem(mediaItem);
            player.prepare();
        });

        try {
            // Wait for metadata with timeout
            if (!latch.await(10, TimeUnit.SECONDS)) {
                return "0|0|0";
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "0|0|0";
        }

        return result.get();
    }

    // Set video source from file path
    public static String setFilePath(Activity activity, int playerId, String filePath) {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("");

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player == null) {
                result.set("0|0|0");
                latch.countDown();
                return;
            }

            MediaItem mediaItem = MediaItem.fromUri(Uri.parse("file://" + filePath));

            // Set up listener to get metadata after player is ready
            Player.Listener listener = new Player.Listener() {
                @Override
                public void onPlaybackStateChanged(int state) {
                    if (state == Player.STATE_READY) {
                        long duration = player.getDuration();
                        int width = player.getVideoSize().width;
                        int height = player.getVideoSize().height;
                        result.set(duration + "|" + width + "|" + height);

                        // Remove listener after use to prevent memory leaks
                        player.removeListener(this);
                        latch.countDown();
                    } else if (state == Player.STATE_IDLE || state == Player.STATE_ENDED) {
                        // Handle error cases
                        if (latch.getCount() > 0) {
                            result.set("0|0|0");
                            player.removeListener(this);
                            latch.countDown();
                        }
                    }
                }
            };

            player.addListener(listener);
            player.setMediaItem(mediaItem);
            player.prepare();
        });

        try {
            // Wait for metadata with timeout
            if (!latch.await(10, TimeUnit.SECONDS)) {
                return "0|0|0";
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "0|0|0";
        }

        return result.get();
    }

    // Play the video
    public static void play(int playerId) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.play();
            }
        });
    }

    // Pause the video
    public static void pause(int playerId) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.pause();
            }
        });
    }

    // Seek to position in milliseconds
    public static void seek(int playerId, long positionMs) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.seekTo(positionMs);
            }
        });
    }

    // Set volume (0.0 to 1.0)
    public static void setVolume(int playerId, float volume) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.setVolume(volume);
            }
        });
    }

    // Set playback speed
    public static void setSpeed(int playerId, float speed) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.setPlaybackSpeed(speed);
            }
        });
    }

    // Set looping
    public static void setLooping(int playerId, boolean looping) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.setRepeatMode(looping ? Player.REPEAT_MODE_ALL : Player.REPEAT_MODE_OFF);
            }
        });
    }

    // Get current position in milliseconds
    public static long getPosition(int playerId) {
        ExoPlayer player = sPlayers.get(playerId);
        if (player == null) return 0;
        return player.getCurrentPosition();
    }

    // Get duration in milliseconds
    public static long getDuration(int playerId) {
        ExoPlayer player = sPlayers.get(playerId);
        if (player == null) return 0;
        long duration = player.getDuration();
        return duration < 0 ? 0 : duration;
    }

    // Get video width
    public static int getWidth(int playerId) {
        ExoPlayer player = sPlayers.get(playerId);
        if (player == null) return 0;
        return player.getVideoSize().width;
    }

    // Get video height
    public static int getHeight(int playerId) {
        ExoPlayer player = sPlayers.get(playerId);
        if (player == null) return 0;
        return player.getVideoSize().height;
    }

    // Check if playing
    public static boolean isPlaying(int playerId) {
        ExoPlayer player = sPlayers.get(playerId);
        if (player == null) return false;
        return player.isPlaying();
    }

    // Show surface at specified position
    public static void showSurface(Activity activity, int playerId, int x, int y, int width, int height) {
        activity.runOnUiThread(() -> {
            SurfaceView surface = sSurfaces.get(playerId);
            if (surface != null) {
                FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(width, height);
                params.leftMargin = x;
                params.topMargin = y;
                surface.setLayoutParams(params);
                surface.setVisibility(View.VISIBLE);
            }
        });
    }

    // Hide surface
    public static void hideSurface(Activity activity, int playerId) {
        activity.runOnUiThread(() -> {
            SurfaceView surface = sSurfaces.get(playerId);
            if (surface != null) {
                surface.setVisibility(View.GONE);
            }
        });
    }

    // Dispose player and clean up resources
    public static void dispose(int playerId) {
        Activity activity = sActivities.get(playerId);
        if (activity == null) return;

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player != null) {
                player.release();
                sPlayers.remove(playerId);
            }

            // Clean up activity reference
            sActivities.remove(playerId);

            // Clean up surface
            SurfaceView surface = sSurfaces.get(playerId);
            if (surface != null) {
                ViewGroup parent = (ViewGroup) surface.getParent();
                if (parent != null) {
                    parent.removeView(surface);
                }
                sSurfaces.remove(playerId);
            }
        });
    }

    // Create and attach video surface
    public static View createVideoSurface(Activity activity, int playerId) {
        final SurfaceView[] result = new SurfaceView[1];
        final CountDownLatch latch = new CountDownLatch(1);

        activity.runOnUiThread(() -> {
            ExoPlayer player = sPlayers.get(playerId);
            if (player == null) {
                latch.countDown();
                return;
            }

            SurfaceView surfaceView = new SurfaceView(activity);
            surfaceView.setVisibility(View.GONE);

            // Attach surface to player
            player.setVideoSurfaceView(surfaceView);

            // Store surface
            sSurfaces.put(playerId, surfaceView);
            result[0] = surfaceView;

            latch.countDown();
        });

        try {
            latch.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return result[0];
    }
}
