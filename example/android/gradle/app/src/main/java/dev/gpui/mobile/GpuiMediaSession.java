package dev.gpui.mobile;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.Intent;
import android.util.Log;

import androidx.annotation.OptIn;
import androidx.media3.common.MediaMetadata;
import androidx.media3.common.Player;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.session.MediaSession;

/**
 * Manages an Android Media3 MediaSession for audio/video playback.
 *
 * Provides:
 * - System media notification with playback controls (handled by Media3)
 * - Lock screen and system panel playback info
 * - Media button handling
 * - Volume key integration
 *
 * All methods are static and called from Rust via JNI.
 */
@OptIn(markerClass = UnstableApi.class)
public class GpuiMediaSession {

    private static final String TAG = "GpuiMediaSession";

    private static MediaSession sSession;
    private static ExoPlayer sDummyPlayer;
    private static Activity sActivity;

    /**
     * Initialize the media session. Call once when playback starts.
     *
     * @param activity The current Activity.
     */
    public static void init(Activity activity) {
        if (sSession != null) return;
        sActivity = activity;

        activity.runOnUiThread(() -> {
            try {
                // Media3 requires a Player instance. We use a dummy ExoPlayer
                // to sync state from Rust to the system media controls.
                sDummyPlayer = new ExoPlayer.Builder(activity).build();

                // Intent to reopen the app when notification is tapped
                Intent openIntent = activity.getPackageManager()
                        .getLaunchIntentForPackage(activity.getPackageName());
                PendingIntent contentIntent = PendingIntent.getActivity(
                        activity, 0, openIntent,
                        PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
                );

                sSession = new MediaSession.Builder(activity, sDummyPlayer)
                        .setSessionActivity(contentIntent)
                        .build();

                sDummyPlayer.addListener(new Player.Listener() {
                    @Override
                    public void onPlayWhenReadyChanged(boolean playWhenReady, int reason) {
                        nativeMediaAction(playWhenReady ? "play" : "pause");
                    }

                    @Override
                    public void onPositionDiscontinuity(Player.PositionInfo oldPosition, Player.PositionInfo newPosition, int reason) {
                        if (reason == Player.DISCONTINUITY_REASON_SEEK) {
                            nativeMediaSeek(newPosition.positionMs);
                        }
                    }
                });

                Log.i(TAG, "Media3 MediaSession initialized");
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize MediaSession", e);
            }
        });
    }

    /**
     * Update the media metadata (title, artist, duration).
     *
     * @param title     Track/video title.
     * @param artist    Artist name (or app name).
     * @param durationMs Duration in milliseconds.
     */
    public static void setMetadata(String title, String artist, long durationMs) {
        if (sActivity == null || sDummyPlayer == null) return;

        sActivity.runOnUiThread(() -> {
            MediaMetadata metadata = new MediaMetadata.Builder()
                    .setTitle(title != null ? title : "Unknown")
                    .setArtist(artist != null ? artist : "GPUI")
                    .build();
            sDummyPlayer.setPlaylistMetadata(metadata);
            Log.i(TAG, "Metadata updated: " + title + " by " + artist);
        });
    }

    /**
     * Update the playback state.
     *
     * @param isPlaying  Whether playback is active.
     * @param positionMs Current playback position in milliseconds.
     * @param speed      Playback speed (1.0 = normal).
     */
    public static void setPlaybackState(boolean isPlaying, long positionMs, float speed) {
        if (sActivity == null || sDummyPlayer == null) return;

        sActivity.runOnUiThread(() -> {
            if (sDummyPlayer.getPlayWhenReady() != isPlaying) {
                sDummyPlayer.setPlayWhenReady(isPlaying);
            }
            // In a more complete implementation, we'd sync positionMs and speed too.
            // For now, we focus on the play/pause state which is most important for notifications.
        });
    }

    /**
     * Release the media session and dismiss the notification.
     */
    public static void release() {
        if (sActivity != null) {
            sActivity.runOnUiThread(() -> {
                if (sSession != null) {
                    sSession.release();
                    sSession = null;
                }
                if (sDummyPlayer != null) {
                    sDummyPlayer.release();
                    sDummyPlayer = null;
                }
                sActivity = null;
                Log.i(TAG, "Media3 MediaSession released");
            });
        }
    }

    /**
     * Get the MediaSession token for volume routing.
     * Returns null if not initialized.
     */
    public static Object getSessionToken() {
        return sSession != null ? sSession.getToken() : null;
    }

    /**
     * JNI callback: notify Rust of a media action from system controls.
     * Actions: "play", "pause", "stop", "next", "previous"
     */
    private static native void nativeMediaAction(String action);

    /**
     * JNI callback: notify Rust of a seek request from system controls.
     */
    private static native void nativeMediaSeek(long positionMs);
}
