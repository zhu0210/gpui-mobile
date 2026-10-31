#!/usr/bin/env bash
# Installs the example APK on the running emulator, launches it, and saves a
# screenshot and logcat to smoke/. Fails if the app is not running afterwards.
set -uo pipefail

APK=example/android/gradle/app/build/outputs/apk/debug/app-debug.apk
OUT=smoke
mkdir -p "$OUT"

adb install -r "$APK"
adb logcat -c
adb shell am start -W -n dev.gpui.mobile.example/dev.gpui.mobile.GpuiActivity
sleep 20

adb exec-out screencap -p > "$OUT/screenshot.png"
adb logcat -d > "$OUT/logcat.txt"
adb shell dumpsys gfxinfo dev.gpui.mobile.example > "$OUT/gfxinfo.txt" || true

if ! adb shell pidof dev.gpui.mobile.example; then
  echo "::error::the example app is not running after launch"
  grep -E "signal [0-9]+|panicked|#0[0-9] pc" "$OUT/logcat.txt" | head -20
  exit 1
fi
