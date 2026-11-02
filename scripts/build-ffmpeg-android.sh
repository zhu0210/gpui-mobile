#!/usr/bin/env bash
# Cross-builds FFmpeg for Android arm64-v8a as LGPL shared libraries, for the
# `ffmpeg` feature (`gpui_mobile::media`).
#
# LGPL 2.1+: no --enable-gpl, no --enable-nonfree, dynamic linking. Ship the
# .so files unmodified, with FFmpeg's licence, this script (the configure line)
# and the source tarball or a written offer.
#
# Usage: scripts/build-ffmpeg-android.sh
#   ANDROID_NDK_HOME  NDK root (default ~/Android/Sdk/ndk/29.0.13846066)
#   FFMPEG_VERSION    release to build (default 9.0.2, the desktop version)
# Output: target/ffmpeg-android/arm64-v8a/{include,lib}. Point FFMPEG_DIR at it
# when building the Rust crates; example/build.sh copies lib/*.so into the APK.
set -euo pipefail

FFMPEG_VERSION="${FFMPEG_VERSION:-9.0.2}"
NDK="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/29.0.13846066}"
API=26
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/target/ffmpeg-android"
PREFIX="$WORK/arm64-v8a"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"

mkdir -p "$WORK/download"
cd "$WORK/download"
TARBALL="ffmpeg-$FFMPEG_VERSION.tar.xz"
if [ ! -f "$TARBALL" ]; then
    curl -fLO "https://ffmpeg.org/releases/$TARBALL"
    curl -fLO "https://ffmpeg.org/releases/$TARBALL.asc"
fi
# Needs FFmpeg's release key in the keyring (gpg --recv-keys FCF986EA15E6E293A5644F10B4322F04D67658D8).
gpg --verify "$TARBALL.asc" "$TARBALL"

SRC="$WORK/ffmpeg-$FFMPEG_VERSION"
rm -rf "$SRC" "$PREFIX"
tar -xf "$TARBALL" -C "$WORK"
cd "$SRC"

./configure \
    --prefix="$PREFIX" \
    --target-os=android \
    --arch=aarch64 \
    --cpu=armv8-a \
    --enable-cross-compile \
    --sysroot="$TOOLCHAIN/sysroot" \
    --cc="$TOOLCHAIN/bin/aarch64-linux-android$API-clang" \
    --cxx="$TOOLCHAIN/bin/aarch64-linux-android$API-clang++" \
    --ar="$TOOLCHAIN/bin/llvm-ar" \
    --nm="$TOOLCHAIN/bin/llvm-nm" \
    --ranlib="$TOOLCHAIN/bin/llvm-ranlib" \
    --strip="$TOOLCHAIN/bin/llvm-strip" \
    --extra-ldflags="-Wl,-z,max-page-size=16384" \
    --enable-shared --disable-static --enable-pic \
    --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
    --disable-network --disable-vulkan \
    --disable-everything \
    --enable-jni --enable-mediacodec \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,hls,mpegts,aac,mp3 \
    --enable-parser=h264,hevc,vp9,av1,aac,mpegaudio,opus \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,vp9_superframe_split,extract_extradata \
    --enable-decoder=h264_mediacodec,hevc_mediacodec,vp9_mediacodec,av1_mediacodec \
    --enable-decoder=h264,hevc,vp9,aac,mp3,opus \
    --enable-decoder=subrip,ass,ssa,webvtt,movtext \
    --enable-swresample --enable-swscale

grep -E "^#define CONFIG_GPL 0" config.h
grep -E "^#define CONFIG_NONFREE 0" config.h

make -j"$(nproc)"
make install
"$TOOLCHAIN/bin/llvm-strip" --strip-unneeded "$PREFIX"/lib/*.so
rm -rf "$SRC"
ls -l "$PREFIX/lib"/*.so
