#!/usr/bin/env bash
# Cross-builds FFmpeg for iOS arm64 (devices or the simulator) as LGPL dynamic
# frameworks with VideoToolbox, for the `ffmpeg` feature (`gpui_mobile::media`).
# macOS with Xcode only.
#
# LGPL 2.1+: no --enable-gpl, no --enable-nonfree, dynamic linking. Ship the
# frameworks unmodified, with FFmpeg's licence, this script (the configure
# line) and the source tarball or a written offer.
#
# Usage: scripts/build-ffmpeg-ios.sh [device|simulator]
#   FFMPEG_VERSION   release to build (default 9.0.2, as on Android)
# Output: target/ffmpeg-ios/<device|simulator>/
#   Frameworks/lib*.framework   embed these in the app (Frameworks/)
#   lib/lib*.dylib              links to the framework binaries, for the linker
#   include/                    headers, for ffmpeg-sys-next
# Point FFMPEG_DIR at the output when building the Rust crates.
set -euo pipefail

KIND="${1:-device}"
case "$KIND" in
    device) SDK=iphoneos; TRIPLE=arm64-apple-ios16.0; PLATFORM=iPhoneOS ;;
    simulator) SDK=iphonesimulator; TRIPLE=arm64-apple-ios16.0-simulator; PLATFORM=iPhoneSimulator ;;
    *) echo "usage: $0 [device|simulator]" >&2; exit 1 ;;
esac
FFMPEG_VERSION="${FFMPEG_VERSION:-9.0.2}"
# The release tarball's SHA-256, checked against its signature
# (key FCF986EA15E6E293A5644F10B4322F04D67658D8) when it was pinned.
FFMPEG_SHA256="${FFMPEG_SHA256:-8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/target/ffmpeg-ios"
PREFIX="$WORK/$KIND"
# Dependency order: each library links only those before it.
LIBS="avutil swresample swscale avcodec avformat"

mkdir -p "$WORK/download"
cd "$WORK/download"
TARBALL="ffmpeg-$FFMPEG_VERSION.tar.xz"
[ -f "$TARBALL" ] || curl -fLO "https://ffmpeg.org/releases/$TARBALL"
echo "$FFMPEG_SHA256  $TARBALL" | shasum -a 256 -c -

SRC="$WORK/ffmpeg-$FFMPEG_VERSION-$KIND"
rm -rf "$SRC" "$PREFIX"
mkdir -p "$SRC"
tar -xf "$TARBALL" -C "$SRC" --strip-components 1
cd "$SRC"

./configure \
    --prefix="$PREFIX" \
    --target-os=darwin \
    --arch=aarch64 \
    --enable-cross-compile \
    --cc="$(xcrun --sdk "$SDK" --find clang)" \
    --sysroot="$(xcrun --sdk "$SDK" --show-sdk-path)" \
    --extra-cflags="-target $TRIPLE" \
    --extra-ldflags="-target $TRIPLE" \
    --install-name-dir='@rpath' \
    --enable-shared --disable-static --enable-pic \
    --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
    --disable-network \
    --disable-everything \
    --enable-videotoolbox \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,hls,mpegts,aac,mp3 \
    --enable-parser=h264,hevc,vp9,av1,aac,mpegaudio,opus \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,vp9_superframe_split,extract_extradata \
    --enable-decoder=h264,hevc,vp9,av1,aac,mp3,opus \
    --enable-decoder=subrip,ass,ssa,webvtt,mov_text \
    --enable-hwaccel=h264_videotoolbox,hevc_videotoolbox,vp9_videotoolbox,av1_videotoolbox \
    --enable-swresample --enable-swscale

grep -E "^#define CONFIG_GPL 0" config.h
grep -E "^#define CONFIG_NONFREE 0" config.h
grep -E "^#define CONFIG_VIDEOTOOLBOX 1" config.h

make -j"$(sysctl -n hw.ncpu)"
make install

# Wrap each library as a framework (what iOS apps embed and SideStore
# signs), named @rpath/libX.framework/libX, and point lib/ at them.
mkdir -p "$PREFIX/Frameworks"
for lib in $LIBS; do
    name="lib$lib"
    framework="$PREFIX/Frameworks/$name.framework"
    mkdir -p "$framework"
    cp -L "$PREFIX/lib/$name.dylib" "$framework/$name"
    rm -f "$PREFIX/lib/$name".*dylib
    ln -s "../Frameworks/$name.framework/$name" "$PREFIX/lib/$name.dylib"
    install_name_tool -id "@rpath/$name.framework/$name" "$framework/$name"
    for dep in $LIBS; do
        old="$(otool -L "$framework/$name" | awk -v dep="@rpath/lib$dep." 'index($1, dep) == 1 { print $1 }')"
        if [ -n "$old" ]; then
            install_name_tool -change "$old" "@rpath/lib$dep.framework/lib$dep" "$framework/$name"
        fi
    done
    strip -x "$framework/$name"
    cat > "$framework/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>$name</string>
    <key>CFBundleIdentifier</key><string>org.ffmpeg.$name</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$name</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>$FFMPEG_VERSION</string>
    <key>CFBundleVersion</key><string>$FFMPEG_VERSION</string>
    <key>CFBundleSupportedPlatforms</key><array><string>$PLATFORM</string></array>
    <key>MinimumOSVersion</key><string>16.0</string>
</dict>
</plist>
EOF
    # install_name_tool broke the linker's ad-hoc signature; the simulator
    # needs one, and Xcode or SideStore sign again for devices.
    codesign --force --sign - "$framework/$name"
done
rm -rf "$PREFIX/lib/pkgconfig" "$SRC"
otool -L "$PREFIX/Frameworks/libavformat.framework/libavformat"
ls -l "$PREFIX/lib" "$PREFIX/Frameworks"
