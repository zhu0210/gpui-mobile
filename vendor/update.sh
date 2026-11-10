#!/bin/sh
# Re-vendors gpui-pre-wgpu and gpui-pre-windows at gpui-pre version $1 from the
# cargo registry (run `cargo fetch` with the new pins first), applies each
# crate's patch and writes the patch again against the new version.
# A hunk that no longer applies stops the script: fix it by hand in
# vendor/<crate>, then run `vendor/update.sh <version> --repatch`.
set -eu
version=$1
cd "$(dirname "$0")"
vendor=$PWD
for crate in gpui-pre-wgpu gpui-pre-windows; do
    source=$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/"$crate-$version" | head -1)
    if [ "${2:-}" != --repatch ]; then
        rm -rf "$crate"
        cp -r "$source" "$crate"
        rm -f "$crate/Cargo.lock" "$crate/.cargo-ok"
        patch -p1 -d "$crate" < "$crate.patch"
    fi
    (cd "$(dirname "$source")" && diff -ruN -x Cargo.lock -x .cargo-ok -x '*.orig' -x '*.rej' "$crate-$version" "$vendor/$crate" || true) |
        sed "s|$vendor/$crate|$crate-patched|g; s|^\(--- [^	]*\)	.*|\1|; s|^\(+++ [^	]*\)	.*|\1|; s|^diff -ruN .* \($crate-$version\)|diff -ruN \1|" > "$crate.patch.new"
    mv "$crate.patch.new" "$crate.patch"
done
