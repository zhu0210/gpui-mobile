# Vendored GPUI crates

Two crates of `gpui-pre`, as published, each with a small patch. Applications
take them through `[patch.crates-io]` (see the root `Cargo.toml` and the README).

| Crate | Why | Patch |
|---|---|---|
| `gpui-pre-wgpu` | The headless renderer behind an `offscreen` feature, a constructor that takes the wgpu instance (iOS needs Metal), Android's external-sync Vulkan device, and external textures drawn in place of an image's atlas tile (`set_external_texture`, for `media`) | `gpui-pre-wgpu.patch` |
| `gpui-pre-windows` | The same external-texture hook in the Direct3D 11 atlas, and the renderer's device handed out (`directx_device`), for `media` on Windows | `gpui-pre-windows.patch` |

macOS needs neither: `media` draws its frames with `Window::paint_surface`.

## Moving to a new `gpui-pre`

When upstream gpui-kit and Ely move to a new `gpui-pre` version, bump the pins
in the root `Cargo.toml`, then:

```sh
cargo fetch --target x86_64-pc-windows-msvc --target x86_64-unknown-linux-gnu
vendor/update.sh <version>
```

It copies both crates from the cargo registry, applies the patches and writes
them again against the new version. A hunk that no longer applies stops it:
fix it by hand in `vendor/<crate>`, then `vendor/update.sh <version> --repatch`.

Then check each system: `cargo test --features ffmpeg --lib` on Linux, the
mobile smoke workflow, and the desktop video workflow (macOS and Windows).
