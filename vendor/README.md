# Vendored GPUI crates

`gpui-pre-wgpu`, as published, with a small patch. Applications take it
through `[patch.crates-io]` (see the root `Cargo.toml` and the README).

| Crate | Why | Patch |
|---|---|---|
| `gpui-pre-wgpu` | The headless renderer behind an `offscreen` feature, a constructor that takes the wgpu instance (iOS needs Metal), Android's external-sync Vulkan device, and, for gpui-media's video frames, external textures drawn in place of an image's atlas tile (`set_external_texture`), DMA-BUF import on Linux and `WgpuContext::window_context` | `gpui-pre-wgpu.patch` |

gpui-media uses this copy on every system wgpu draws on (Android, iOS,
Linux), so one patched crate serves both `offscreen` and video; it vendors
`gpui-pre-windows` itself.

## Moving to a new `gpui-pre`

When upstream gpui-kit and Ely move to a new `gpui-pre` version, bump the pins
in the root `Cargo.toml` and `example/Cargo.toml`, then:

```sh
cargo fetch --target x86_64-unknown-linux-gnu
vendor/update.sh <version>
```

It copies the crate from the cargo registry, applies the patch and writes it
again against the new version. A hunk that no longer applies stops it: fix it
by hand in `vendor/gpui-pre-wgpu`, then `vendor/update.sh <version> --repatch`.

Then check each system: `cargo test --lib`, `cargo ndk -t arm64-v8a check
--lib`, the mobile smoke workflow, and gpui-media on the new version (it takes
this crate by `rev`).
