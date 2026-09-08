// Settings for the GPUI Mobile Android Example project.
//
// This Gradle project packages the Rust
// native library (compiled via cargo-ndk) into an APK using NativeActivity.

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "GPUIMobileExample"
include(":app")

// CI checks out the pinned Lumina source here. For a local checkout, pass
// -PluminaVideoDir=/path/to/lumina-video (the repository root).
val luminaVideoDir = providers.gradleProperty("luminaVideoDir")
    .orElse("../../../.dependencies/lumina-video")
include(":lumina-video-bridge")
project(":lumina-video-bridge").projectDir =
    file(luminaVideoDir.get()).resolve("android/lumina-video-bridge")
