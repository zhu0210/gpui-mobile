//! In-app WebView for loading web content.
//!
//! Provides a cross-platform WebView API backed by:
//! - Android: `android.webkit.WebView` via JNI
//! - iOS: `WKWebView` via Objective-C
//!
//! Uses the platform view system for native view embedding.
//!
//! Feature-gated behind `webview`.

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "ios")]
mod ios;

use crate::platform_view::{
    PlatformViewBounds, PlatformViewHandle, PlatformViewParams, PlatformViewRegistry,
};
use std::sync::Arc;

/// Configuration for creating a WebView.
#[derive(Debug, Clone)]
pub struct WebViewSettings {
    /// Enable JavaScript execution (default: true).
    pub javascript_enabled: bool,
    /// User-agent string override (None = platform default).
    pub user_agent: Option<String>,
    /// Allow zoom gestures (default: true).
    pub zoom_enabled: bool,
    /// Enable DOM storage / localStorage (default: true).
    pub dom_storage_enabled: bool,
    /// Top offset in logical points — the WebView starts this many points
    /// below the top of the screen, leaving room for a GPUI-rendered app bar.
    /// Default: 0.0 (fullscreen).
    pub top_offset: f32,
    /// Left offset in logical points, as `top_offset`: room for a sidebar beside it.
    pub left_offset: f32,
    /// Navigations to URLs starting with this are stopped and handed to the app
    /// through [`WebViewHandle::take_intercepted`], such as a sign-in page's
    /// redirect back to the app. Default: none.
    pub intercept: Option<String>,
}

impl Default for WebViewSettings {
    fn default() -> Self {
        Self {
            javascript_enabled: true,
            user_agent: None,
            zoom_enabled: true,
            dom_storage_enabled: true,
            top_offset: 0.0,
            left_offset: 0.0,
            intercept: None,
        }
    }
}

/// Register the "webview" platform view factory.
fn ensure_factory_registered() {
    let registry = PlatformViewRegistry::global();
    if !registry.has_factory("webview") {
        #[cfg(target_os = "android")]
        {
            use crate::android::platform_view::AndroidPlatformViewFactory;
            registry.register(
                "webview",
                Box::new(AndroidPlatformViewFactory::new("webview")),
            );
        }
        #[cfg(target_os = "ios")]
        {
            use crate::ios::platform_view::IosPlatformViewFactory;
            registry.register("webview", Box::new(IosPlatformViewFactory::new("webview")));
        }
    }
}

fn settings_to_creation_params(
    settings: &WebViewSettings,
) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();
    params.insert(
        "javascript_enabled".to_string(),
        settings.javascript_enabled.to_string(),
    );
    params.insert(
        "dom_storage_enabled".to_string(),
        settings.dom_storage_enabled.to_string(),
    );
    params.insert(
        "zoom_enabled".to_string(),
        settings.zoom_enabled.to_string(),
    );
    if let Some(ref ua) = settings.user_agent {
        params.insert("user_agent".to_string(), ua.clone());
    }
    if let Some(ref prefix) = settings.intercept {
        params.insert("intercept".to_string(), prefix.clone());
    }
    params
}

/// Load a URL in a platform-native WebView.
///
/// Creates a platform view of type "webview" and loads the URL.
/// Returns a handle for controlling and dismissing the view.
pub fn load_url(url: &str, settings: &WebViewSettings) -> Result<WebViewHandle, String> {
    ensure_factory_registered();

    let mut params_map = settings_to_creation_params(settings);
    // No element places it: the platform shows it over the window below `top_offset`, right of `left_offset`.
    params_map.insert("overlay".to_string(), "1".to_string());
    params_map.insert("url".to_string(), url.to_string());

    let params = PlatformViewParams {
        bounds: PlatformViewBounds {
            x: settings.left_offset,
            y: settings.top_offset,
            width: 0.0, // Will be set by platform_view_element
            height: 0.0,
        },
        creation_params: params_map,
    };

    let handle = PlatformViewRegistry::global().create_view("webview", params)?;
    Ok(WebViewHandle {
        platform_handle: Some(Arc::new(handle)),
    })
}

/// Load raw HTML content in a WebView.
pub fn load_html(html: &str, settings: &WebViewSettings) -> Result<WebViewHandle, String> {
    ensure_factory_registered();

    let mut params_map = settings_to_creation_params(settings);
    // No element places it: the platform shows it over the window below `top_offset`, right of `left_offset`.
    params_map.insert("overlay".to_string(), "1".to_string());
    params_map.insert("html".to_string(), html.to_string());

    let params = PlatformViewParams {
        bounds: PlatformViewBounds {
            x: settings.left_offset,
            y: settings.top_offset,
            width: 0.0,
            height: 0.0,
        },
        creation_params: params_map,
    };

    let handle = PlatformViewRegistry::global().create_view("webview", params)?;
    Ok(WebViewHandle {
        platform_handle: Some(Arc::new(handle)),
    })
}

/// Evaluate JavaScript in an existing WebView.
pub fn evaluate_javascript(handle: &WebViewHandle, script: &str) -> Result<(), String> {
    if handle.platform_handle.is_none() {
        return Err("No active WebView".into());
    }
    #[cfg(target_os = "ios")]
    {
        ios::evaluate_javascript(handle, script)
    }
    #[cfg(target_os = "android")]
    {
        android::evaluate_javascript(handle, script)
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = (handle, script);
        Err("webview is only available on iOS and Android".into())
    }
}

/// Navigate the WebView back one page in its history.
pub fn go_back(handle: &WebViewHandle) -> Result<(), String> {
    if handle.platform_handle.is_none() {
        return Err("No active WebView".into());
    }
    #[cfg(target_os = "ios")]
    {
        ios::go_back(handle)
    }
    #[cfg(target_os = "android")]
    {
        android::go_back(handle)
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = handle;
        Err("webview is only available on iOS and Android".into())
    }
}

/// Reload the current page in the WebView.
pub fn reload(handle: &WebViewHandle) -> Result<(), String> {
    if handle.platform_handle.is_none() {
        return Err("No active WebView".into());
    }
    #[cfg(target_os = "ios")]
    {
        ios::reload(handle)
    }
    #[cfg(target_os = "android")]
    {
        android::reload(handle)
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = handle;
        Err("webview is only available on iOS and Android".into())
    }
}

/// Stop loading the current page in the WebView.
pub fn stop_loading(handle: &WebViewHandle) -> Result<(), String> {
    if handle.platform_handle.is_none() {
        return Err("No active WebView".into());
    }
    #[cfg(target_os = "ios")]
    {
        ios::stop_loading(handle)
    }
    #[cfg(target_os = "android")]
    {
        android::stop_loading(handle)
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = handle;
        Err("webview is only available on iOS and Android".into())
    }
}

/// Dismiss / destroy a WebView.
pub fn dismiss(mut handle: WebViewHandle) -> Result<(), String> {
    if let Some(h) = handle.platform_handle.take() {
        h.dispose();
    }
    Ok(())
}

/// Opaque handle to a native WebView instance.
///
/// Wraps a platform view handle for lifecycle management.
#[derive(Debug)]
pub struct WebViewHandle {
    /// Platform view handle managing the native WebView.
    pub platform_handle: Option<Arc<PlatformViewHandle>>,
}

impl WebViewHandle {
    /// Get the platform view handle for embedding in a GPUI element.
    pub fn platform_view_handle(&self) -> Option<Arc<PlatformViewHandle>> {
        self.platform_handle.clone()
    }

    /// The last navigation stopped by [`WebViewSettings::intercept`], once.
    // ponytail: one slot for the whole app, so one intercepting WebView at a time;
    // key it by view id if two ever need to intercept together.
    pub fn take_intercepted(&self) -> Option<String> {
        #[cfg(target_os = "android")]
        return android::take_intercepted();
        #[cfg(target_os = "ios")]
        return crate::ios::platform_view::take_intercepted();
        #[allow(unreachable_code)]
        None
    }
}

impl Drop for WebViewHandle {
    fn drop(&mut self) {
        if let Some(h) = self.platform_handle.take() {
            h.dispose();
        }
    }
}
