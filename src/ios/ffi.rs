//! FFI (Foreign Function Interface) module for iOS.
//!
//! This module exposes C-compatible functions that can be called from
//! Objective-C code in the iOS app delegate to initialize and control
//! the GPUI application lifecycle.
//!
//! ## Typical call sequence from Obj-C
//!
//! ```text
//! gpui_ios_run_demo()          // sets up platform + invokes finish-launching
//! gpui_ios_get_window()        // retrieve the GPUI window pointer
//! gpui_ios_request_frame(ptr)  // called every CADisplayLink tick
//! ```

use gpui::{App, AppContext, Application, RequestFrameOptions, WindowOptions};
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::OnceLock;

/// Global storage for the GPUI application state.
/// This is set during initialization and used by FFI callbacks.
static IOS_APP_STATE: OnceLock<IosAppState> = OnceLock::new();

/// Holds the state needed for iOS FFI callbacks.
/// Note: On iOS, all UI code runs on the main thread, so we use a RefCell
/// instead of Mutex and don't require Send.
struct IosAppState {
    /// The callback to invoke when the app finishes launching.
    /// This is the closure passed to Application::run().
    /// Using std::cell::UnsafeCell since this is only accessed from the main thread.
    finish_launching: std::cell::UnsafeCell<Option<Box<dyn FnOnce()>>>,
}

// Safety: On iOS, all GPUI operations happen on the main thread.
// The FFI functions are only called from the iOS app delegate which runs on main thread.
// We implement both Send and Sync because OnceLock requires Send for its value type,
// and we need Sync for the static. The actual access is always single-threaded.
unsafe impl Send for IosAppState {}
unsafe impl Sync for IosAppState {}

thread_local! {
    static WINDOWS: std::cell::RefCell<super::window_registry::WindowRegistry<super::window::IosWindow>> =
        std::cell::RefCell::new(Default::default());
}

pub(crate) fn window_for_handle(handle: *mut c_void) -> Option<Rc<super::window::IosWindow>> {
    // Never access UIKit-owned objects from a foreign-thread FFI call.
    objc2::MainThreadMarker::new()?;
    WINDOWS.with(|windows| windows.borrow().get(handle as usize))
}

pub(crate) fn window_ids() -> Vec<usize> {
    if objc2::MainThreadMarker::new().is_none() {
        return Vec::new();
    }
    WINDOWS.with(|windows| windows.borrow().ids())
}

pub(crate) fn unregister_window(handle: usize) {
    WINDOWS.with(|windows| windows.borrow_mut().remove(handle));
}

/// Initialize the GPUI iOS application.
///
/// This should be called from `application:didFinishLaunchingWithOptions:`
/// in the iOS app delegate, before any other GPUI functions.
///
/// Returns a pointer to the app state that should be passed to other FFI functions.
/// Returns null if initialization fails.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_initialize() -> *mut c_void {
    log::info!("GPUI iOS: Initializing");

    // Initialize the app state
    let state = IosAppState {
        finish_launching: std::cell::UnsafeCell::new(None),
    };

    if IOS_APP_STATE.set(state).is_err() {
        log::error!("GPUI iOS: Already initialized");
        return std::ptr::null_mut();
    }

    // Return a non-null pointer to indicate success
    // The actual state is stored in the static
    std::ptr::dangling_mut::<c_void>()
}

/// Register a window with the FFI layer.
///
/// This is called internally when a new IosWindow is created.
/// The opaque window handle can then be retrieved by Objective-C code.
///
/// # Safety
/// This must only be called from the main thread.
pub(crate) fn register_window(window: &Rc<super::window::IosWindow>) -> anyhow::Result<usize> {
    anyhow::ensure!(
        objc2::MainThreadMarker::new().is_some(),
        "iOS window registration requires the main thread"
    );
    WINDOWS
        .with(|windows| windows.borrow_mut().register(window))
        .ok_or_else(|| anyhow::anyhow!("iOS window handle space exhausted"))
}

/// Returns an opaque handle for the most recently registered live window.
/// The value is never dereferenced; stale handles are rejected by every callback.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_get_window() -> *mut c_void {
    if objc2::MainThreadMarker::new().is_none() {
        return std::ptr::null_mut();
    }
    WINDOWS.with(|windows| windows.borrow().latest().unwrap_or(0) as *mut c_void)
}

/// Store the finish launching callback.
///
/// This is called internally by IosPlatform::run() to store the callback
/// that will be invoked when the app finishes launching.
///
/// # Safety
/// This must only be called from the main thread.
pub(crate) fn set_finish_launching_callback(callback: Box<dyn FnOnce()>) {
    if let Some(state) = IOS_APP_STATE.get() {
        // Safety: Only called from main thread
        unsafe {
            *state.finish_launching.get() = Some(callback);
        }
    }
}

/// Called when the iOS app has finished launching.
///
/// This should be called from `application:didFinishLaunchingWithOptions:`
/// in the iOS app delegate, after `gpui_ios_initialize()` returns.
///
/// This invokes the callback passed to Application::run().
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_did_finish_launching(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Did finish launching");

    if let Some(state) = IOS_APP_STATE.get() {
        // Safety: Only called from main thread
        let callback = unsafe { (*state.finish_launching.get()).take() };
        if let Some(callback) = callback {
            log::info!("GPUI iOS: Invoking finish launching callback");
            callback();
        } else {
            log::warn!("GPUI iOS: No finish launching callback registered");
        }
    } else {
        log::error!("GPUI iOS: Not initialized");
    }
}

/// Called when the iOS app will enter the foreground.
///
/// This should be called from `applicationWillEnterForeground:` in the app delegate.
/// This notifies all GPUI windows that the app is becoming active.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_will_enter_foreground(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Will enter foreground");

    // Notify all windows that they're becoming active
    for id in window_ids() {
        if let Some(window) = window_for_handle(id as *mut c_void) {
            window.notify_active_status_change(true);
        }
    }
}

/// Called when the iOS app did become active.
///
/// This should be called from `applicationDidBecomeActive:` in the app delegate.
/// This indicates the app is now in the foreground and receiving events.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_did_become_active(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Did become active");

    // App is now fully active - windows should be notified
    for id in window_ids() {
        if let Some(window) = window_for_handle(id as *mut c_void) {
            window.notify_active_status_change(true);
        }
    }
}

/// Called when the iOS app will resign active.
///
/// This should be called from `applicationWillResignActive:` in the app delegate.
/// This indicates the app is about to become inactive (e.g., incoming call, switching apps).
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_will_resign_active(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Will resign active");

    // App is about to become inactive
    for id in window_ids() {
        if let Some(window) = window_for_handle(id as *mut c_void) {
            window.notify_active_status_change(false);
        }
    }
}

/// Called when the iOS app did enter the background.
///
/// This should be called from `applicationDidEnterBackground:` in the app delegate.
/// At this point, the app should have already saved any user data and released
/// shared resources. The app will be suspended shortly after this returns.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_did_enter_background(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Did enter background");

    // Notify windows they're no longer visible
    for id in window_ids() {
        if let Some(window) = window_for_handle(id as *mut c_void) {
            window.notify_active_status_change(false);
        }
    }
}

/// Called when the iOS app will terminate.
///
/// This should be called from `applicationWillTerminate:` in the app delegate.
/// This is a good place to save any unsaved data.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_will_terminate(_app_ptr: *mut c_void) {
    log::info!("GPUI iOS: Will terminate");

    // Quit callbacks would be invoked here if registered.
}

/// Called when a touch event occurs.
///
/// This bridges UIKit touch events to GPUI's input system.
/// Parameters:
/// - `window_ptr`: Pointer to the IosWindow
/// - `touch_ptr`: Pointer to the UITouch object
/// - `event_ptr`: Pointer to the UIEvent object
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_handle_touch(
    window_ptr: *mut c_void,
    touch_ptr: *mut c_void,
    event_ptr: *mut c_void,
) {
    if window_ptr.is_null() || touch_ptr.is_null() {
        return;
    }

    // Resolve the opaque handle and retain the window throughout the callback
    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };
    window.handle_touch(
        touch_ptr as *mut objc2::runtime::AnyObject,
        event_ptr as *mut objc2::runtime::AnyObject,
    );
}

/// Request a frame to be rendered.
///
/// This should be called from CADisplayLink callback to trigger GPUI rendering.
/// The window_ptr should be the value returned by gpui_ios_get_window().
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_request_frame(window_ptr: *mut c_void) {
    if window_ptr.is_null() {
        return;
    }

    // Resolve and retain the window for this callback, rejecting stale handles.
    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };

    // ── Momentum scrolling ───────────────────────────────────────────────
    // Pump the momentum scroller BEFORE the render callback so that any
    // synthetic ScrollWheel events are processed during this frame's
    // layout/paint cycle.  This produces the smooth, decelerating inertia
    // scroll that users expect on iOS after a fling gesture.
    window.pump_momentum();

    // Only consume pending text when an actual callback can render it. Nested
    // frame requests leave the dirty flag for the outer or next display tick.
    super::callback::invoke(&window.request_frame_callback, |callback| {
        let text_dirty = crate::TEXT_INPUT_DIRTY.swap(false, std::sync::atomic::Ordering::AcqRel);
        callback(RequestFrameOptions {
            force_render: text_dirty,
            ..Default::default()
        });
    });
}

/// Show the software keyboard.
///
/// Call this when a text input field gains focus.
/// The window_ptr should be the value returned by gpui_ios_get_window().
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_show_keyboard(window_ptr: *mut c_void) {
    if window_ptr.is_null() {
        return;
    }

    log::info!("GPUI iOS: Show keyboard requested");

    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };
    window.show_keyboard_with_type(crate::KeyboardType::Default);
}

/// Hide the software keyboard.
///
/// Call this when a text input field loses focus.
/// The window_ptr should be the value returned by gpui_ios_get_window().
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_hide_keyboard(window_ptr: *mut c_void) {
    if window_ptr.is_null() {
        return;
    }

    log::info!("GPUI iOS: Hide keyboard requested");

    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };
    window.hide_keyboard();
}

/// Handle text input from the software keyboard.
///
/// This is called when the user types on the keyboard.
/// Parameters:
/// - `window_ptr`: Pointer to the IosWindow
/// - `text_ptr`: Pointer to NSString with the entered text
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_handle_text_input(window_ptr: *mut c_void, text_ptr: *mut c_void) {
    if window_ptr.is_null() || text_ptr.is_null() {
        return;
    }

    log::info!("GPUI iOS: Handle text input");

    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };
    window.handle_text_input(text_ptr as *mut objc2::runtime::AnyObject);
}

/// Handle a key event from an external keyboard.
///
/// Parameters:
/// - `window_ptr`: Pointer to the IosWindow
/// - `key_code`: The key code from UIKeyboardHIDUsage
/// - `modifiers`: Modifier flags from UIKeyModifierFlags
/// - `is_key_down`: true for key down, false for key up
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_handle_key_event(
    window_ptr: *mut c_void,
    key_code: u32,
    modifiers: u32,
    is_key_down: bool,
) {
    if window_ptr.is_null() {
        return;
    }

    log::info!(
        "GPUI iOS: Handle key event - code: {}, modifiers: {}, down: {}",
        key_code,
        modifiers,
        is_key_down
    );

    let Some(window) = window_for_handle(window_ptr) else {
        return;
    };
    window.handle_key_event(key_code, modifiers, is_key_down);
}

/// Called from ObjC when the app receives a URL to open.
///
/// Parameters:
/// - url_ptr: Pointer to an NSString containing the URL
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_handle_open_url(url_ptr: *mut c_void) {
    if url_ptr.is_null() {
        return;
    }

    let url_string = unsafe {
        use objc2::msg_send;
        let ns_str = url_ptr as *mut objc2::runtime::AnyObject;
        let cstr: *const std::ffi::c_char = msg_send![ns_str, UTF8String];
        if cstr.is_null() {
            return;
        }
        std::ffi::CStr::from_ptr(cstr)
            .to_string_lossy()
            .into_owned()
    };

    log::info!("GPUI iOS: Received deep link: {}", url_string);

    #[cfg(feature = "deeplink")]
    {
        crate::packages::deeplink::ios::handle_open_url(url_string);
    }
}

// ── App callback storage ─────────────────────────────────────────────────────

/// Wrapper around an `UnsafeCell<Option<Box<dyn FnOnce(&mut App)>>>`.
///
/// # Safety
/// On iOS all UI work happens on the main thread.  The FFI entry points
/// (`set_app_callback`, `run_app`, `gpui_ios_run_demo`) are only ever
/// called from the main thread, so interior-mutable access is safe.
#[allow(clippy::type_complexity)]
struct AppCallbackCell(std::cell::UnsafeCell<Option<Box<dyn FnOnce(&mut App)>>>);

// Safety: only accessed from the iOS main thread.
unsafe impl Send for AppCallbackCell {}
unsafe impl Sync for AppCallbackCell {}

static APP_CALLBACK: OnceLock<AppCallbackCell> = OnceLock::new();

/// Register a callback that will be invoked inside `Application::run`.
///
/// This must be called **before** [`run_app`] so that the run-loop
/// has something to do (open a window, create views, etc.).
///
/// # Safety
/// Must be called from the main thread only.
pub fn set_app_callback(cb: Box<dyn FnOnce(&mut App)>) {
    let cell = APP_CALLBACK.get_or_init(|| AppCallbackCell(std::cell::UnsafeCell::new(None)));
    unsafe {
        *cell.0.get() = Some(cb);
    }
}

#[allow(clippy::type_complexity)]
fn take_app_callback() -> Option<Box<dyn FnOnce(&mut App)>> {
    APP_CALLBACK
        .get()
        .and_then(|cell| unsafe { (*cell.0.get()).take() })
}

/// C entry point called from `main.m`'s app delegate.
///
/// Consumer crates should call [`set_app_callback`] **before** this function
/// to register their root view.  If no callback is registered an empty
/// window is opened as a fallback.
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_run_demo() {
    run_app();
}

/// Run the GPUI iOS application.
///
/// This initialises the platform, creates the `Application`, and enters the
/// GPUI run loop.  The actual UI is determined by a callback previously
/// registered via [`set_app_callback`].  If no callback was registered a
/// default empty window is opened so the app doesn't crash.
pub fn run_app() {
    log::info!("GPUI iOS: Starting application");

    // Initialise the FFI layer if not already done.
    if IOS_APP_STATE.get().is_none() {
        let state = IosAppState {
            finish_launching: std::cell::UnsafeCell::new(None),
        };
        let _ = IOS_APP_STATE.set(state);
    }

    let platform = Rc::new(super::IosPlatform::new());
    Application::with_platform(platform).run(|cx: &mut App| {
        if let Some(cb) = take_app_callback() {
            log::info!("GPUI iOS: Invoking user-provided app callback");
            cb(cx);
        } else {
            log::warn!("GPUI iOS: No app callback registered — opening default empty window");
            cx.open_window(
                WindowOptions {
                    window_bounds: None,
                    ..Default::default()
                },
                |_, cx| cx.new(|_| gpui::Empty),
            )
            .expect("Failed to open default window");
            cx.activate(true);
        }
    });

    // On iOS, Application::run() stores the callback and returns immediately.
    // The finish-launching callback is forwarded to set_finish_launching_callback
    // and invoked here synchronously (in a real app the app delegate does this).
    if let Some(state) = IOS_APP_STATE.get() {
        let callback = unsafe { (*state.finish_launching.get()).take() };
        if let Some(callback) = callback {
            log::info!("GPUI iOS: Invoking Application::run callback");
            callback();
        }
    }
}
