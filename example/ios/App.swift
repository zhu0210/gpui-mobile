import UIKit

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions options: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let window = UIWindow(frame: UIScreen.main.bounds)
        // The demo menu is dark. Set it on the window so the status bar follows
        // it too; otherwise a device in light mode draws dark text on the dark page.
        window.overrideUserInterfaceStyle = .dark
        // The menu has its own header and back button, so no navigation bar.
        window.rootViewController = GPUIContainerController()
        window.makeKeyAndVisible()
        self.window = window
        return true
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        gpui_ios_did_become_active(nil)
    }

    func applicationWillResignActive(_ application: UIApplication) {
        gpui_ios_will_resign_active(nil)
    }
}

/// UIKit owns this view's frame; GPUI owns only the content rendered inside it.
final class GPUIView: UIView {
    private let gpuiWindow: UnsafeMutableRawPointer
    let contentController: UIViewController

    override init(frame: CGRect) {
        gpui_ios_set_embedded()
        gpui_ios_register_app()
        gpui_ios_run_demo()
        guard let window = gpui_ios_get_window(),
              let controller = gpui_ios_view_controller(window) else {
            fatalError("Could not create the GPUI view")
        }
        gpuiWindow = window
        contentController = Unmanaged<UIViewController>.fromOpaque(controller).takeUnretainedValue()
        super.init(frame: frame)
        clipsToBounds = true
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is unsupported") }

    func attach(to parent: UIViewController) {
        parent.addChild(contentController)
        addSubview(contentController.view)
        contentController.didMove(toParent: parent)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard bounds.width > 0, bounds.height > 0,
              contentController.view.frame != bounds else { return }
        // Publish the new geometry and its rendered content together. Otherwise
        // Core Animation can stretch the previous drawable until the next tick.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        contentController.view.frame = bounds
        gpui_ios_layout_view(gpuiWindow)
        drawFrame()
        CATransaction.commit()
    }

    func drawFrame() { _ = gpui_ios_request_frame(gpuiWindow) }
}

final class GPUIContainerController: UIViewController {
    private var gpuiView: GPUIView!
    private var displayLink: CADisplayLink?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        gpuiView = GPUIView(frame: .zero)
        gpuiView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(gpuiView)
        gpuiView.attach(to: self)

        // Full screen: the menu pads itself by the view's safe-area insets.
        // Without the bottom safe area the keyboard guide ends at the screen
        // edge while the keyboard is hidden (iOS 17; on 16 the menu pads for
        // the home indicator twice).
        if #available(iOS 17.0, *) {
            view.keyboardLayoutGuide.usesBottomSafeArea = false
        }
        NSLayoutConstraint.activate([
            gpuiView.topAnchor.constraint(equalTo: view.topAnchor),
            gpuiView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            gpuiView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            gpuiView.bottomAnchor.constraint(equalTo: view.keyboardLayoutGuide.topAnchor),
        ])

    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        displayLink = CADisplayLink(target: self, selector: #selector(renderFrame))
        displayLink?.add(to: .main, forMode: .common)
    }

    override func viewWillDisappear(_ animated: Bool) {
        displayLink?.invalidate()
        displayLink = nil
        super.viewWillDisappear(animated)
    }

    @objc private func renderFrame() { gpuiView.drawFrame() }

}
