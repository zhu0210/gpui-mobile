//! Keep one native-vs-Java owner for an Android touch stream.

#[derive(Default)]
pub(super) struct GestureRoute {
    platform_view: Option<bool>,
}

impl GestureRoute {
    /// True forwards the original event to Java via InputStatus::Unhandled.
    /// Only the initial DOWN may hit-test; secondary pointers share its owner.
    pub(super) fn route(&mut self, action: u32, hit_test: impl FnOnce() -> bool) -> bool {
        match action & 0xff {
            0 => {
                let platform_view = hit_test();
                self.platform_view = Some(platform_view);
                platform_view
            }
            1 | 3 => self.platform_view.take().unwrap_or(true),
            2 | 5 | 6 => self.platform_view.unwrap_or(true),
            _ => true,
        }
    }

    pub(super) fn owns_gpui_gesture(&self) -> bool {
        self.platform_view == Some(false)
    }

    pub(super) fn reset(&mut self) {
        self.platform_view = None;
    }
}

#[cfg(test)]
mod tests {
    use super::GestureRoute;

    #[test]
    fn gpui_drag_cannot_be_stolen_when_it_crosses_a_platform_view() {
        let mut route = GestureRoute::default();
        assert!(!route.route(0, || false));
        assert!(route.owns_gpui_gesture());
        for action in [2, 5 | (1 << 8), 2, 6 | (1 << 8), 1] {
            assert!(!route.route(action, || panic!("continuations must not hit-test")));
        }
        // A new DOWN can choose a different owner.
        assert!(route.route(0, || true));
    }

    #[test]
    fn java_receives_up_even_after_the_finger_leaves_its_view() {
        let mut route = GestureRoute::default();
        assert!(route.route(0, || true));
        for action in [2, 5, 6, 1] {
            assert!(route.route(action, || panic!("Java owns the original stream")));
        }
        assert!(!route.route(0, || false));
    }

    #[test]
    fn cancel_is_delivered_to_the_owner_then_clears_it() {
        for platform_view in [false, true] {
            let mut route = GestureRoute::default();
            route.route(0, || platform_view);
            assert_eq!(route.route(3, || unreachable!()), platform_view);
            // An orphan MOVE never starts a GPUI mouse/scroll gesture.
            assert!(route.route(2, || unreachable!()));
            assert!(!route.route(0, || false));
        }
    }

    #[test]
    fn lifecycle_reset_and_unrelated_motion_do_not_reassign_a_gesture() {
        let mut route = GestureRoute::default();
        route.route(0, || false);
        assert!(route.route(8, || unreachable!())); // wheel/hover is not this touch stream
        assert!(!route.route(2, || unreachable!()));
        route.reset();
        assert!(route.route(1, || unreachable!()));
        assert!(route.route(0, || true));
    }
}
