//! Drag and drop by touch.
//!
//! GPUI starts `on_drag` from a held mouse button, which a finger never
//! sends: taps arrive as clicks and pans as scrolls. [`touch_drags`] claims a
//! long press no element took and, once the finger moves, replays it as a
//! left-button press, moves and release, so every drag source and drop target
//! works by touch as it does with a mouse.
//!
//! A control that a finger should drag from its first move, such as a slider,
//! a knob or a split handle, claims the [`TouchDragEvent`] GPUI offers when the
//! finger starts moving over it (`prevent_default` in its `Started` phase) and
//! lets the event propagate: [`touch_drags`] replays that drag the same way.

use std::cell::Cell;

use gpui::{
    canvas, px, App, IntoElement, LongPressEvent, Modifiers, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, PlatformInput, Point, Styled, TouchDragEvent, TouchPhase,
    Window,
};

/// How far a held finger moves before the press becomes a drag, in logical pixels.
const SLOP: Pixels = px(8.);

#[derive(Clone, Copy)]
enum Press {
    Idle,
    /// Claimed, the finger still where it landed.
    Held(Point<Pixels>),
    /// The button is down for GPUI: moves drag.
    Dragging,
}

thread_local! {
    // ponytail: one press at a time for the whole thread; per-window state if
    // a phone app ever shows two windows.
    static PRESS: Cell<Press> = const { Cell::new(Press::Idle) };
}

/// Turns a long press that no element claims, and a touch drag that one does,
/// into a mouse drag. Add it once, anywhere in the window: it draws nothing.
pub fn touch_drags() -> impl IntoElement {
    canvas(
        |_, _, _| {},
        |_, _, window, _| {
            window.on_mouse_event(|event: &LongPressEvent, phase, window, cx| {
                if !phase.bubble() {
                    return;
                }
                let next = match (event.phase, PRESS.get()) {
                    (TouchPhase::Started, _) if window.default_prevented() => Press::Idle,
                    (TouchPhase::Started, _) => {
                        window.prevent_default();
                        Press::Held(event.start_position)
                    }
                    (TouchPhase::Moved, Press::Held(start))
                        if (event.position - start).magnitude() >= f64::from(SLOP) =>
                    {
                        press(start, event.position, window, cx)
                    }
                    (phase, press) => follow(phase, press, event.position, window, cx),
                };
                PRESS.set(next);
            });
            window.on_mouse_event(|event: &TouchDragEvent, phase, window, cx| {
                if !phase.bubble() {
                    return;
                }
                let next = match (event.phase, PRESS.get()) {
                    (TouchPhase::Started, _) if window.default_prevented() => {
                        press(event.start_position, event.position, window, cx)
                    }
                    (TouchPhase::Started, _) => Press::Idle,
                    (phase, press) => follow(phase, press, event.position, window, cx),
                };
                PRESS.set(next);
            });
        },
    )
    .absolute()
    .size_0()
}

/// Puts the button down where the finger landed and moves it to where it is.
fn press(start: Point<Pixels>, at: Point<Pixels>, window: &Window, cx: &mut App) -> Press {
    replay(down(start), window, cx);
    replay(moved(at), window, cx);
    Press::Dragging
}

/// Carries a press on through a later phase of its gesture.
fn follow(
    phase: TouchPhase,
    press: Press,
    at: Point<Pixels>,
    window: &mut Window,
    cx: &mut App,
) -> Press {
    match (phase, press) {
        (TouchPhase::Moved, Press::Dragging) => {
            replay(moved(at), window, cx);
            Press::Dragging
        }
        (TouchPhase::Started | TouchPhase::Moved, press) => press,
        (TouchPhase::Ended, Press::Dragging) => {
            replay(up(at), window, cx);
            Press::Idle
        }
        (TouchPhase::Cancelled, Press::Dragging) => {
            cx.stop_active_drag(window);
            Press::Idle
        }
        (TouchPhase::Ended | TouchPhase::Cancelled, _) => Press::Idle,
    }
}

/// Dispatches `input` after the current event: GPUI doesn't take events re-entrantly.
fn replay(input: PlatformInput, window: &Window, cx: &mut App) {
    let handle = window.window_handle();
    cx.defer(move |cx| {
        handle
            .update(cx, |_, window, cx| {
                window.dispatch_event(input, cx);
            })
            .ok();
    });
}

fn down(position: Point<Pixels>) -> PlatformInput {
    PlatformInput::MouseDown(MouseDownEvent {
        button: MouseButton::Left,
        position,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    })
}

fn moved(position: Point<Pixels>) -> PlatformInput {
    PlatformInput::MouseMove(MouseMoveEvent {
        position,
        pressed_button: Some(MouseButton::Left),
        modifiers: Modifiers::default(),
    })
}

fn up(position: Point<Pixels>) -> PlatformInput {
    PlatformInput::MouseUp(MouseUpEvent {
        button: MouseButton::Left,
        position,
        modifiers: Modifiers::default(),
        click_count: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        div, point, size, AppContext as _, Context, InteractiveElement, ParentElement, Render,
        StatefulInteractiveElement, TestAppContext, VisualTestContext,
    };
    use std::{cell::RefCell, rc::Rc};

    #[derive(Clone)]
    struct Card(&'static str);

    struct Preview;

    impl Render for Preview {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size(px(10.))
        }
    }

    /// A card at the top left and a bin below it that logs what lands in it.
    struct Board {
        dropped: Rc<RefCell<Vec<&'static str>>>,
        clicked: Rc<Cell<usize>>,
    }

    impl Render for Board {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let dropped = self.dropped.clone();
            let clicked = self.clicked.clone();
            div()
                .size_full()
                .child(touch_drags())
                .child(
                    div()
                        .id("card")
                        .size(px(50.))
                        .on_click(move |_, _, _| clicked.set(clicked.get() + 1))
                        .on_drag(Card("ace"), |_, _, _, cx| cx.new(|_| Preview))
                        .child(claims_touch_drags()),
                )
                .child(
                    div()
                        .id("bin")
                        .size(px(50.))
                        .on_drop(move |card: &Card, _, _| dropped.borrow_mut().push(card.0)),
                )
        }
    }

    /// Claims the touch drags that start over it, as a slider does.
    fn claims_touch_drags() -> impl IntoElement {
        canvas(
            |bounds, window, _| window.insert_hitbox(bounds, gpui::HitboxBehavior::Normal),
            |_, hitbox, window, _| {
                window.on_mouse_event(move |event: &TouchDragEvent, phase, window, _| {
                    if phase.bubble()
                        && event.phase == TouchPhase::Started
                        && hitbox.is_hovered(window)
                    {
                        window.prevent_default();
                    }
                });
            },
        )
        .absolute()
        .size_full()
    }

    fn touch_drag(
        phase: TouchPhase,
        start: Point<Pixels>,
        position: Point<Pixels>,
        cx: &mut VisualTestContext,
    ) {
        cx.simulate_event(TouchDragEvent {
            phase,
            start_position: start,
            position,
        });
        cx.run_until_parked();
    }

    fn press(phase: TouchPhase, position: Point<Pixels>, cx: &mut VisualTestContext) {
        cx.simulate_event(LongPressEvent {
            phase,
            start_position: point(px(25.), px(25.)),
            position,
        });
        cx.run_until_parked();
    }

    fn board(cx: &mut TestAppContext) -> (Board, &mut VisualTestContext) {
        let dropped = Rc::new(RefCell::new(Vec::new()));
        let clicked = Rc::new(Cell::new(0));
        let board = Board {
            dropped: dropped.clone(),
            clicked: clicked.clone(),
        };
        let (_, cx) = cx.add_window_view(move |_, _| Board { dropped, clicked });
        cx.simulate_resize(size(px(100.), px(200.)));
        cx.update(|window, cx| window.draw(cx).clear(cx));
        (board, cx)
    }

    #[gpui::test]
    fn a_long_press_that_moves_drags_and_drops(cx: &mut TestAppContext) {
        let (board, cx) = board(cx);
        press(TouchPhase::Started, point(px(25.), px(25.)), cx);
        press(TouchPhase::Moved, point(px(25.), px(50.)), cx);
        press(TouchPhase::Moved, point(px(25.), px(75.)), cx);
        press(TouchPhase::Ended, point(px(25.), px(75.)), cx);
        assert_eq!(*board.dropped.borrow(), ["ace"]);
        assert_eq!(board.clicked.get(), 0);
    }

    #[gpui::test]
    fn a_long_press_held_still_neither_drags_nor_clicks(cx: &mut TestAppContext) {
        let (board, cx) = board(cx);
        press(TouchPhase::Started, point(px(25.), px(25.)), cx);
        press(TouchPhase::Moved, point(px(27.), px(26.)), cx);
        press(TouchPhase::Ended, point(px(27.), px(26.)), cx);
        assert!(board.dropped.borrow().is_empty());
        assert_eq!(board.clicked.get(), 0);
        assert!(!cx.update(|_, cx| cx.has_active_drag()));
    }

    #[gpui::test]
    fn a_touch_drag_the_card_claims_drags_from_the_first_move(cx: &mut TestAppContext) {
        let (board, cx) = board(cx);
        let start = point(px(25.), px(25.));
        touch_drag(TouchPhase::Started, start, point(px(25.), px(35.)), cx);
        touch_drag(TouchPhase::Moved, start, point(px(25.), px(75.)), cx);
        touch_drag(TouchPhase::Ended, start, point(px(25.), px(75.)), cx);
        assert_eq!(*board.dropped.borrow(), ["ace"]);
    }

    #[gpui::test]
    fn a_touch_drag_no_element_claims_is_left_to_scroll(cx: &mut TestAppContext) {
        let (board, cx) = board(cx);
        let start = point(px(25.), px(75.));
        touch_drag(TouchPhase::Started, start, point(px(25.), px(85.)), cx);
        assert!(!cx.update(|_, cx| cx.has_active_drag()));
        touch_drag(TouchPhase::Ended, start, point(px(25.), px(95.)), cx);
        assert!(board.dropped.borrow().is_empty());
    }
}
