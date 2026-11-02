//! Two fingers pinching, surfaced as GPUI [`PinchEvent`]s.
//!
//! GPUI's touch recognizer follows one contact and ignores the others, and it
//! does not recognize pinches yet. So the platform does: a second contact that
//! lands while one is down cancels the first (its pan or tap ends there) and
//! starts a pinch; the two contacts' spacing changes zoom from their midpoint,
//! as a trackpad's pinch does; and the pinch ends when either lifts. The
//! contact left down, and any further ones, reach GPUI not at all: a pinch
//! does not turn into a pan on the way out.
//!
//! Main thread only, like everything in GPUI.

use std::collections::HashMap;

use gpui::{Modifiers, PinchEvent, Pixels, Point, TouchEvent, TouchId, TouchPhase};

/// What one contact turns into.
pub(crate) enum Relayed {
    Touch(TouchEvent),
    Pinch(PinchEvent),
}

pub(crate) struct Pinch {
    /// Each live contact's position, and whether GPUI is following it.
    contacts: HashMap<TouchId, (Point<Pixels>, bool)>,
    /// The two contacts pinching, and how far apart they were last.
    pinching: Option<(TouchId, TouchId, f32)>,
}

impl Pinch {
    pub(crate) fn new() -> Self {
        Self {
            contacts: HashMap::new(),
            pinching: None,
        }
    }

    pub(crate) fn relay(&mut self, event: TouchEvent) -> Vec<Relayed> {
        let id = event.id;
        match event.phase {
            TouchPhase::Started => {
                let followed: Vec<TouchId> = self
                    .contacts
                    .iter()
                    .filter(|(_, (_, followed))| *followed)
                    .map(|(id, _)| *id)
                    .collect();
                match (self.pinching, followed.as_slice()) {
                    (None, []) if self.contacts.is_empty() => {
                        self.contacts.insert(id, (event.position, true));
                        vec![Relayed::Touch(event)]
                    }
                    (None, [first]) => {
                        let first = *first;
                        let at = self.contacts[&first].0;
                        self.contacts.insert(first, (at, false));
                        self.contacts.insert(id, (event.position, false));
                        self.pinching = Some((first, id, distance(at, event.position)));
                        vec![
                            Relayed::Touch(TouchEvent {
                                id: first,
                                phase: TouchPhase::Cancelled,
                                position: at,
                                ..event
                            }),
                            pinch(TouchPhase::Started, at, event.position, 0.),
                        ]
                    }
                    _ => {
                        self.contacts.insert(id, (event.position, false));
                        Vec::new()
                    }
                }
            }
            TouchPhase::Moved => {
                let Some(contact) = self.contacts.get_mut(&id) else {
                    return Vec::new();
                };
                contact.0 = event.position;
                if contact.1 {
                    return vec![Relayed::Touch(event)];
                }
                match self.pinching {
                    Some((a, b, last)) if id == a || id == b => {
                        let (pa, pb) = (self.contacts[&a].0, self.contacts[&b].0);
                        let now = distance(pa, pb);
                        if last <= 0. || now <= 0. {
                            return Vec::new();
                        }
                        self.pinching = Some((a, b, now));
                        vec![pinch(TouchPhase::Moved, pa, pb, now / last - 1.)]
                    }
                    _ => Vec::new(),
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                let Some((at, followed)) = self.contacts.remove(&id) else {
                    return Vec::new();
                };
                if followed {
                    return vec![Relayed::Touch(event)];
                }
                match self.pinching {
                    Some((a, b, _)) if id == a || id == b => {
                        self.pinching = None;
                        let other = if id == a { b } else { a };
                        let other = self.contacts.get(&other).map_or(at, |c| c.0);
                        vec![pinch(TouchPhase::Ended, at, other, 0.)]
                    }
                    _ => Vec::new(),
                }
            }
        }
    }
}

fn distance(a: Point<Pixels>, b: Point<Pixels>) -> f32 {
    (b - a).magnitude() as f32
}

fn pinch(phase: TouchPhase, a: Point<Pixels>, b: Point<Pixels>, delta: f32) -> Relayed {
    Relayed::Pinch(PinchEvent {
        position: (a + b) / 2.,
        delta,
        modifiers: Modifiers::default(),
        phase,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{point, px};

    fn touch(id: u64, phase: TouchPhase, x: f32, y: f32) -> TouchEvent {
        TouchEvent {
            id: TouchId(id),
            phase,
            position: point(px(x), px(y)),
            predicted_position: None,
            force: None,
        }
    }

    /// What reached GPUI: a contact's id and phase, or a pinch as `P` with
    /// its phase and delta.
    fn relay(pinch: &mut Pinch, event: TouchEvent) -> Vec<String> {
        pinch
            .relay(event)
            .into_iter()
            .map(|r| match r {
                Relayed::Touch(t) => format!("{} {:?}", t.id.0, t.phase),
                Relayed::Pinch(p) => format!("P {:?} {:.2}", p.phase, p.delta),
            })
            .collect()
    }

    #[test]
    fn one_finger_passes_through() {
        let mut p = Pinch::new();
        assert_eq!(
            relay(&mut p, touch(1, TouchPhase::Started, 0., 0.)),
            ["1 Started"]
        );
        assert_eq!(
            relay(&mut p, touch(1, TouchPhase::Moved, 0., 50.)),
            ["1 Moved"]
        );
        assert_eq!(
            relay(&mut p, touch(1, TouchPhase::Ended, 0., 50.)),
            ["1 Ended"]
        );
    }

    #[test]
    fn a_second_finger_cancels_the_first_and_pinches() {
        let mut p = Pinch::new();
        relay(&mut p, touch(1, TouchPhase::Started, 100., 100.));
        assert_eq!(
            relay(&mut p, touch(2, TouchPhase::Started, 200., 100.)),
            ["1 Cancelled", "P Started 0.00"]
        );
        // Twice as far apart: zoomed in by 100 %.
        assert_eq!(
            relay(&mut p, touch(2, TouchPhase::Moved, 300., 100.)),
            ["P Moved 1.00"]
        );
        // Back to half: zoomed out by 50 %.
        assert_eq!(
            relay(&mut p, touch(1, TouchPhase::Moved, 200., 100.)),
            ["P Moved -0.50"]
        );
        assert_eq!(
            relay(&mut p, touch(1, TouchPhase::Ended, 200., 100.)),
            ["P Ended 0.00"]
        );
        // The finger left down is no pan.
        assert!(relay(&mut p, touch(2, TouchPhase::Moved, 250., 150.)).is_empty());
        assert!(relay(&mut p, touch(2, TouchPhase::Ended, 250., 150.)).is_empty());
        // And the next touch is an ordinary one again.
        assert_eq!(
            relay(&mut p, touch(3, TouchPhase::Started, 0., 0.)),
            ["3 Started"]
        );
    }

    #[test]
    fn a_third_finger_is_ignored() {
        let mut p = Pinch::new();
        relay(&mut p, touch(1, TouchPhase::Started, 100., 100.));
        relay(&mut p, touch(2, TouchPhase::Started, 200., 100.));
        assert!(relay(&mut p, touch(3, TouchPhase::Started, 0., 0.)).is_empty());
        assert!(relay(&mut p, touch(3, TouchPhase::Moved, 50., 0.)).is_empty());
        assert!(relay(&mut p, touch(3, TouchPhase::Ended, 50., 0.)).is_empty());
        assert_eq!(
            relay(&mut p, touch(2, TouchPhase::Moved, 300., 100.)),
            ["P Moved 1.00"]
        );
    }
}
