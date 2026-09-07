//! Invoke UIKit-driven callbacks without holding their registration slot borrowed.
use std::cell::RefCell;

pub(super) fn invoke<T, R>(slot: &RefCell<Option<T>>, call: impl FnOnce(&mut T) -> R) -> Option<R> {
    let mut callback = slot.borrow_mut().take()?;
    let result = call(&mut callback);
    let mut slot = slot.borrow_mut();
    if slot.is_none() {
        *slot = Some(callback);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn callback_can_replace_itself_without_borrow_conflict() {
        let slot = RefCell::new(Some(1));
        assert_eq!(
            invoke(&slot, |old| {
                *slot.borrow_mut() = Some(2);
                *old
            }),
            Some(1)
        );
        assert_eq!(*slot.borrow(), Some(2));
    }
    #[test]
    fn nested_dispatch_does_not_invoke_or_consume_pending_work() {
        let slot = RefCell::new(Some(1));
        let mut pending = true;
        invoke(&slot, |_| {
            assert_eq!(
                invoke(&slot, |_| {
                    pending = false;
                }),
                None
            );
        });
        assert!(pending);
        assert_eq!(*slot.borrow(), Some(1));
    }
}
