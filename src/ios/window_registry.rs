//! Non-owning registry with opaque, non-reused handles for native callbacks.
use std::{
    collections::BTreeMap,
    rc::{Rc, Weak},
};

pub(super) struct WindowRegistry<T> {
    next: usize,
    windows: BTreeMap<usize, Weak<T>>,
}

impl<T> Default for WindowRegistry<T> {
    fn default() -> Self {
        Self {
            next: 0,
            windows: BTreeMap::new(),
        }
    }
}

impl<T> WindowRegistry<T> {
    pub fn register(&mut self, window: &Rc<T>) -> Option<usize> {
        self.next = self.next.checked_add(1)?;
        self.windows.insert(self.next, Rc::downgrade(window));
        Some(self.next)
    }
    pub fn remove(&mut self, id: usize) {
        self.windows.remove(&id);
    }
    pub fn get(&self, id: usize) -> Option<Rc<T>> {
        self.windows.get(&id)?.upgrade()
    }
    pub fn latest(&self) -> Option<usize> {
        self.windows
            .iter()
            .rev()
            .find_map(|(id, window)| (window.strong_count() > 0).then_some(*id))
    }
    pub fn ids(&self) -> Vec<usize> {
        self.windows.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removed_handle_never_resolves_to_a_replacement_window() {
        let mut registry = WindowRegistry::default();
        let first = Rc::new(1);
        let id = registry.register(&first).unwrap();
        registry.remove(id);
        let replacement = Rc::new(2);
        let next = registry.register(&replacement).unwrap();
        assert_ne!(id, next);
        assert!(registry.get(id).is_none());
        assert_eq!(registry.latest(), Some(next));
        assert_eq!(registry.ids(), vec![next]);
    }
    #[test]
    fn callback_guard_survives_owner_close_without_retaining_window_forever() {
        let mut registry = WindowRegistry::default();
        let owner = Rc::new(1);
        let weak = Rc::downgrade(&owner);
        let id = registry.register(&owner).unwrap();
        let callback_guard = registry.get(id).unwrap();
        registry.remove(id);
        drop(owner);
        assert_eq!(*callback_guard, 1);
        assert!(registry.get(id).is_none());
        drop(callback_guard);
        assert!(weak.upgrade().is_none());
    }
}
