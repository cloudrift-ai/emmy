//! The bounded memos: what a process remembers across bodies, keyed by a body's text under its content hash.
//! Body equality leaves the axis windows out and a window is identity material, so the hash finds the bucket and the
//! text confirms the hit. Past its capacity a memo forgets its oldest entry.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// A memo of `V` under a key and a body's text, bounded by entry count.
pub struct Memo<K, V> {
    capacity: usize,
    store: HashMap<K, Vec<(String, V)>>,
    order: VecDeque<(K, String)>,
}

impl<K: Hash + Eq + Clone, V: Clone> Memo<K, V> {
    pub fn new(capacity: usize) -> Memo<K, V> {
        Memo {
            capacity,
            store: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn get(&self, key: &K, text: &str) -> Option<V> {
        self.store
            .get(key)?
            .iter()
            .find(|(spelled, _)| spelled == text)
            .map(|(_, value)| value.clone())
    }

    pub fn put(&mut self, key: K, text: String, value: V) {
        self.store
            .entry(key.clone())
            .or_default()
            .push((text.clone(), value));
        self.order.push_back((key, text));
        while self.order.len() > self.capacity {
            let Some((key, oldest)) = self.order.pop_front() else {
                break;
            };
            if let Some(entries) = self.store.get_mut(&key) {
                entries.retain(|(spelled, _)| *spelled != oldest);
                if entries.is_empty() {
                    self.store.remove(&key);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.store.clear();
        self.order.clear();
    }
}
