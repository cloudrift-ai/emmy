//! The memos: what a process remembers across bodies, as one store of declared caches. A memo is declared with
//! [`memo!`], which names its capacity in entries, and registers itself in one list, so every memo is inspected
//! (entries, hits, misses) and cleared from one place: [`stats`] and [`clear_all`] here, their exports in Python.
//! The caches are `quick_cache`'s: sharded, locked per shard and never across a computation, bounded by entry
//! count, evicting by its S3-FIFO policy (a probation queue ahead of the main one and a ghost list of what left, so
//! a body met once does not push out one met many times). A key is a body's exact form (`body::Exact`), never its
//! text.

use std::hash::Hash;
use std::sync::Mutex;

use quick_cache::sync::Cache;

use crate::locked;

/// One declared memo: a bounded cache under a name.
pub struct Memo<K, V> {
    name: &'static str,
    cache: Cache<K, V>,
}

impl<K: Eq + Hash + Clone, V: Clone> Memo<K, V> {
    pub fn new(name: &'static str, capacity: usize) -> Memo<K, V> {
        Memo {
            name,
            cache: Cache::new(capacity),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.cache.get(key)
    }

    pub fn insert(&self, key: K, value: V) {
        self.cache.insert(key, value);
    }

    /// The value under `key`, computed by `compute` once when the memo has none; a second asker of one key waits
    /// for the first's result instead of computing it again.
    pub fn get_or_try_insert<E>(
        &self,
        key: &K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        self.cache.get_or_insert_with(key, compute)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn clear(&self) {
        self.cache.clear();
    }

    pub fn stats(&self) -> Stats {
        Stats {
            name: self.name,
            entries: self.cache.len(),
            capacity: Some(self.cache.capacity()),
            hits: self.cache.hits(),
            misses: self.cache.misses(),
        }
    }
}

/// What a memo reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stats {
    pub name: &'static str,
    pub entries: usize,
    /// The bound, when the store has one.
    pub capacity: Option<u64>,
    pub hits: u64,
    pub misses: u64,
}

/// A store the registry lists: a memo, or a table of the extension's that is purged rather than bounded.
pub trait Store: Send + Sync {
    fn stats(&self) -> Stats;
    /// Forget what can be forgotten: every entry of a memo, the dead entries of a table.
    fn clear(&self);
}

impl<K: Eq + Hash + Clone + Send + Sync, V: Clone + Send + Sync> Store for Memo<K, V> {
    fn stats(&self) -> Stats {
        Memo::stats(self)
    }

    fn clear(&self) {
        Memo::clear(self);
    }
}

fn registry() -> &'static Mutex<Vec<&'static dyn Store>> {
    static REGISTRY: Mutex<Vec<&'static dyn Store>> = Mutex::new(Vec::new());
    &REGISTRY
}

/// List `store`: declared memos register themselves on first use.
pub fn register(store: &'static dyn Store) {
    locked(registry()).push(store);
}

/// Every registered store's figures.
pub fn stats() -> Vec<Stats> {
    let stores: Vec<&'static dyn Store> = locked(registry()).clone();
    stores.iter().map(|store| store.stats()).collect()
}

/// Clear every registered store. The registry's lock is not held while a store clears: a table's purge drops Python
/// objects whose finalizers may come back here.
pub fn clear_all() {
    let stores: Vec<&'static dyn Store> = locked(registry()).clone();
    for store in stores {
        store.clear();
    }
}

/// Declare a memo: a function returning the one `Memo` of its name, built on first use with the capacity stated and
/// registered for `stats` and `clear_all`.
///
/// ```ignore
/// memo! {
///     /// The normal forms of the bodies met in this process.
///     pub fn normal_forms() -> Memo<Exact, Body> [capacity = 16_384]
/// }
/// ```
#[macro_export]
macro_rules! memo {
    ($(#[$attr:meta])* $vis:vis fn $name:ident() -> Memo<$K:ty, $V:ty> [capacity = $capacity:expr]) => {
        $(#[$attr])*
        $vis fn $name() -> &'static $crate::memo::Memo<$K, $V> {
            static MEMO: std::sync::OnceLock<$crate::memo::Memo<$K, $V>> = std::sync::OnceLock::new();
            static REGISTERED: std::sync::Once = std::sync::Once::new();
            let memo = MEMO.get_or_init(|| $crate::memo::Memo::new(stringify!($name), $capacity));
            REGISTERED.call_once(|| $crate::memo::register(memo));
            memo
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    memo! {
        /// A memo of this test's own.
        fn squares() -> Memo<u64, u64> [capacity = 8]
    }

    #[test]
    fn a_memo_is_bounded_and_counts_what_it_answers() {
        squares().clear();
        for n in 0..64u64 {
            let value = squares().get_or_try_insert(&n, || Ok::<u64, ()>(n * n));
            assert_eq!(value, Ok(n * n));
        }
        let stats = squares().stats();
        assert_eq!(stats.name, "squares");
        assert!(stats.entries <= 8, "{stats:?}");
        assert_eq!(stats.capacity, Some(8));
        assert!(stats.misses >= 64, "{stats:?}");
        assert_eq!(stats.hits, 0);
        let kept = (0..64u64).filter(|n| squares().get(n).is_some()).count();
        assert_eq!(kept, squares().len());
        assert_eq!(squares().stats().hits, kept as u64);
    }

    #[test]
    fn the_registry_lists_and_clears_every_memo() {
        squares().insert(3, 9);
        assert!(
            stats()
                .iter()
                .any(|s| s.name == "squares" && s.entries >= 1)
        );
        clear_all();
        assert!(squares().is_empty());
        assert!(
            stats()
                .iter()
                .any(|s| s.name == "squares" && s.entries == 0)
        );
    }
}
