//! The memos: what a process remembers across bodies, as one store of declared caches. A memo is declared with
//! [`memo!`], which names its capacity in entries, and registers itself in one list, so every memo is inspected
//! (entries, hits, misses) and cleared from one place: [`stats`] and [`clear_all`] here, their exports in Python. The caches are `quick_cache`'s: sharded, locked per shard and never across a computation, bounded by
//! entry count, evicting by its one policy (S3-FIFO: a probation queue ahead of the main one and a ghost list of
//! what left, so an entry met once does not push out one met many times). A key is a body's exact form
//! (`body::MemoKey`), never its text.
//!
//! A computation that asks its own memo for the key it is computing would wait on itself forever (the cache parks a
//! second asker of a key until the first inserts); [`Memo::get_or_try_insert`] refuses that with an error instead.

use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, Once, OnceLock};

use quick_cache::sync::Cache;

use crate::{Error, locked};

/// How many bodies the body-keyed memos keep: a cut's arms lower the same pieces again and again.
pub const BODY_MEMO_CAPACITY: usize = 16_384;

/// One declared memo: a bounded cache under a name.
pub struct Memo<K, V> {
    name: &'static str,
    cache: Cache<K, V>,
}

/// A key this thread is computing, listed while the computation runs and unlisted when it ends, however it ends.
struct Computing((&'static str, u64));

impl Drop for Computing {
    fn drop(&mut self) {
        COMPUTING.with(|keys| keys.borrow_mut().remove(&self.0));
    }
}

thread_local! {
    /// The keys this thread is computing a value for, by memo and key hash.
    static COMPUTING: RefCell<HashSet<(&'static str, u64)>> = RefCell::new(HashSet::new());
}

impl<K: Eq + Hash + Clone, V: Clone> Memo<K, V> {
    pub fn new(name: &'static str, capacity: usize) -> Memo<K, V> {
        Memo {
            name,
            cache: Cache::new(capacity),
        }
    }

    /// The value under `key`, computed by `compute` once when the memo has none; a second asker of one key waits
    /// for the first's result instead of computing it again. A computation that asks for its own key is an error.
    pub fn get_or_try_insert<E: From<Error>>(
        &self,
        key: &K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let computing = (self.name, hasher.finish());
        if !COMPUTING.with(|keys| keys.borrow_mut().insert(computing)) {
            return Err(Error::Value(format!(
                "the {} memo was asked for a key it is computing",
                self.name
            ))
            .into());
        }
        let _entered = Computing(computing);
        self.cache.get_or_insert_with(key, compute)
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

/// What a store reports.
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

/// The one store under `cell`, built by `init` on first use and listed once: what every declaration goes through.
pub fn declared<S: Store + 'static>(
    cell: &'static OnceLock<S>,
    listed: &'static Once,
    init: impl FnOnce() -> S,
) -> &'static S {
    let store = cell.get_or_init(init);
    listed.call_once(|| locked(registry()).push(store));
    store
}

fn stores() -> Vec<&'static dyn Store> {
    locked(registry()).clone()
}

/// Every registered store's figures.
pub fn stats() -> Vec<Stats> {
    stores().iter().map(|store| store.stats()).collect()
}

/// Clear every registered store. The registry's lock is not held while a store clears: a table's purge drops Python
/// objects whose finalizers may come back here.
pub fn clear_all() {
    for store in stores() {
        store.clear();
    }
}

/// Declare a memo: a function returning the one `Memo` of its name, built on first use with the capacity stated and
/// registered for `stats` and `clear_all`.
///
/// ```ignore
/// memo! {
///     /// The normal forms of the bodies met in this process.
///     pub fn normal_forms() -> Memo<MemoKey, Body> [capacity = BODY_MEMO_CAPACITY]
/// }
/// ```
#[macro_export]
macro_rules! memo {
    ($(#[$attr:meta])* $vis:vis fn $name:ident() -> Memo<$K:ty, $V:ty> [capacity = $capacity:expr]) => {
        $(#[$attr])*
        $vis fn $name() -> &'static $crate::memo::Memo<$K, $V> {
            static MEMO: std::sync::OnceLock<$crate::memo::Memo<$K, $V>> = std::sync::OnceLock::new();
            static LISTED: std::sync::Once = std::sync::Once::new();
            $crate::memo::declared(&MEMO, &LISTED, || {
                $crate::memo::Memo::new(stringify!($name), $capacity)
            })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    memo! {
        /// A memo of the bound test's own.
        fn squares() -> Memo<u64, u64> [capacity = 8]
    }

    memo! {
        /// A memo of the registry test's own.
        fn cubes() -> Memo<u64, u64> [capacity = 8]
    }

    memo! {
        /// A memo of the recursion test's own.
        fn loops() -> Memo<u64, u64> [capacity = 8]
    }

    fn square(n: u64) -> Result<u64, Error> {
        Ok(n * n)
    }

    #[test]
    fn a_memo_is_bounded_and_counts_what_it_answers() {
        for n in 0..64u64 {
            assert_eq!(squares().get_or_try_insert(&n, || square(n)), Ok(n * n));
        }
        let stats = squares().stats();
        assert_eq!((stats.name, stats.capacity), ("squares", Some(8)));
        assert!(stats.entries <= 8, "{stats:?}");
        assert!(stats.misses >= 64, "{stats:?}");
        assert_eq!(stats.hits, 0);
        // A probe that refuses to compute inserts nothing: a hit answers, a miss errors.
        let miss = || Err::<u64, Error>(Error::Value("a miss".into()));
        let kept = (0..64u64)
            .filter(|n| squares().get_or_try_insert(n, miss).is_ok())
            .count();
        assert_eq!(kept, stats.entries);
        assert_eq!(squares().stats().hits, kept as u64);
    }

    #[test]
    fn the_registry_lists_and_clears_every_memo() {
        assert_eq!(
            cubes().get_or_try_insert(&3, || Ok::<u64, Error>(27)),
            Ok(27)
        );
        assert!(stats().iter().any(|s| s.name == "cubes" && s.entries == 1));
        clear_all();
        assert!(stats().iter().any(|s| s.name == "cubes" && s.entries == 0));
    }

    #[test]
    fn a_computation_that_asks_for_its_own_key_is_refused() {
        let nested =
            loops().get_or_try_insert(&1, || loops().get_or_try_insert(&1, || Ok::<u64, Error>(1)));
        assert!(nested.is_err());
        assert_eq!(loops().get_or_try_insert(&1, || Ok::<u64, Error>(1)), Ok(1));
    }
}
