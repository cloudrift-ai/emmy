//! The memos: what a process remembers across bodies. Two caches, the normal forms (`normalize::normal_forms`) and the
//! identities (`identity::identities`), each bounded by the statements its entries hold between them and keyed by a
//! body's exact form (`body::MemoKey`), never its text. The caches are `quick_cache`'s: sharded, locked per shard and
//! never across a computation (a second asker of a key waits for the first's value instead of computing it again),
//! evicting by its one policy (S3-FIFO: a probation queue ahead of the main one and a ghost list of what left, so an
//! entry met once does not push out one met many times). [`clear_all`] forgets both: what a test worker does to start
//! clean.

use std::hash::Hash;

use quick_cache::Weighter;
use quick_cache::sync::Cache;

/// How many statements the body-keyed memos keep between them, each body weighing its statement count: a cut's arms
/// lower the same pieces again and again, and a whole-layer body is thousands of statements, so the bound is on what
/// the memo holds rather than on how many bodies.
pub const BODY_MEMO_WEIGHT: u64 = 1_000_000;

/// The weight of an entry: a function of its key and value, so a memo is bounded by what it holds.
pub struct Weigh<K, V>(pub fn(&K, &V) -> u64);

impl<K, V> Clone for Weigh<K, V> {
    fn clone(&self) -> Self {
        Weigh(self.0)
    }
}

impl<K, V> Weighter<K, V> for Weigh<K, V> {
    fn weight(&self, key: &K, val: &V) -> u64 {
        (self.0)(key, val)
    }
}

/// A bounded cache: at most `capacity` of weight, each entry weighing what `weigh` says of it.
pub struct Memo<K, V> {
    cache: Cache<K, V, Weigh<K, V>>,
}

impl<K: Eq + Hash + Clone, V: Clone> Memo<K, V> {
    pub fn weighted(capacity: u64, weigh: fn(&K, &V) -> u64) -> Memo<K, V> {
        Memo {
            cache: Cache::with_weighter(1024, capacity, Weigh(weigh)),
        }
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

    pub fn clear(&self) {
        self.cache.clear();
    }

    /// The weight the entries hold now.
    pub fn weight(&self) -> u64 {
        self.cache.weight()
    }
}

/// Forget every memoized normal form and identity.
pub fn clear_all() {
    crate::normalize::normal_forms().clear();
    crate::normalize::optimized_forms().clear();
    crate::identity::identities().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memo_is_bounded_and_clears() {
        let squares: Memo<u64, u64> = Memo::weighted(8, |_, _| 1);
        for n in 0..1024u64 {
            assert_eq!(
                squares.get_or_try_insert(&n, || Ok::<u64, ()>(n * n)),
                Ok(n * n)
            );
        }
        // The cache rounds its bound up per shard, but it evicts: far fewer entries stay than went in.
        assert!(
            0 < squares.weight() && squares.weight() < 256,
            "{}",
            squares.weight()
        );
        // A probe that refuses to compute inserts nothing: a hit answers, a miss errors.
        let kept = (0..1024u64)
            .filter(|n| squares.get_or_try_insert(n, || Err::<u64, ()>(())).is_ok())
            .count();
        assert_eq!(kept as u64, squares.weight());
        squares.clear();
        assert_eq!(squares.weight(), 0);
    }
}
