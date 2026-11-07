//! A body's identity: the digest of its value-numbered scope tree, with every external buffer keyed by its type and
//! its use instead of its spelling, and the roles its buffers fill.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::body::Body;
use crate::elementwise::Op;
use crate::kinds::{Assign, Statement};
use crate::normalize::normalize_body;
use crate::values::{digest, hex};
use crate::{Name, Result};

/// What an identity was computed under: the clustered reading or the exact one, and each buffer's color.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IdentityKey {
    pub cluster: bool,
    pub colors: Option<Vec<(Name, String)>>,
}

/// A body's identity material: its key and the external buffers in canonical rank order (`arguments[i]` fills role
/// `i`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Identity {
    pub key: String,
    pub arguments: Vec<Name>,
}

/// Replace each statement's operation with its compute-unit cluster representative. The result is digest material
/// and must not be executed.
fn canonicalize_op_clusters(body: &Body) -> Body {
    body.map_stmts(&mut |stmt| {
        Some(match &**stmt {
            // A sum is FMA-class work; which coefficients it carries is not the cluster's business.
            Statement::Affine(a) => Arc::new(Statement::Assign(Assign {
                name: a.name.clone(),
                op: Op::new("add").cluster_representative(),
                args: a.args.clone(),
                dtype: a.dtype.clone(),
            })),
            other => match other.op() {
                Some(op) => {
                    let representative = op.cluster_representative();
                    if representative != *op { Arc::new(other.with_op(representative)) } else { stmt.clone() }
                }
                None => stmt.clone(),
            },
        })
    })
}

struct Memo {
    store: HashMap<(u64, IdentityKey), Vec<(Body, Identity)>>,
    order: std::collections::VecDeque<(u64, IdentityKey)>,
}

fn identities() -> &'static Mutex<Memo> {
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo { store: HashMap::new(), order: std::collections::VecDeque::new() }))
}

pub fn clear_memo() {
    let mut memo = identities().lock().unwrap();
    memo.store.clear();
    memo.order.clear();
}

/// Key the normal form of `body`, optionally with operations collapsed to compute-unit clusters. `colors` types
/// each external buffer, so differently typed roles never share a rank.
pub fn canonicalize_identity(body: &Body, cluster: bool, colors: Option<&HashMap<Name, String>>) -> Result<Identity> {
    let mut stmts = normalize_body(body)?;
    if cluster {
        stmts = normalize_body(&canonicalize_op_clusters(&stmts))?;
    }
    let colors_key = colors.map(|c| {
        let mut sorted: Vec<(Name, String)> = c.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        sorted.sort();
        sorted
    });
    let key = IdentityKey { cluster, colors: colors_key };
    {
        let found = stmts.0.identities.lock().unwrap();
        if let Some((_, identity)) = found.iter().find(|(k, _)| *k == key) {
            return Ok(identity.clone());
        }
    }
    let memo_key = (stmts.content_hash(), key.clone());
    if let Some(entries) = identities().lock().unwrap().store.get(&memo_key)
        && let Some((_, identity)) = entries.iter().find(|(b, _)| *b == stmts)
    {
        stmts.0.identities.lock().unwrap().push((key, identity.clone()));
        return Ok(identity.clone());
    }
    let (tree, arguments) = digest(&stmts, colors);
    let identity = Identity { key: hex(tree), arguments };
    stmts.0.identities.lock().unwrap().push((key, identity.clone()));
    let mut memo = identities().lock().unwrap();
    memo.store.entry(memo_key.clone()).or_default().push((stmts.clone(), identity.clone()));
    memo.order.push_back(memo_key);
    while memo.order.len() > 16384 {
        let oldest = memo.order.pop_front().unwrap();
        if let Some(entries) = memo.store.get_mut(&oldest) {
            entries.remove(0);
            if entries.is_empty() {
                memo.store.remove(&oldest);
            }
        }
    }
    Ok(identity)
}
