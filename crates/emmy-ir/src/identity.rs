//! A body's identity: the digest of its value-numbered scope tree, with every external buffer keyed by its type and
//! its use instead of its spelling, and the roles its buffers fill.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::body::Body;
use crate::kinds::elementwise::Op;
use crate::kinds::{Assign, Statement};
use crate::memo::Memo;
use crate::normalize::normalize_body;
use crate::numbering::{digest, hex};
use crate::text::body_text;
use crate::{Name, Result, locked};

/// How many identities the process keeps: a cut's arms lower the same pieces again and again.
const MEMO_SIZE: usize = 16384;

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
                    if representative != *op {
                        Arc::new(other.with_op(representative))
                    } else {
                        stmt.clone()
                    }
                }
                None => stmt.clone(),
            },
        })
    })
}

/// The identities of the bodies met in this process, per coloring.
fn identities() -> &'static Mutex<Memo<(u64, IdentityKey), Identity>> {
    static MEMO: OnceLock<Mutex<Memo<(u64, IdentityKey), Identity>>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo::new(MEMO_SIZE)))
}

pub fn clear_memo() {
    locked(identities()).clear();
}

/// Key the normal form of `body`, optionally with operations collapsed to compute-unit clusters. `colors` types
/// each external buffer, so differently typed roles never share a rank.
pub fn canonicalize_identity(
    body: &Body,
    cluster: bool,
    colors: Option<&HashMap<Name, String>>,
) -> Result<Identity> {
    let mut stmts = normalize_body(body)?;
    if cluster {
        stmts = normalize_body(&canonicalize_op_clusters(&stmts))?;
    }
    let colors_key = colors.map(|c| {
        let mut sorted: Vec<(Name, String)> =
            c.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        sorted.sort();
        sorted
    });
    let key = IdentityKey {
        cluster,
        colors: colors_key,
    };
    let memo_key = (stmts.content_hash(), key.clone());
    let text = body_text(&stmts)?;
    let remembered = locked(identities()).get(&memo_key, &text);
    if let Some(identity) = remembered {
        return Ok(identity);
    }
    let (tree, arguments) = digest(&stmts, colors);
    let identity = Identity {
        key: hex(tree),
        arguments,
    };
    locked(identities()).put(memo_key, text, identity.clone());
    Ok(identity)
}
