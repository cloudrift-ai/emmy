//! `Lambda`: the one binder kind every IR level shares. Explicit binders over the pure statement vocabulary, an
//! A-normal-form let-chain in `body`, the returned definitions named by `results`. It binds; it never executes.
//!
//! Construction normalizes the body ([`normal_body`]) and checks the local formation invariant: every statement is
//! pure, every result is defined, and every name the body reads is bound. A lambda is CLOSED: [`Lambda::closing`]
//! forms one by appending what the body still reads as trailing params; [`Lambda::new`] only refuses. α-invariance is
//! canonical renumbering ([`Lambda::canonical`]), not de Bruijn indices.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use crate::body::{Body, exposed_defines};
use crate::elementwise::ElementwiseOp;
use crate::expr::{Expr, Expression};
use crate::normalize::{canonical_values, fragment_order, one_operation_per_definition};
use crate::statement::{Assign, PURE, Statement};
use crate::{Error, Name, Result};

/// A pure function over the statement vocabulary. Immutable and shared: a clone is the same lambda.
#[derive(Clone)]
pub struct Lambda(Arc<Data>);

struct Data {
    params: Arc<[Name]>,
    body: Body,
    results: Arc<[Name]>,
    canonical: OnceLock<Lambda>,
}

impl PartialEq for Lambda {
    fn eq(&self, other: &Lambda) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.params == other.0.params
                && self.0.results == other.0.results
                && self.0.body == other.0.body)
    }
}

impl Eq for Lambda {}

impl Hash for Lambda {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.params.hash(state);
        self.0.body.hash(state);
        self.0.results.hash(state);
    }
}

impl std::fmt::Debug for Lambda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lambda")
            .field("params", &self.0.params)
            .field("results", &self.0.results)
            .finish_non_exhaustive()
    }
}

/// A lambda body's normal form under `params`: its statements in the canonical order with its names as they stand (a
/// read of a param ahead of its first binding reads the parameter), then every value in its canonical spelling.
pub fn normal_body(body: &Body, params: &[Name]) -> Result<Body> {
    let ordered = if body.len() > 1 {
        fragment_order(body, &params.iter().copied().collect())?
    } else {
        body.clone()
    };
    Ok(canonical_values(&ordered))
}

/// Names as Python spells a list of them: what an error message shows.
fn listed(names: &[impl AsRef<str>]) -> String {
    let items: Vec<String> = names
        .iter()
        .map(|name| format!("'{}'", name.as_ref()))
        .collect();
    format!("[{}]", items.join(", "))
}

impl Lambda {
    /// A lambda over a body already in its normal form under `params`, checks taken as passed: what a rename that keeps
    /// every name apart, or a value read back from transport, builds.
    pub fn formed(params: Vec<Name>, body: Body, results: Vec<Name>) -> Lambda {
        Lambda(Arc::new(Data {
            params: params.into(),
            body,
            results: results.into(),
            canonical: OnceLock::new(),
        }))
    }

    /// The lambda over these parts: the body normalized, then the formation checks, which refuse rather than repair.
    pub fn new(params: Vec<Name>, body: &Body, results: Vec<Name>) -> Result<Lambda> {
        let body = normal_body(body, &params)?;
        let impure: Vec<&str> = body
            .stmts()
            .iter()
            .filter(|stmt| !stmt.flags().contains(PURE))
            .map(|stmt| stmt.ty().name())
            .collect();
        if !impure.is_empty() {
            return Err(Error::Value(format!(
                "Lambda body must be pure — impure statement type(s): {}",
                listed(&impure)
            )));
        }
        let defined: HashSet<Name> = params.iter().chain(body.defined()).copied().collect();
        let missing: Vec<Name> = results
            .iter()
            .filter(|r| !defined.contains(r))
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(Error::Value(format!(
                "Lambda results {} are not defined by the body or params",
                listed(&missing)
            )));
        }
        let mut free: Vec<Name> = body
            .free()
            .iter()
            .filter(|name| !defined.contains(name))
            .copied()
            .collect();
        if !free.is_empty() {
            free.sort();
            free.dedup();
            return Err(Error::Value(format!(
                "Lambda body reads {} it does not bind. Pass them as params — Lambda.closing(params, body, results) \
                 appends whatever the body still reads.",
                listed(&free)
            )));
        }
        Ok(Lambda::formed(params, body, results))
    }

    /// The closed lambda over these parts, the one former: every name the body still reads, and every result it does
    /// not define, is appended as a TRAILING param in sorted order, so the positional prefix the caller gave stays
    /// intact. A write may pass an enclosing value straight through, and that result binds as a param like any read.
    pub fn closing(params: Vec<Name>, body: &Body, results: Vec<Name>) -> Result<Lambda> {
        let body = normal_body(body, &params)?;
        let bound: HashSet<Name> = params.iter().chain(body.defined()).copied().collect();
        let mut residual: Vec<Name> = body
            .free()
            .iter()
            .chain(&results)
            .filter(|name| !bound.contains(name))
            .copied()
            .collect();
        residual.sort();
        residual.dedup();
        let mut all = params;
        all.extend(residual);
        Lambda::new(all, &body, results)
    }

    /// The componentwise binary program `nᵢ = ⊕ᵢ(nᵢ, nᵢ__o)`: `S × S → S`, the second operand spelled `<n>__o`.
    pub fn componentwise(ops: &[ElementwiseOp], names: &[Name]) -> Result<Lambda> {
        if ops.len() != names.len() {
            return Err(Error::Value(format!(
                "{} ops for {} names",
                ops.len(),
                names.len()
            )));
        }
        let other: Vec<Name> = names
            .iter()
            .map(|name| Name::new(&format!("{name}__o")))
            .collect();
        let stmts: Vec<Arc<Statement>> = names
            .iter()
            .zip(ops)
            .zip(&other)
            .map(|((name, op), second)| {
                Arc::new(Statement::Assign(Assign {
                    name: *name,
                    value: Expression::call(
                        op.name(),
                        vec![Expression::var(name), Expression::var(second)],
                    ),
                    dtype: None,
                }))
            })
            .collect();
        let mut params = names.to_vec();
        params.extend(other);
        Lambda::new(params, &Body::new(stmts), names.to_vec())
    }

    pub fn params(&self) -> &[Name] {
        &self.0.params
    }

    pub fn body(&self) -> &Body {
        &self.0.body
    }

    pub fn results(&self) -> &[Name] {
        &self.0.results
    }

    pub fn ptr_eq(&self, other: &Lambda) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Every name this lambda binds: the params and every definition its body exposes, deep.
    pub fn defined(&self) -> HashSet<Name> {
        self.0
            .params
            .iter()
            .chain(self.0.body.defined())
            .copied()
            .collect()
    }

    /// Every name the lambda spells: params, results, and every SSA and axis name of its body.
    pub fn names(&self) -> HashSet<Name> {
        let mut out: HashSet<Name> = self
            .0
            .params
            .iter()
            .chain(self.0.results.iter())
            .copied()
            .collect();
        for stmt in self.0.body.stmts() {
            out.extend(stmt.names());
        }
        out
    }

    /// The per-result ops when this program is componentwise — every result `rᵢ = ⊕ᵢ(pᵢ, pₙ₊ᵢ)` on its own, in either
    /// argument order for a commutative ⊕ — else `None`.
    pub fn components(&self) -> Option<Vec<ElementwiseOp>> {
        let n = self.0.results.len();
        if self.0.body.len() != n || self.0.params.len() != 2 * n {
            return None;
        }
        let definitions = self.0.body.definitions();
        let mut ops = Vec::with_capacity(n);
        for (index, result) in self.0.results.iter().enumerate() {
            let pair = (self.0.params[index], self.0.params[n + index]);
            let here = definitions.get(result)?;
            let Statement::Assign(assign) = &**here.stmt() else {
                return None;
            };
            let Expr::Apply(op, args) = &*assign.value else {
                return None;
            };
            let operands: Vec<Option<Name>> =
                args.iter().map(|arg| arg.as_var().copied()).collect();
            let straight = operands == [Some(pair.0), Some(pair.1)];
            let swapped = operands == [Some(pair.1), Some(pair.0)];
            if !straight && !(op.commutative() && swapped) {
                return None;
            }
            ops.push(op.clone());
        }
        Some(ops)
    }

    /// The closed cone of one definition — the statements `name` depends on, as a lambda over what they read; a param
    /// names itself (an empty body returning it). A param's re-binding is hidden from the walk, so a read of the
    /// incoming value closes back as that param instead of resolving forward to the trailing write that re-spells it.
    pub fn cone(&self, name: Name) -> Result<Lambda> {
        if self.0.params.contains(&name) {
            return Lambda::closing(Vec::new(), &Body::empty(), vec![name]);
        }
        let params: HashSet<Name> = self.0.params.iter().copied().collect();
        let visible: Vec<Arc<Statement>> = self
            .0
            .body
            .stmts()
            .iter()
            .filter(|stmt| {
                let defines = exposed_defines(stmt);
                !defines.iter().any(|d| params.contains(d)) || defines.contains(&name)
            })
            .cloned()
            .collect();
        let visible = Body::new(visible);
        let members: Vec<Arc<Statement>> = visible
            .backward_cone(&[name])
            .0
            .iter()
            .map(|here| here.stmt().clone())
            .collect();
        Lambda::closing(Vec::new(), &Body::new(members), vec![name])
    }

    /// α-rename params, body and results through one map, in lockstep. A map that keeps every name apart keeps the
    /// formation checks true, so its result is formed without running them again; one that merges names is checked.
    pub fn rename(&self, mapping: &HashMap<Name, Name>) -> Result<Lambda> {
        if mapping.is_empty() {
            return Ok(self.clone());
        }
        let spelled = self.names();
        let image: HashSet<Name> = spelled
            .iter()
            .map(|name| *mapping.get(name).unwrap_or(name))
            .collect();
        let rename = |name: &Name| *mapping.get(name).unwrap_or(name);
        let params: Vec<Name> = self.0.params.iter().map(rename).collect();
        let results: Vec<Name> = self.0.results.iter().map(rename).collect();
        // Names only, no axis map: the rewrite a statement's own `rewrite(names)` makes.
        let empty = HashMap::new();
        let by_role = |_: crate::statement::Roles, name: Name| mapping.get(&name).copied();
        let rewrite = crate::protocol::Rewrite {
            rename: &by_role,
            sigma: &empty,
            axis_fn: None,
        };
        let renamed = self
            .0
            .body
            .map_each(&mut |stmt| Statement::rewrite(stmt, &rewrite));
        if image.len() != spelled.len() {
            return Lambda::new(params, &renamed, results);
        }
        Ok(Lambda::formed(
            params.clone(),
            normal_body(&renamed, &params)?,
            results,
        ))
    }

    /// The α-canonical form: every definition one operation (a result stays a statement), then the params renumbered
    /// `_p0…` and the body's own definitions `_v0…` in walk order; free names pass through. Two lambdas computing one
    /// thing have equal canonical forms. Computed once.
    pub fn canonical(&self) -> Result<Lambda> {
        if let Some(found) = self.0.canonical.get() {
            return Ok(found.clone());
        }
        let exposed: HashSet<Name> = self.0.results.iter().copied().collect();
        let split = Lambda::new(
            self.0.params.to_vec(),
            &one_operation_per_definition(&self.0.body, &exposed),
            self.0.results.to_vec(),
        )?;
        let mut mapping: HashMap<Name, Name> = HashMap::new();
        for (index, param) in split.params().iter().enumerate() {
            mapping
                .entry(*param)
                .or_insert_with(|| Name::new(&format!("_p{index}")));
        }
        let mut counter = 0;
        for here in split.body().iter_refs() {
            for name in here.stmt().defines() {
                if !mapping.contains_key(&name) {
                    mapping.insert(name, Name::new(&format!("_v{counter}")));
                    counter += 1;
                }
            }
        }
        let canonical = split.rename(&mapping)?;
        let _ = self.0.canonical.set(canonical.clone());
        Ok(canonical)
    }

    /// α-invariant equality: canonical forms compared structurally.
    pub fn alpha_eq(&self, other: &Lambda) -> Result<bool> {
        Ok(self.canonical()? == other.canonical()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse_body;

    fn names(items: &[&str]) -> Vec<Name> {
        items.iter().map(|item| Name::new(item)).collect()
    }

    #[test]
    fn closing_appends_what_the_body_reads() {
        let body = parse_body("t = add(x, c)").unwrap();
        let lam = Lambda::closing(names(&["x"]), &body, names(&["t"])).unwrap();
        assert_eq!(lam.params(), names(&["x", "c"]).as_slice());
        let error = Lambda::new(names(&["x"]), &body, names(&["t"])).unwrap_err();
        assert!(format!("{error:?}").contains("reads ['c']"), "{error:?}");
    }

    #[test]
    fn a_renamed_lambda_is_alpha_equal_and_componentwise_reads_back() {
        let ops = [ElementwiseOp::new("maximum"), ElementwiseOp::new("add")];
        let lam = Lambda::componentwise(&ops, &names(&["m", "l"])).unwrap();
        assert_eq!(lam.components().unwrap(), ops.to_vec());
        let renamed = lam
            .rename(&HashMap::from([
                (Name::new("m"), Name::new("acc")),
                (Name::new("m__o"), Name::new("acc__o")),
            ]))
            .unwrap();
        assert!(lam.alpha_eq(&renamed).unwrap());
        assert_ne!(lam, renamed);
    }
}
