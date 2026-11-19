//! A value a body defines for itself, as the computation it stands for at each read.
//!
//! A `Carry` without a seed defines one cell of a value per coordinate of the loops around it, and a `Pre` reads a
//! cell by index. [`inline_values`] spells every such read as the value's own statements at the coordinate read and
//! drops the definition: the form in which each reader computes the value for itself, which is what a lowering with
//! no storage for the value compiles. It is no step of the normal form: a body keeps its definitions, and the lift
//! that has no place for them asks for this.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::body::{Body, exposed_defines, free_names};
use crate::expr::{Expr, Expression};
use crate::statement::*;
use crate::{Error, Name, Result};

/// One definition: the statements its value needs, as the nest they stand in, and the axis of each coordinate.
struct Definition {
    name: Name,
    value: Name,
    /// Per index position, the loop axis that is the coordinate there; `None` where the index holds a literal.
    coordinates: Vec<Option<Name>>,
    cone: Body,
}

/// The names of the values `body` defines for itself: every `Carry` without a seed.
pub fn defined_values(body: &Body) -> Vec<Name> {
    let mut out = Vec::new();
    body.walk(&mut |here| {
        if let Statement::Carry(carry) = &**here.stmt()
            && carry.seed.is_none()
            && !out.contains(&carry.name)
        {
            out.push(carry.name);
        }
    });
    out
}

/// `body` cut to what the definition of `name` needs: its `Carry`, and every statement that carry's value reads
/// through, each block kept with what its own bodies keep. No other store is kept. `None` when the body holds no
/// such definition.
fn cone(body: &Body, name: Name) -> Option<Body> {
    fn cut(body: &Body, name: Name, needed: &mut HashSet<Name>) -> Vec<Arc<Statement>> {
        let mut kept: Vec<Arc<Statement>> = Vec::new();
        for stmt in body.stmts().iter().rev() {
            let nested = stmt.nested();
            let stmt = if nested.is_empty() {
                let defining =
                    matches!(&**stmt, Statement::Carry(c) if c.name == name && c.seed.is_none());
                if !(defining || exposed_defines(stmt).iter().any(|n| needed.contains(n))) {
                    continue;
                }
                stmt.clone()
            } else {
                let inner: Vec<Body> = nested
                    .iter()
                    .map(|child| Body::new(cut(child, name, needed)))
                    .collect();
                if inner.iter().all(Body::is_empty) {
                    continue;
                }
                Statement::rebuild_bodies(stmt, inner)
            };
            needed.extend(free_names(&stmt));
            kept.push(stmt);
        }
        kept.reverse();
        kept
    }
    let mut needed: HashSet<Name> = HashSet::new();
    // A state may be read ahead of its update, so the cut runs until it needs nothing more.
    loop {
        let before = needed.len();
        let kept = cut(body, name, &mut needed);
        if needed.len() == before {
            return (!kept.is_empty()).then(|| Body::new(kept));
        }
    }
}

/// The definition of `name` in `body`, or why it cannot be read as a computation: its index holds something other
/// than the axes of the loops around it, each once, and literals.
fn definition(body: &Body, name: Name) -> Result<Definition> {
    let cone = cone(body, name)
        .ok_or_else(|| Error::Value(format!("the body defines no value '{name}'")))?;
    fn find(body: &Body, name: Name, around: &mut Vec<Name>) -> Option<(Carry, Vec<Name>)> {
        for stmt in body.stmts() {
            if let Statement::Carry(carry) = &**stmt
                && carry.name == name
            {
                return Some((carry.clone(), around.clone()));
            }
            let bound = stmt.binds_axes();
            around.extend(bound.iter().copied());
            for child in stmt.nested() {
                if let Some(found) = find(&child, name, around) {
                    return Some(found);
                }
            }
            around.truncate(around.len() - bound.len());
        }
        None
    }
    let (carry, around) = find(&cone, name, &mut Vec::new())
        .ok_or_else(|| Error::Value(format!("the body defines no value '{name}'")))?;
    let coordinates: Vec<Option<Name>> = carry
        .index
        .iter()
        .map(|entry| match &**entry {
            Expr::Var(axis) if around.contains(axis) => Ok(Some(*axis)),
            Expr::Literal(_) => Ok(None),
            _ => Err(Error::Value(format!(
                "the value '{name}' is defined at '{}', which is no axis of a loop around it",
                entry.text()
            ))),
        })
        .collect::<Result<_>>()?;
    let named: HashSet<Name> = coordinates.iter().flatten().copied().collect();
    if named.len() != coordinates.iter().flatten().count() || named.len() != around.len() {
        return Err(Error::Value(format!(
            "the value '{name}' has one coordinate per loop around its definition"
        )));
    }
    Ok(Definition {
        name,
        value: carry.value,
        coordinates,
        cone,
    })
}

/// The statements of `body` with every loop over a name in `coordinates` opened: its body in its place.
fn opened(body: &Body, coordinates: &HashSet<Name>, out: &mut Vec<Arc<Statement>>) {
    for stmt in body.stmts() {
        match &**stmt {
            Statement::Loop(l) if coordinates.contains(&l.axis.name) => {
                opened(&l.body, coordinates, out)
            }
            _ => out.push(stmt.clone()),
        }
    }
}

impl Definition {
    /// The statements that compute the cell `read` asks for and bind it to the name `read` defines: the cone with
    /// each coordinate loop opened at the index read, every name of it spelled apart by `site`.
    fn at(&self, read: &Pre, site: usize) -> Result<Vec<Arc<Statement>>> {
        if read.index.len() != self.coordinates.len() {
            return Err(Error::Value(format!(
                "'{}' reads '{}' at {} coordinates and the value has {}",
                read.name,
                self.name,
                read.index.len(),
                self.coordinates.len()
            )));
        }
        let sigma: HashMap<Name, Expression> = self
            .coordinates
            .iter()
            .zip(read.index.iter())
            .filter_map(|(axis, at)| axis.map(|axis| (axis, at.clone())))
            .collect();
        let coordinates: HashSet<Name> = sigma.keys().copied().collect();
        let mut stmts = Vec::new();
        opened(&self.cone, &coordinates, &mut stmts);
        // Every name the copy binds, a value or an axis of a loop of its own, is spelled apart from the reader's.
        let mut apart: HashMap<Name, Name> = HashMap::new();
        for stmt in &stmts {
            Body::new(vec![stmt.clone()]).walk(&mut |here| {
                let stmt = here.stmt();
                for name in stmt.defines().into_iter().chain(stmt.binds_axes()) {
                    apart
                        .entry(name)
                        .or_insert_with(|| Name::new(&format!("{name}__{}{site}", self.name)));
                }
            });
        }
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in &stmts {
            if let Statement::Carry(carry) = &**stmt
                && carry.name == self.name
            {
                let value = apart.get(&self.value).copied().unwrap_or(self.value);
                out.push(Arc::new(Statement::Assign(Assign {
                    name: read.name,
                    value: Expression::var(&value),
                    dtype: None,
                })));
                continue;
            }
            let renamed = Statement::rename(stmt, &apart);
            out.push(Statement::substitute(&renamed, &sigma));
        }
        Ok(out)
    }
}

/// `body` with every statement mapped through `f`, at any depth, and every block left empty by it dropped.
fn rewritten(
    body: &Body,
    f: &mut dyn FnMut(&Arc<Statement>) -> Result<Option<Vec<Arc<Statement>>>>,
) -> Result<Body> {
    let mut out = Vec::new();
    for stmt in body.stmts() {
        match f(stmt)? {
            Some(replaced) => out.extend(replaced),
            None => {
                let nested = stmt.nested();
                if nested.is_empty() {
                    out.push(stmt.clone());
                    continue;
                }
                let inner = nested
                    .iter()
                    .map(|child| rewritten(child, f))
                    .collect::<Result<Vec<Body>>>()?;
                if inner.iter().all(Body::is_empty) && !nested.iter().all(Body::is_empty) {
                    continue;
                }
                out.push(Statement::rebuild_bodies(stmt, inner));
            }
        }
    }
    Ok(body.rebuilt(out))
}

/// `body` with no value of its own left: every read of one is the value's statements at the coordinate read, and
/// the definition is gone. A value defined from another is spelled after that one. The result is not in normal form.
pub fn inline_values(body: &Body) -> Result<Body> {
    let mut body = body.clone();
    let mut site = 0;
    loop {
        let names = defined_values(&body);
        if names.is_empty() {
            return Ok(body);
        }
        // A definition that reads no other one goes first, so every copy made is of plain statements.
        let mut ready: Option<Definition> = None;
        for name in &names {
            let candidate = definition(&body, *name)?;
            let mut reads_another = false;
            candidate.cone.walk(&mut |here| {
                reads_another |=
                    matches!(&**here.stmt(), Statement::Pre(p) if names.contains(&p.carrier));
            });
            if !reads_another {
                ready = Some(candidate);
                break;
            }
        }
        let Some(found) = ready else {
            return Err(Error::Value(format!(
                "the values {names:?} are defined from one another"
            )));
        };
        body = rewritten(&body, &mut |stmt| match &**stmt {
            Statement::Carry(c) if c.name == found.name && c.seed.is_none() => Ok(Some(Vec::new())),
            Statement::Pre(p) if p.carrier == found.name => {
                site += 1;
                found.at(p, site).map(Some)
            }
            _ => Ok(None),
        })?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_body;
    use crate::text::{body_text, parse_body};

    fn normal(text: &str) -> String {
        body_text(&normalize_body(&parse_body(text).unwrap()).unwrap()).unwrap()
    }

    fn inlined(text: &str) -> String {
        let body = normalize_body(&parse_body(text).unwrap()).unwrap();
        body_text(&normalize_body(&inline_values(&body).unwrap()).unwrap()).unwrap()
    }

    const DEFINED: &str = "for h in 0..4:
  for a in 0..8:
    k = load K[h, a]
    w = load W[h]
    v = multiply(k, w)
    kn[h, a] <- v
  for n in 0..8:
    for m in 0..8:
      b = pre kn[h, m]
      c = pre kn[h, n]
      p = multiply(b, c)
      s <- add(s, p) over m
    out[h, n] = s";

    const COPIED: &str = "for h in 0..4:
  for n in 0..8:
    for m in 0..8:
      w = load W[h]
      k = load K[h, m]
      b = multiply(k, w)
      j = load K[h, n]
      c = multiply(j, w)
      p = multiply(b, c)
      s <- add(s, p) over m
    out[h, n] = s";

    #[test]
    fn a_value_read_at_two_places_inlines_to_the_body_that_computes_it_at_each() {
        assert_eq!(inlined(DEFINED), normal(COPIED));
        assert!(defined_values(&parse_body(COPIED).unwrap()).is_empty());
    }

    #[test]
    fn a_value_defined_from_another_inlines_after_it() {
        let chained = "for h in 0..4:
  for a in 0..8:
    k = load K[h, a]
    kn[h, a] <- k
  for a in 0..8:
    x = pre kn[h, a]
    y = exp(x)
    ke[h, a] <- y
  for n in 0..8:
    for m in 0..8:
      b = pre ke[h, m]
      c = pre ke[h, n]
      p = multiply(b, c)
      s <- add(s, p) over m
    out[h, n] = s";
        let copied = "for h in 0..4:
  for n in 0..8:
    for m in 0..8:
      k = load K[h, m]
      b = exp(k)
      j = load K[h, n]
      c = exp(j)
      p = multiply(b, c)
      s <- add(s, p) over m
    out[h, n] = s";
        assert_eq!(inlined(chained), normal(copied));
    }

    #[test]
    fn a_value_folded_by_a_loop_of_its_own_keeps_that_loop_at_each_read() {
        let defined = "for h in 0..4:
  for a in 0..8:
    for d in 0..16:
      x = load X[h, a, d]
      t <- add(t, x) over d
    g[h, a] <- t
  for n in 0..8:
    for m in 0..8:
      b = pre g[h, m]
      c = pre g[h, n]
      p = multiply(b, c)
      s <- add(s, p) over m
    out[h, n] = s";
        let text = inlined(defined);
        assert!(!text.contains("pre ") && !text.contains("<- v"), "{text}");
        assert_eq!(text.matches("load X[").count(), 2, "{text}");
    }

    #[test]
    fn a_seeded_state_is_no_value_to_inline() {
        let carried = "for t in 0..8:
  for i in 0..4:
    p = pre s[i]
    x = load X[t, i]
    v = add(p, x)
    s[i] <- v seed 0
    out[t, i] = v";
        let body = normalize_body(&parse_body(carried).unwrap()).unwrap();
        assert!(inline_values(&body).unwrap().ptr_eq(&body));
    }
}
