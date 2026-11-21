//! A value read by index, as the computation it stands for.
//!
//! A term that reads its operand by index ([`Fold::as_read`]) names a value defined once, over its own coordinates.
//! No tier keeps such a value for its readers, so each reader computes it for itself: the operand with every
//! coordinate the read binds replaced by the index read. [`Fold::inlined`] gives a tree in that form, the one every
//! pass and the kernel's identity read.
//!
//! One copy is taken of a term per substitution of the coordinates THAT TERM is evaluated over. Two reads that
//! differ only in a coordinate a term does not read share that term, as two readers of one value at one place do:
//! a softmax's row maximum is one term under the sum over a row and under each cell of it.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::axis::Axis;
use emmy_ir::expr::Expression;
use emmy_ir::statement::Statement;
use emmy_ir::{Name, Result};

use crate::fold::Fold;
use crate::rewrite::TermRewrite;

/// Every name `term` itself defines: what it exposes, what its lift, its ⊕ and its observer bind and define, and
/// the axis it walks.
fn own_names(term: &Fold) -> Vec<Name> {
    let mut names: Vec<Name> = term.exposes().to_vec();
    names.extend(term.lift().results());
    for lambda in [Some(term.lift()), term.observe()].into_iter().flatten() {
        for here in lambda.body().iter_refs() {
            names.extend(here.stmt().defines());
        }
    }
    for lambda in [term.base(), term.observe()].into_iter().flatten() {
        names.extend(lambda.params());
        names.extend(lambda.results());
    }
    names.extend(term.axis());
    names
}

/// A substitution as a key: its entries in name order.
type Key = Vec<(Name, Expression)>;

struct Inliner {
    /// The copy of each term under each substitution of the coordinates it is evaluated over.
    copies: HashMap<(usize, Key), Fold>,
    /// The value read of each definition at each index, by the components read: two reads of one value at one
    /// index are one term, as two readers of it at one place are.
    values: HashMap<(usize, Key, Vec<usize>), Fold>,
    /// How many substitutions each term has been copied under.
    taken: HashMap<usize, usize>,
    /// Whether a read by index stands in a term's tree.
    reads: HashMap<usize, bool>,
    table: Vec<Arc<Axis>>,
    apart: usize,
}

impl Inliner {
    fn holds_reads(&mut self, term: &Fold) -> bool {
        if let Some(found) = self.reads.get(&term.address()) {
            return *found;
        }
        let found =
            term.as_read().is_some() || term.operands().iter().any(|edge| self.holds_reads(edge));
        self.reads.insert(term.address(), found);
        found
    }

    /// Whether `term` has been copied before, and the number its names are spelled apart by if so.
    fn again(&mut self, term: &Fold) -> Option<usize> {
        let taken = self.taken.entry(term.address()).or_default();
        *taken += 1;
        (*taken > 1).then(|| {
            self.apart += 1;
            self.apart
        })
    }

    /// `term` under `sigma`, every read by index in it replaced by the value read.
    fn copy(&mut self, term: &Fold, sigma: &HashMap<Name, Expression>) -> Result<Fold> {
        // What moves `term` is what `sigma` does to the coordinates it is evaluated over.
        let mut key: Key = term
            .free_axes()
            .iter()
            .filter_map(|name| sigma.get(name).map(|at| (*name, at.clone())))
            .filter(|(name, at)| at.as_var() != Some(name))
            .collect();
        key.sort_by_key(|(name, _)| *name);
        if key.is_empty() && !self.holds_reads(term) {
            return Ok(term.clone());
        }
        let slot = (term.address(), key);
        if let Some(found) = self.copies.get(&slot) {
            return Ok(found.clone());
        }
        let moved: HashMap<Name, Expression> = slot.1.iter().cloned().collect();
        let out = match term.as_read() {
            Some(index) => self.value(term, index, &moved)?,
            None => self.moved(term, moved)?,
        };
        self.copies.insert(slot, out.clone());
        Ok(out)
    }

    /// The value the read `read` stands for: its operand with each coordinate the read binds at the index read,
    /// exposing the read's names.
    fn value(
        &mut self,
        read: &Fold,
        index: &[Expression],
        sigma: &HashMap<Name, Expression>,
    ) -> Result<Fold> {
        let definition = &read.operands()[0];
        let at: HashMap<Name, Expression> = read
            .cells()
            .iter()
            .copied()
            .zip(index.iter().map(|entry| entry.substitute(sigma)))
            .collect();
        let copy = self.copy(definition, &at)?;
        let bound: HashMap<Name, usize> = read
            .bindings()
            .iter()
            .map(|binding| (binding.param, binding.component))
            .collect();
        let mut components: Vec<usize> = Vec::new();
        let mut reads: Vec<Name> = Vec::new();
        for stmt in read.lift().body().stmts() {
            if let Statement::Pre(pre) = &**stmt {
                components.push(bound[&pre.carrier]);
                reads.push(pre.name);
            }
        }
        let mut key: Key = definition
            .free_axes()
            .iter()
            .filter_map(|name| at.get(name).map(|at| (*name, at.clone())))
            .collect();
        key.sort_by_key(|(name, _)| *name);
        let slot = (definition.address(), key, components);
        // Binding is positional, so the value two reads share keeps the names the first read gave it.
        if let Some(found) = self.values.get(&slot) {
            return Ok(found.clone());
        }
        // The values read take the read's names: as they stand for the first copy of the read, spelled apart
        // for every later one, since a read that stands in a value is copied with it.
        let again = self.again(read);
        let taken: Vec<Name> = reads
            .iter()
            .map(|name| match again {
                Some(apart) => Name::new(&format!("{name}__r{apart}")),
                None => *name,
            })
            .collect();
        let names: HashMap<Name, Name> = slot
            .2
            .iter()
            .zip(&taken)
            .map(|(component, name)| (copy.exposes()[*component], *name))
            .collect();
        let renamed = TermRewrite {
            names,
            ..TermRewrite::default()
        };
        let value = renamed
            .node(&copy, copy.operands().to_vec(), &HashMap::new())?
            .exposing(&taken)?;
        self.values.insert(slot, value.clone());
        Ok(value)
    }

    /// `term` under the substitution `moved` of its own coordinates: its operands copied, and the term rewritten
    /// over them. The first copy of a term keeps the names it has; every later one spells apart what the term
    /// defines and the axis it walks, which is what lets it stand beside the first.
    fn moved(&mut self, term: &Fold, moved: HashMap<Name, Expression>) -> Result<Fold> {
        let again = self.again(term);
        let mut names: HashMap<Name, Name> = HashMap::new();
        let mut below = moved.clone();
        if let Some(apart) = again {
            for name in own_names(term) {
                names
                    .entry(name)
                    .or_insert_with(|| Name::new(&format!("{name}__r{apart}")));
            }
            if let Some(axis) = term.axis() {
                let renamed = names[&axis];
                below.insert(axis, Expression::var(&renamed));
                if let Some(known) = self.table.iter().find(|known| known.name == axis).cloned() {
                    self.table.push(Arc::new(Axis {
                        name: renamed,
                        ..(*known).clone()
                    }));
                }
            }
        }
        let operands = term
            .operands()
            .iter()
            .map(|edge| self.copy(edge, &below))
            .collect::<Result<Vec<_>>>()?;
        if moved.is_empty()
            && again.is_none()
            && operands
                .iter()
                .zip(term.operands())
                .all(|(new, old)| new.ptr_eq(old))
        {
            return Ok(term.clone());
        }
        let rewrite = TermRewrite {
            names,
            ..TermRewrite::default()
        };
        rewrite.node(term, operands, &moved)
    }
}

impl Fold {
    /// This tree with every read by index replaced by the value it reads, and the axis table it then reads: an
    /// axis a later copy of a term walks under a name of its own joins the table. A tree that reads nothing by
    /// index is the tree itself.
    pub fn inlined(&self, axes: &[Arc<Axis>]) -> Result<(Fold, Vec<Arc<Axis>>)> {
        let mut inliner = Inliner {
            copies: HashMap::new(),
            values: HashMap::new(),
            taken: HashMap::new(),
            reads: HashMap::new(),
            table: axes.to_vec(),
            apart: 0,
        };
        if !inliner.holds_reads(self) {
            return Ok((self.clone(), inliner.table));
        }
        let out = inliner.copy(self, &HashMap::new())?;
        Ok((out, inliner.table))
    }
}
