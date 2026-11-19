//! Relations between names, and the one test over them: do the comparisons in scope imply this one.
//!
//! A loop with a range states that its axis lies between two expressions of the coordinates around it
//! (`a0 + 2 <= a2`, `a2 < a0 + 3`). An interval per name cannot hold that, so the context keeps each such comparison
//! as an affine form that is at least zero ([`Linear`]), and [`SimplifyCtx::implies`] decides a comparison by
//! Fourier-Motzkin elimination: the comparisons in scope, the intervals of the names they read and the negation of
//! the one asked have no integer solution. Every name is an integer, so `a < b` is `a + 1 <= b` and a form's
//! constant is rounded down after its coefficients are divided by their common divisor. The test is sound and not
//! complete: it answers `false` where it cannot tell, and gives up past [`MOST_FORMS`] forms.

use std::collections::{BTreeMap, BTreeSet};

use super::{Expr, Expression, Lit, LitType, Num, Operation, SimplifyCtx};
use crate::Name;

/// The most forms one elimination holds before the test gives up.
const MOST_FORMS: usize = 96;

/// `Σ coefficient · name + constant`, read as "at least zero" where it stands for a comparison.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Linear {
    pub coefficients: BTreeMap<Name, i64>,
    pub constant: i64,
}

impl Linear {
    fn constant(value: i64) -> Linear {
        Linear {
            coefficients: BTreeMap::new(),
            constant: value,
        }
    }

    /// The affine form of an index over names and integer literals: sums, differences, negations and products by a
    /// literal. `None` for anything else, and where a coefficient would overflow.
    pub fn of(expression: &Expression) -> Option<Linear> {
        match &**expression {
            Expr::Var(name) => Some(Linear {
                coefficients: BTreeMap::from([(*name, 1)]),
                constant: 0,
            }),
            Expr::Literal(Lit {
                value: Num::Int(value),
                dtype: LitType::Int,
            }) => Some(Linear::constant(*value)),
            _ => {
                if let Some((Operation::Negative, [inner])) = expression.operation() {
                    return Linear::of(inner)?.scaled(-1);
                }
                let (op, left, right) = expression.as_binary()?;
                match op {
                    Operation::Add => Linear::of(left)?.plus(&Linear::of(right)?, 1),
                    Operation::Subtract => Linear::of(left)?.plus(&Linear::of(right)?, -1),
                    Operation::Multiply => match (left.int_literal(), right.int_literal()) {
                        (Some(factor), _) => Linear::of(right)?.scaled(factor),
                        (_, Some(factor)) => Linear::of(left)?.scaled(factor),
                        _ => None,
                    },
                    _ => None,
                }
            }
        }
    }

    fn scaled(&self, factor: i64) -> Option<Linear> {
        let mut out = Linear::constant(self.constant.checked_mul(factor)?);
        for (name, coefficient) in &self.coefficients {
            let scaled = coefficient.checked_mul(factor)?;
            if scaled != 0 {
                out.coefficients.insert(*name, scaled);
            }
        }
        Some(out)
    }

    /// `self + sign · other`.
    fn plus(&self, other: &Linear, sign: i64) -> Option<Linear> {
        let mut out = self.clone();
        out.constant = out
            .constant
            .checked_add(other.constant.checked_mul(sign)?)?;
        for (name, coefficient) in &other.coefficients {
            let sum = out
                .coefficients
                .get(name)
                .copied()
                .unwrap_or(0)
                .checked_add(coefficient.checked_mul(sign)?)?;
            if sum == 0 {
                out.coefficients.remove(name);
            } else {
                out.coefficients.insert(*name, sum);
            }
        }
        Some(out)
    }

    /// `left >= right` as a form that is at least zero.
    pub fn at_least(left: &Expression, right: &Expression) -> Option<Linear> {
        Linear::of(left)?.plus(&Linear::of(right)?, -1)
    }

    /// The form over integers at its tightest: coefficients divided by their common divisor, the constant rounded
    /// down.
    fn tightened(mut self) -> Linear {
        let divisor = self
            .coefficients
            .values()
            .fold(0i64, |divisor, coefficient| gcd(divisor, coefficient.abs()));
        if divisor > 1 {
            for coefficient in self.coefficients.values_mut() {
                *coefficient /= divisor;
            }
            self.constant = self.constant.div_euclid(divisor);
        }
        self
    }

    /// The negation over integers: `self < 0`, which is `-self - 1 >= 0`.
    fn negated(&self) -> Option<Linear> {
        let mut out = self.scaled(-1)?;
        out.constant = out.constant.checked_sub(1)?;
        Some(out)
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Whether `forms`, each at least zero, have no solution: eliminate one name after another, and a form with no name
/// left and a negative constant is the contradiction. `None` when the elimination grows past [`MOST_FORMS`] or a
/// coefficient overflows.
fn infeasible(forms: Vec<Linear>) -> Option<bool> {
    let mut forms: BTreeSet<Linear> = forms.into_iter().map(Linear::tightened).collect();
    loop {
        if forms
            .iter()
            .any(|form| form.coefficients.is_empty() && form.constant < 0)
        {
            return Some(true);
        }
        // The name that makes the fewest new forms goes first.
        let mut names: BTreeMap<Name, (usize, usize)> = BTreeMap::new();
        for form in &forms {
            for (name, coefficient) in &form.coefficients {
                let counts = names.entry(*name).or_default();
                if *coefficient > 0 {
                    counts.0 += 1;
                } else {
                    counts.1 += 1;
                }
            }
        }
        let Some((name, _)) = names.iter().min_by_key(|(_, (above, below))| above * below) else {
            return Some(false);
        };
        let name = *name;
        let (reading, rest): (Vec<Linear>, Vec<Linear>) = forms
            .into_iter()
            .partition(|form| form.coefficients.contains_key(&name));
        let mut next: BTreeSet<Linear> = rest.into_iter().collect();
        let (lower, upper): (Vec<&Linear>, Vec<&Linear>) = reading
            .iter()
            .partition(|form| form.coefficients[&name] > 0);
        for low in &lower {
            for high in &upper {
                // `p·name + L >= 0` and `-q·name + U >= 0` give `q·L + p·U >= 0`.
                let (p, q) = (low.coefficients[&name], -high.coefficients[&name]);
                let combined = low.scaled(q)?.plus(&high.scaled(p)?, 1)?;
                next.insert(combined.tightened());
                if next.len() > MOST_FORMS {
                    return None;
                }
            }
        }
        forms = next;
    }
}

impl SimplifyCtx {
    /// The context with one more comparison in scope: `form` is at least zero.
    pub fn with_relation(&self, form: Linear) -> SimplifyCtx {
        let mut out = self.clone();
        out.relations.push(form);
        out
    }

    /// Whether the context holds `name` as an integer: it has an interval, or a comparison in scope reads it.
    fn knows(&self, name: &Name) -> bool {
        self.ranges.contains_key(name)
            || self
                .relations
                .iter()
                .any(|form| form.coefficients.contains_key(name))
    }

    /// Whether the comparisons and intervals in scope imply that `goal` is at least zero. Only the forms that share
    /// a name with the goal, directly or through one another, take part. `false` where the test cannot tell, and for
    /// a goal over a name the context knows nothing about: such a name may be no integer.
    pub fn implies(&self, goal: &Linear) -> bool {
        if goal.coefficients.is_empty() {
            return goal.constant >= 0;
        }
        if !goal.coefficients.keys().all(|name| self.knows(name)) {
            return false;
        }
        let mut names: BTreeSet<Name> = goal.coefficients.keys().copied().collect();
        let mut taken = vec![false; self.relations.len()];
        loop {
            let mut grew = false;
            for (index, form) in self.relations.iter().enumerate() {
                if !taken[index] && form.coefficients.keys().any(|name| names.contains(name)) {
                    taken[index] = true;
                    names.extend(form.coefficients.keys().copied());
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        let Some(negated) = goal.negated() else {
            return false;
        };
        let mut forms: Vec<Linear> = vec![negated];
        forms.extend(
            self.relations
                .iter()
                .zip(&taken)
                .filter(|(_, taken)| **taken)
                .map(|(form, _)| form.clone()),
        );
        for name in &names {
            let Some(range) = self.ranges.get(name) else {
                continue;
            };
            if let (Ok(lo), Ok(hi)) = (i64::try_from(range.lo), i64::try_from(range.hi)) {
                forms.push(Linear {
                    coefficients: BTreeMap::from([(*name, 1)]),
                    constant: -lo,
                });
                forms.push(Linear {
                    coefficients: BTreeMap::from([(*name, -1)]),
                    constant: hi,
                });
            }
        }
        infeasible(forms) == Some(true)
    }

    /// What the comparisons in scope decide of `left op right`, where the intervals alone do not: `Some(true)` when
    /// they imply it, `Some(false)` when they imply its negation. `None` with no comparison in scope, for a side
    /// that is not affine or reads a name the context does not hold as an integer, and where neither is implied.
    pub fn decides(&self, op: Operation, left: &Expression, right: &Expression) -> Option<bool> {
        if self.relations.is_empty()
            || !(left.free_vars().iter())
                .chain(&right.free_vars())
                .all(|name| self.knows(name))
        {
            return None;
        }
        // `left - right`, and each comparison as "at least zero" of it or of its negation, shifted by one.
        let difference = Linear::at_least(left, right)?;
        let shifted = |form: &Linear, by: i64| {
            let mut out = form.clone();
            out.constant = out.constant.checked_add(by)?;
            Some(out)
        };
        let reversed = difference.scaled(-1)?;
        let (holds, fails) = match op {
            // left < right: right - left - 1 >= 0; its negation: left - right >= 0.
            Operation::Less => (shifted(&reversed, -1)?, difference.clone()),
            Operation::LessEqual => (reversed.clone(), shifted(&difference, -1)?),
            Operation::Greater => (shifted(&difference, -1)?, reversed.clone()),
            Operation::GreaterEqual => (difference.clone(), shifted(&reversed, -1)?),
            Operation::Equal => {
                if self.implies(&difference) && self.implies(&reversed) {
                    return Some(true);
                }
                let apart = self.implies(&shifted(&difference, -1)?)
                    || self.implies(&shifted(&reversed, -1)?);
                return apart.then_some(false);
            }
            _ => return None,
        };
        if self.implies(&holds) {
            Some(true)
        } else if self.implies(&fails) {
            Some(false)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Interval, parse};

    fn ctx(ranges: &[(&str, i128, i128)], relations: &[(&str, &str)]) -> SimplifyCtx {
        let mut ctx = SimplifyCtx::empty();
        for (name, lo, hi) in ranges {
            ctx = ctx.extend(name, Interval::new(*lo, *hi), None);
        }
        for (left, right) in relations {
            let form = Linear::at_least(&parse(left).unwrap(), &parse(right).unwrap()).unwrap();
            ctx = ctx.with_relation(form);
        }
        ctx
    }

    fn decides(ctx: &SimplifyCtx, comparison: &str) -> Option<bool> {
        let parsed = parse(comparison).unwrap();
        let (op, left, right) = parsed.as_binary().unwrap();
        ctx.decides(op, left, right)
    }

    #[test]
    fn a_range_that_reads_a_coordinate_decides_a_comparison_the_intervals_do_not() {
        // The solve's triangle: `a4` walks `0..a0 + 2`.
        let ctx = ctx(
            &[("a0", 0, 60), ("a4", 0, 61)],
            &[("a4", "0"), ("a0 + 1", "a4")],
        );
        assert_eq!(decides(&ctx, "a4 < a0 + 2"), Some(true));
        assert_eq!(decides(&ctx, "a4 <= a0 + 1"), Some(true));
        assert_eq!(decides(&ctx, "a4 >= a0 + 2"), Some(false));
        assert_eq!(
            decides(&ctx, "a4 < a0 + 1"),
            None,
            "the last row is in range"
        );
        assert_eq!(decides(&ctx, "a4 == a0 + 2"), Some(false));
    }

    #[test]
    fn a_comparison_follows_through_a_chain_of_ranges() {
        // `m` in a block of sixteen rows, `r` in the same block past it.
        let ctx = ctx(
            &[("rb", 0, 3), ("r", 0, 63), ("m", 0, 63)],
            &[
                ("r", "16 * rb"),
                ("16 * rb + 15", "r"),
                ("m", "16 * rb"),
                ("r - 1", "m"),
            ],
        );
        assert_eq!(decides(&ctx, "m < 16 * rb + 16"), Some(true));
        assert_eq!(decides(&ctx, "m < r"), Some(true));
        assert_eq!(decides(&ctx, "r < m"), Some(false));
        assert_eq!(
            decides(&ctx, "m + 16 < r + 17"),
            Some(true),
            "spelled after a substitution"
        );
    }

    #[test]
    fn integers_round_a_bound_that_rationals_leave_open() {
        // 2 * a >= 1 means a >= 1 over integers.
        let ctx = ctx(&[("a", 0, 9)], &[("2 * a", "1")]);
        assert_eq!(decides(&ctx, "a >= 1"), Some(true));
    }

    #[test]
    fn a_name_the_context_does_not_know_decides_nothing() {
        let ctx = ctx(&[("a", 0, 9)], &[("a", "1")]);
        assert_eq!(decides(&ctx, "x < x + 1"), None, "x may be no integer");
        assert_eq!(
            decides(&SimplifyCtx::empty(), "a < a + 1"),
            None,
            "nothing is in scope"
        );
    }
}
