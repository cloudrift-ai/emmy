//! The proof that bodies are one step taken again and again, and that step at any place in the walk.
//!
//! The bodies are numbered with the integer of every affine coordinate expression apart ([`value_numbers_apart`]).
//! They are one step where their scope trees are one tree once those integers are left out, and where every integer
//! moves by one amount from each body to the next. A statement is paired with the statement of the next body by
//! what it computes and where it nests, never by where it stands in the text.

use std::collections::HashMap;
use std::sync::Arc;

use super::{
    Chains, Effects, Key, Number, Numbering, scope_tree_apart, spelled_key, value_numbers_apart,
};
use crate::Name;
use crate::axis::{Axis, Dim};
use crate::body::Body;
use crate::elementwise::Operation;
use crate::expr::{Expr, Expression};
use crate::statement::{Loop, Statement};

/// What [`prove`] found: the first body, how far each of its integers moves per step, and how many of the bodies
/// it holds for.
#[derive(Debug, Clone)]
pub struct Proved {
    /// The first body, every expression node and loop axis its own, so one integer is one place.
    step: Body,
    /// Each integer of `step`, in the order its numbering met it: where it is written, its value, its move per step.
    moves: Vec<(usize, bool, i64, i64)>,
    /// How many of the bodies, from the first, are that step: at least two.
    pub steps: usize,
    /// Why the body after those is no such step, where there is one.
    pub stopped: Option<String>,
}

/// Where the first body stands in the walk [`Proved::at`] spells: a step number, or the axis of the loop that walks.
pub enum By {
    Step(i64),
    Axis(Name),
}

/// `body` with every expression node and every loop axis an object of its own: two places that held one shared
/// node are then two nodes, and a place is told by its node.
fn unshared(body: &Body) -> Body {
    fn fresh(e: &Expression) -> Expression {
        let over = e.map_children(&mut |child| fresh(child));
        Expression::new(over.0.expr.clone())
    }
    body.map(&mut |stmt| {
        let stmt = Statement::map_expressions(stmt, &mut |e| fresh(e));
        Some(vec![match &*stmt {
            Statement::Loop(l) => Arc::new(Statement::Loop(Loop {
                axis: Arc::new((*l.axis).clone()),
                ..l.clone()
            })),
            _ => stmt,
        }])
    })
}

/// A body as the proof reads it: its apart numbering, its tree with the integers left out, and its integers in
/// the order of that tree.
struct Read<'a> {
    body: &'a Body,
    numbering: Numbering,
    shape: Number,
    sites: Vec<i64>,
}

impl Read<'_> {
    fn of(body: &Body) -> Read<'_> {
        let numbering = value_numbers_apart(body, &spelled_key, Chains::Flat);
        let mut sites = Vec::new();
        let shape = scope_tree_apart(
            &numbering,
            body,
            &|_, _| Key::None,
            &mut Effects::new(),
            &mut sites,
        );
        Read {
            body,
            numbering,
            shape,
            sites,
        }
    }

    /// The tree with every integer keyed by the value `value` gives its site.
    fn valued(&self, value: &dyn Fn(i64) -> i64) -> Number {
        scope_tree_apart(
            &self.numbering,
            self.body,
            &|site, _| Key::Int(value(site)),
            &mut Effects::new(),
            &mut Vec::new(),
        )
    }
}

/// The proof that `bodies` are one step taken again and again, as far as they are: the first two must be, and the
/// proof stops at the first body after them that is not, saying why.
pub fn prove(bodies: &[Body]) -> std::result::Result<Proved, String> {
    if bodies.len() < 2 {
        return Err("one body is no step taken again".to_string());
    }
    let step = unshared(&bodies[0]);
    let first = Read::of(&step);
    // Each integer of the first body, by where the tree holds it: its value at every step so far.
    let mut values: Vec<Vec<i64>> = first
        .sites
        .iter()
        .map(|site| vec![first.numbering.sites[*site as usize].value])
        .collect();
    let mut steps = 1;
    let mut stopped = None;
    for (index, body) in bodies.iter().enumerate().skip(1) {
        let other = Read::of(body);
        let why = if other.shape != first.shape || other.sites.len() != first.sites.len() {
            Some(format!(
                "step {index} is another body once the integers are left out"
            ))
        } else {
            let its: Vec<i64> = other
                .sites
                .iter()
                .map(|site| other.numbering.sites[*site as usize].value)
                .collect();
            let off = values.iter().zip(&its).find_map(|(held, value)| {
                let delta = if index == 1 {
                    value - held[0]
                } else {
                    held[1] - held[0]
                };
                let expected = held[0] + delta * index as i64;
                (*value != expected).then(|| {
                    format!(
                        "an integer of step {index} is {value}, where {} moving {delta} a step gives {expected}",
                        held[0]
                    )
                })
            });
            match off {
                Some(why) => Some(why),
                None => {
                    // The pairing above is the trees' order, which two members that key alike share by the order
                    // they are written in. The first body with its integers moved to this step is this body exactly
                    // when their trees agree with every integer keyed by its value.
                    let moved: HashMap<i64, i64> = first
                        .sites
                        .iter()
                        .zip(&its)
                        .map(|(site, value)| (*site, *value))
                        .collect();
                    let own = |site: i64| other.numbering.sites[site as usize].value;
                    (first.valued(&|site| moved[&site]) != other.valued(&own)).then(|| {
                        format!("step {index} pairs its integers with the first step's another way")
                    })
                }
            }
        };
        if let Some(why) = why {
            stopped = Some(why);
            break;
        }
        for (held, site) in values.iter_mut().zip(&other.sites) {
            held.push(other.numbering.sites[*site as usize].value);
        }
        steps += 1;
    }
    if steps < 2 {
        return Err(stopped.unwrap_or_default());
    }
    let mut moves: Vec<(usize, bool, i64, i64)> =
        vec![(0, false, 0, 0); first.numbering.sites.len()];
    for (site, held) in first.sites.iter().zip(&values) {
        let at = &first.numbering.sites[*site as usize];
        moves[*site as usize] = (at.place, at.extent, held[0], held[1] - held[0]);
    }
    Ok(Proved {
        step,
        moves,
        steps,
        stopped,
    })
}

impl Proved {
    /// Every integer that moves: its value in the first body and its move per step.
    pub fn moving(&self) -> Vec<(i64, i64)> {
        self.moves
            .iter()
            .filter(|(_, _, _, delta)| *delta != 0)
            .map(|(_, _, base, delta)| (*base, *delta))
            .collect()
    }

    /// How many integers the first body's coordinate expressions and static extents hold.
    pub fn integers(&self) -> usize {
        self.moves.len()
    }

    /// The step at `by`: the first body with every integer moved to that step, or to the step the axis `by` names.
    /// A reduction whose extent grows with the step takes, under an axis, its widest extent over `steps` steps and
    /// walks the range of the step it is in; a loop that stores cannot grow so.
    pub fn at(&self, by: &By, steps: usize) -> std::result::Result<Body, String> {
        let add = |delta: i64| match by {
            By::Step(step) => Expression::int(delta * step),
            By::Axis(axis) => Expression::binary(
                Operation::Multiply,
                Expression::new(Expr::Var(*axis)),
                Expression::int(delta),
            ),
        };
        let places: HashMap<usize, (i64, i64)> = self
            .moves
            .iter()
            .filter(|(_, _, _, delta)| *delta != 0)
            .map(|(place, _, base, delta)| (*place, (*base, *delta)))
            .collect();
        fn moved(
            e: &Expression,
            places: &HashMap<usize, (i64, i64)>,
            add: &dyn Fn(i64) -> Expression,
        ) -> Expression {
            match places.get(&(Arc::as_ptr(&e.0) as usize)) {
                Some((_, delta)) => Expression::binary(Operation::Add, e.clone(), add(*delta)),
                None => e.map_children(&mut |child| moved(child, places, add)),
            }
        }
        let mut refused = None;
        let body = self.step.map(&mut |stmt| {
            let stmt = Statement::map_expressions(stmt, &mut |e| moved(e, &places, &add));
            let Statement::Loop(l) = &*stmt else {
                return Some(vec![stmt]);
            };
            let Some((extent, delta)) = places.get(&(Arc::as_ptr(&l.axis) as usize)) else {
                return Some(vec![stmt]);
            };
            let sized = |extent: i64| {
                Arc::new(Axis {
                    name: l.axis.name,
                    extent: Dim::static_(extent),
                    window: l.axis.window.clone(),
                })
            };
            let grown = match by {
                By::Step(step) => Loop {
                    axis: sized(extent + delta * step),
                    ..l.clone()
                },
                By::Axis(_) => {
                    if !l.body.writes().is_empty() || !l.body.carries().is_empty() {
                        refused =
                            Some("a loop whose extent grows with the step stores".to_string());
                    }
                    let last = extent + delta * (steps as i64 - 1);
                    let end =
                        Expression::binary(Operation::Add, Expression::int(*extent), add(*delta));
                    Loop {
                        axis: sized((*extent).max(last)),
                        range: Some((Expression::int(0), end)),
                        ..l.clone()
                    }
                }
            };
            Some(vec![Arc::new(Statement::Loop(grown))])
        });
        match refused {
            Some(why) => Err(why),
            None => Ok(body),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_body;
    use crate::text::{body_text, parse_body};

    fn step(text: &str) -> Body {
        parse_body(text).unwrap()
    }

    fn read_at(offset: i64, bound: i64) -> Body {
        // The first copy reads at no offset at all, as the normal form spells it.
        let offset = if offset == 0 {
            String::new()
        } else {
            format!(" + {offset}")
        };
        step(&format!(
            "for a0 in 0..4:\n  for a1 in 0..8:\n    in0 = load x[a0, a1{offset}]\n    acc0 <- add(acc0, in0) over a1\n  \
             in1 = load s[a0]\n  v0 = select:\n    acc0 when a0 <= {bound}\n    in1 when true\n  out[a0] = v0"
        ))
    }

    fn grows(extent: i64) -> Body {
        step(&format!(
            "for a0 in 0..4:\n  for a1 in 0..{extent}:\n    in0 = load x[a0, a1]\n    acc0 <- add(acc0, in0) over a1\n  out[a0] = acc0"
        ))
    }

    fn normal(body: &Body) -> String {
        body_text(&normalize_body(body).unwrap()).unwrap()
    }

    #[test]
    fn copies_that_read_at_successive_offsets_inside_a_reduction_are_one_step() {
        let copies = [read_at(0, 1), read_at(8, 2), read_at(16, 3)];
        let proved = prove(&copies).unwrap();
        assert_eq!((proved.steps, &proved.stopped), (3, &None));
        let mut moving = proved.moving();
        moving.sort_unstable();
        assert_eq!(moving, [(0, 8), (1, 1)]);
        // The step at a number is that copy, and under an axis the offsets read the step.
        for (index, copy) in copies.iter().enumerate() {
            assert_eq!(
                normal(&proved.at(&By::Step(index as i64), 3).unwrap()),
                normal(copy)
            );
        }
        let walked = normal(&proved.at(&By::Axis("step".into()), 3).unwrap());
        assert!(
            walked.contains("8 * `step`") && walked.contains("`step` + 1"),
            "{walked}"
        );
    }

    #[test]
    fn a_reduction_that_grows_by_one_element_a_step_walks_a_range_of_the_step() {
        let proved = prove(&[grows(2), grows(3), grows(4)]).unwrap();
        assert_eq!(proved.moving(), [(2, 1)]);
        assert_eq!(
            normal(&proved.at(&By::Step(2), 3).unwrap()),
            normal(&grows(4))
        );
        let walked = body_text(&proved.at(&By::Axis("step".into()), 3).unwrap()).unwrap();
        assert!(walked.contains("of 4"), "{walked}");
    }

    #[test]
    fn the_proof_stops_at_the_first_body_that_is_no_such_step_and_names_why() {
        let proved = prove(&[read_at(0, 1), read_at(8, 2), read_at(17, 3)]).unwrap();
        assert_eq!(proved.steps, 2);
        let why = proved.stopped.unwrap();
        assert!(
            why.contains("is 17, where 0 moving 8 a step gives 16"),
            "{why}"
        );
        let other = step("for a0 in 0..4:\n  in0 = load x[a0, 0]\n  out[a0] = in0");
        assert!(
            prove(&[read_at(0, 1), other])
                .unwrap_err()
                .contains("another body")
        );
    }

    #[test]
    fn an_integer_is_paired_by_where_its_statement_nests() {
        // Two loops and two stores. The extents trade places between the copies: each goes with the loop that holds
        // the same store, whichever loop is written first.
        let nest = |first: i64, second: i64, flipped: bool| {
            let p = format!("for a0 in 0..{first}:\n  in0 = load x[a0]\n  p[a0] = in0");
            let q = format!("for a1 in 0..{second}:\n  in1 = load y[a1]\n  q[a1] = in1");
            step(&if flipped {
                format!("{q}\n{p}")
            } else {
                format!("{p}\n{q}")
            })
        };
        let copies = [nest(4, 8, false), nest(5, 7, true), nest(6, 6, false)];
        let proved = prove(&copies).unwrap();
        assert_eq!((proved.steps, &proved.stopped), (3, &None));
        let mut moving = proved.moving();
        moving.sort_unstable();
        assert_eq!(moving, [(4, 1), (8, -1)]);
        // A store that computes another function under the same loops is another body.
        let other = step(
            "for a0 in 0..4:\n  in0 = load y[a0]\n  p[a0] = in0\nfor a1 in 0..8:\n  in1 = load x[a1]\n  q[a1] = in1",
        );
        assert!(prove(&[copies[0].clone(), other]).is_err());
    }

    #[test]
    fn two_reads_that_key_alike_are_paired_however_they_are_written() {
        // One read stays and one moves; the copies list them in the order of their offsets, which flips.
        let pair = |a: i64, b: i64| {
            step(&format!(
                "for a0 in 0..4:\n  in0 = load x[a0 + {a}]\n  in1 = load x[a0 + {b}]\n  v0 = in0 * in1\n  out[a0] = v0"
            ))
        };
        let proved = prove(&[pair(1, 4), pair(4, 7), pair(4, 10)]);
        // Either the proof holds with every copy rebuilt exactly, or it stops: it never pairs wrongly.
        if let Ok(proved) = proved {
            for (index, copy) in [pair(1, 4), pair(4, 7), pair(4, 10)]
                .iter()
                .enumerate()
                .take(proved.steps)
            {
                assert_eq!(
                    normal(&proved.at(&By::Step(index as i64), 3).unwrap()),
                    normal(copy)
                );
            }
        }
    }

    #[test]
    fn the_default_numbering_takes_no_integer_apart() {
        let body = read_at(8, 2);
        let plain = super::super::value_numbers(&body, &spelled_key, Chains::Flat);
        assert!(plain.sites.is_empty());
        let none = plain
            .numbers
            .iter()
            .flatten()
            .all(|(_, params)| params.iter().all(|p| super::super::offset_of(p).is_none()));
        assert!(none);
    }
}
