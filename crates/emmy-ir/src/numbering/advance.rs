//! The proof that bodies are one step taken again and again.
//!
//! The bodies are numbered with the integer of every affine coordinate expression apart ([`value_numbers_apart`]).
//! They are one step where they hold the same statements and blocks, each computing the same function of its
//! parameters, and where every such integer moves by one amount from each body to the next. A statement is paired
//! with the statement of the next body by what it computes, never by where it stands in the text.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::{Chains, Key, Param, offset_of, spelled_key, value_numbers_apart};
use crate::body::Body;

/// What [`prove`] found: how many integers the first body's coordinate expressions add, and for each that moves,
/// the place it is written at in the first body, its value there and how far it moves per step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proved {
    pub sites: usize,
    pub moves: Vec<(i64, i64, i64)>,
}

/// A statement or a block as the proof compares it: what it computes, and its parameters.
type Instance = (Key, Vec<Param>);

/// The coordinate forms among `params`, an integer taken apart standing as nothing: what two statements that
/// compute one function are told apart by.
fn forms(params: &[Param]) -> Vec<Option<&Key>> {
    params
        .iter()
        .map(|param| offset_of(param).is_none().then(|| param.key()))
        .collect()
}

/// The first place an integer of `params` is written at.
fn first_site(params: &[Param]) -> Option<i64> {
    params
        .iter()
        .filter_map(|p| offset_of(p).map(|(_, site)| site))
        .min()
}

/// Every statement and block of `body`, in an order of what each computes: its number, then the forms of its
/// parameters, then, for two that tie on both, the order they are written in.
fn instances(body: &Body) -> Vec<Instance> {
    let numbering = value_numbers_apart(body, &spelled_key, Chains::Flat);
    let mut out: Vec<Instance> = Vec::new();
    for (leaf, block) in numbering.numbers.iter().zip(&numbering.blocks) {
        if let Some((number, params)) = leaf {
            out.push((Key::Num(*number), params.clone()));
        }
        if let Some((header, params)) = block {
            out.push((header.clone(), params.clone()));
        }
    }
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| forms(&a.1).cmp(&forms(&b.1)))
            .then_with(|| match (first_site(&a.1), first_site(&b.1)) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => Ordering::Equal,
            })
    });
    out
}

/// The proof that `bodies` are one step taken again and again, or the first difference that keeps them from it.
pub fn prove(bodies: &[Body]) -> std::result::Result<Proved, String> {
    if bodies.len() < 2 {
        return Err("one body is no step taken again".to_string());
    }
    let all: Vec<Vec<Instance>> = bodies.iter().map(instances).collect();
    let first = &all[0];
    // The value at every step of each integer the first body writes, by the place it is written at.
    let mut values: HashMap<i64, Vec<i64>> = HashMap::new();
    for (step, other) in all.iter().enumerate() {
        if other.len() != first.len() {
            return Err(format!(
                "step {step} holds {} statements and blocks where step 0 holds {}",
                other.len(),
                first.len()
            ));
        }
        let mut paired: HashMap<i64, i64> = HashMap::new();
        for (at, ((ours, our_params), (theirs, their_params))) in
            first.iter().zip(other).enumerate()
        {
            if ours != theirs || our_params.len() != their_params.len() {
                return Err(format!(
                    "step {step} computes another function at statement {at} of {}",
                    first.len()
                ));
            }
            for (a, b) in our_params.iter().zip(their_params) {
                match (offset_of(a), offset_of(b)) {
                    (Some((base, site)), Some((value, its))) => {
                        if *paired.entry(site).or_insert(its) != its {
                            return Err(format!(
                                "step {step} pairs one integer of step 0 with two of its own"
                            ));
                        }
                        let held = values.entry(site).or_insert_with(|| vec![base]);
                        if held.len() == step {
                            held.push(value);
                        }
                    }
                    (None, None) if a == b => {}
                    _ => {
                        return Err(format!(
                            "step {step} reads another coordinate form at statement {at} of {}",
                            first.len()
                        ));
                    }
                }
            }
        }
    }
    let mut sites: Vec<(i64, Vec<i64>)> = values.into_iter().collect();
    sites.sort_unstable();
    let mut moves = Vec::new();
    for (site, held) in &sites {
        let delta = held[1] - held[0];
        for (step, value) in held.iter().enumerate() {
            let expected = held[0] + delta * step as i64;
            if *value != expected {
                return Err(format!(
                    "an integer of step {step} is {value}, where {} moving {delta} a step gives {expected}",
                    held[0]
                ));
            }
        }
        if delta != 0 {
            moves.push((*site, held[0], delta));
        }
    }
    Ok(Proved {
        sites: sites.len(),
        moves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse_body;

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

    #[test]
    fn copies_that_read_at_successive_offsets_inside_a_reduction_are_one_step() {
        let proved = prove(&[read_at(0, 1), read_at(8, 2), read_at(16, 3)]).unwrap();
        let moved: Vec<(i64, i64)> = proved
            .moves
            .iter()
            .map(|(_, base, delta)| (*base, *delta))
            .collect();
        assert!(moved.contains(&(0, 8)), "{proved:?}");
        assert!(moved.contains(&(1, 1)), "{proved:?}");
    }

    #[test]
    fn a_reduction_that_grows_by_one_element_a_step_is_one_step() {
        let grows = |extent: i64| {
            step(&format!(
                "for a0 in 0..4:\n  for a1 in 0..{extent}:\n    in0 = load x[a0, a1]\n    acc0 <- add(acc0, in0) over a1\n  out[a0] = acc0"
            ))
        };
        let proved = prove(&[grows(2), grows(3), grows(4)]).unwrap();
        assert_eq!(
            proved
                .moves
                .iter()
                .map(|(_, base, delta)| (*base, *delta))
                .collect::<Vec<_>>(),
            [(2, 1)]
        );
    }

    #[test]
    fn an_integer_off_the_line_is_named() {
        let why = prove(&[read_at(0, 1), read_at(8, 2), read_at(17, 3)]).unwrap_err();
        assert!(
            why.contains("is 17, where 0 moving 8 a step gives 16"),
            "{why}"
        );
    }

    #[test]
    fn another_function_is_no_step() {
        let other = step("for a0 in 0..4:\n  in0 = load x[a0, 0]\n  out[a0] = in0");
        assert!(prove(&[read_at(0, 1), other]).is_err());
    }

    #[test]
    fn the_default_numbering_takes_no_integer_apart() {
        let body = read_at(8, 2);
        let plain = super::super::value_numbers(&body, &spelled_key, Chains::Flat);
        let none = plain
            .numbers
            .iter()
            .flatten()
            .all(|(_, params)| params.iter().all(|p| offset_of(p).is_none()));
        assert!(none);
    }
}
