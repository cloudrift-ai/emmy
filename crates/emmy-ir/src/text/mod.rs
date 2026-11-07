//! The text of a body: one printer and one reader, exact inverses. One statement is one line; a block's statements
//! are indented four spaces under its header; a field at its default is left out, nothing else is. The shared kinds
//! have clauses of their own; a kind of another layer prints and reads through its declaration (`Kind(field=value,
//! …):` over its blocks). A subroutine a call names is defined once, ahead of the statements.

pub mod parse;
pub mod print;

pub use parse::*;
pub use print::*;

pub const INDENT: &str = "    ";

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kinds::{Kind, Statement};

    const STATEMENTS: &[&str] = &[
        "in0 = load x[a0, a1 + 1]",
        "f8e4m3 in0, in1 = load x[a0]",
        "in0 = load x[a0] carried=unpack",
        "in0 = load `model.layers.0.weight`[a0]",
        "p = pre state[a0, 0]",
        "v1 = multiply(in0, in1)",
        "f16 v1 = exp(in0)",
        "`load` = copy(`for`)",
        "v2 = x",
        "v2 = -x + 0.5*y",
        "f32 v2 = 2*x - y + 1e-06",
        "v2 = 3*x - 4",
        "let v0 = 0.125",
        "let f32 i = a0 * 4 + a1",
        "init f32 acc = -inf",
        "acc0 <- add(acc0, v1)",
        "f32 acc0 <- maximum(acc0, v1) over a1, a2",
        "acc0 <- add(scaled, v1)",
        "state[a0, a1] <- v seed 0",
        "f32 state[a0] <- v seed state_in",
        "y[a0, a1] = v",
        "f16 y[a0] = (v0, v1)",
        "y[a0] += v",
        "slab[a0] = v swizzle=B32",
        "zero acc words=1024",
        "v3 = select:\n    x when a0 < 1\n    y when a0 >= 1",
        "for a0 in 0..4:\n    y[a0] = v",
        "for a0 in 0..seq~128 unroll=true seed=false:",
        "for a0 in 0..seq * 2~64:",
        "for kb in 0..16 window(parent=ko in 0..32 window(parent=k in 0..64), partition=true):",
        "for kb in 0..16 window(base=a0, bound=a0 + 1):",
        "for kb in 0..16 window():",
        "for k in t..64 step 32:\n    y[k] = v",
        "for k in 0..64 step 8 until k_end unroll=true:",
        "if a0 < 1:\n    y[a0] = v\nelse:\n    y[a0] = w",
        "if a0 < 1:",
        "SetMaxNReg(count=232, direction=inc)",
        "Smem(name=s_x, extents=(64, 64), dtype=f16)",
        "Tile(axes=(a0 in 0..64, a1 in 0..64), block_threads=128):\n    RegStore(dst_buffer=y, dst_index=(a0, a1), frag=c0, shape=(16, 8, 8)):\n        epilogue:\n            v1 = multiply(v0, s)\n        run:\n            RegStore(dst_buffer=y, dst_index=(a0, a1 + 8), frag=c1, shape=(16, 8, 8))",
        "FragmentApply(out=o, op=where, args=(FragmentArg(kind=coord, expressions=(_vc > _vr)), FragmentArg(kind=gmem, expressions=(i, j), buffer=mask)), row_base=q)",
        "TreeHalve(bufs=(b), state=(acc), state_b=(acc__o), length=8, tid_var=t):\n    combine_states:\n        t = multiply(acc, acc__o)\n        acc = add(acc, t)",
    ];

    const CALLS: &str = "sub inner(r in 0..4) -> s:\n    for k in 0..8:\n        x = load a[r, k]\n        s <- add(s, x) over k\nsub outer(r in 0..4, c in 0..2 window(parent=cc in 0..4)) -> t:\n    u = call inner[a](r)\n    w = load b[c]\n    t = multiply(u, w)\nfor i in 0..4:\n    p = call outer[a, b](i, 0)\n    q = call outer[a, b](i, 1)\n    z = call outer[m, b](i, 1)\n    v = p + q + z\n    y[i] = v";

    #[test]
    fn every_statement_reads_back() {
        for text in STATEMENTS {
            let body = parse_body(text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(body_text(&body).unwrap(), *text);
            assert_eq!(parse_body(&body_text(&body).unwrap()).unwrap(), body);
        }
    }

    #[test]
    fn calls_read_back() {
        let body = parse_body(CALLS).unwrap();
        assert_eq!(body_text(&body).unwrap(), CALLS);
        let calls = body.of_kinds(&[Kind::Call]);
        let targets: Vec<_> = calls
            .iter()
            .map(|c| match &**c.stmt() {
                Statement::Call(call) => call.target.0.clone(),
                _ => unreachable!(),
            })
            .collect();
        assert!(Arc::ptr_eq(&targets[0], &targets[1]) && !Arc::ptr_eq(&targets[0], &targets[2]));
    }

    #[test]
    fn errors_name_their_line() {
        for (text, message) in [
            ("y[a0] = v\n    z[a0] = v", "line 2: unexpected indentation"),
            ("v = 1", "line 1: "),
            ("for a0 in 2..4:", "line 1: "),
            (
                "a, b = multiply(x, y)",
                "line 1: only a load binds several names",
            ),
            ("v = x + ", "line 1: unexpected end"),
            ("y[a0] = v extra", "line 1: unexpected text"),
            (
                "p = call missing[a](i)",
                "line 1: no subroutine 'missing' is defined above",
            ),
        ] {
            let err = parse_body(text)
                .err()
                .unwrap_or_else(|| panic!("{text} parsed"));
            assert!(err.to_string().starts_with(message), "{text}: {err}");
        }
    }
}
