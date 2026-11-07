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
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::*;
    use crate::body::{Body, Subroutine};
    use crate::normalize::testing::strs;
    use crate::statement::{Call, Kind};

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
        // The other Kernel IR kinds, and the two with the most fields at their fullest.
        "Sync",
        "CpAsyncCopy(smem=s, smem_index=(i, j), src=a, src_index=(i, j), lane_index=(i, j), valid=i < 2)",
        "CpAsyncCommit",
        "CpAsyncWait(group=1)",
        "TmaDescriptor(name=d, src_buf=a, src_shape=(64, 64), box_extents=(16, 16))",
        "TmaLoad(smem=s, smem_index=(i, j), desc=d, coords=(i, j), mbar=mb, mbar_slot=p)",
        "MbarrierInit(mbar=mb, slot=p)",
        "MbarrierArriveExpectTx(mbar=mb, bytes_=64, slot=p)",
        "MbarrierArrive(mbar=mb, slot=p)",
        "MbarrierWait(mbar=mb, phase=ph, slot=p)",
        "WarpShuffle(state=(acc), state_b=(acc__o), length=8):\n    combine_states:\n        t = multiply(acc, acc__o)\n        acc = add(acc, t)",
        "WarpBroadcast(state=(acc), length=8)",
        "FragmentApply(out=o, op=where, args=(FragmentArg(kind=coord, expressions=(__fcol > __frow)), FragmentArg(kind=uniform, names=(u)), FragmentArg(kind=row, names=(r0, r1)), FragmentArg(kind=frag, names=(f)), FragmentArg(kind=gmem, expressions=(i, j), buffer=mask)), row_base=q, col_base=kv)",
        "FragmentRowReduce(top=t0, bot=b0, frags=(f0, f1), op=maximum)",
        "RegFragment(name=f, role=c, shape=(16, 8, 16), dtype=f32)",
        "BlockScaleLoad(frag=sf, src_buffer=s, src_index=(i, j), role=a)",
        "LdmatrixLoad(frag=f, src_buffer=s, src_index=(i, j), role=b, gmem_guard=(i, j), k_zero=(i, j), pair_frag=g, scale_buffer=sc, scale_index=(i, j))",
        "MmaSyncPtx(c_frag=c, a_frag=a, b_frag=b, shape=(16, 8, 16), sfa_frag=sa, sfb_frag=sb)",
        "FragmentPromote(dst=c, src=h)",
        "FragmentRepack(frag=a, srcs=(c0, c1))",
        "WgmmaDescriptor(name=bD, smem=s, smem_index=i, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)",
        "WgmmaMma(c_frags=(c0, c1, c2, c3, c4, c5, c6, c7), b_desc=bD, shape=(64, 64, 16), a_desc=aD)",
        "WgmmaFence",
        "WgmmaCommit",
        "WgmmaWait",
        "RegStore(dst_buffer=out, dst_index=(i, j), frag=c, shape=(16, 8, 16), epilogue_params=(acc), epilogue_result=y, extra_frags=(c2), m_guard=(i, j), n_guard=(i, j)):\n    epilogue:\n        r = load res[i, j]\n        y = add(acc, r)\n    run:\n        RegStore(dst_buffer=out, dst_index=(i, j), frag=d, shape=(16, 8, 16))",
    ];

    const CALLS: &str = "sub inner(r in 0..4) -> s:\n    for k in 0..8:\n        x = load a[r, k]\n        s <- add(s, x) over k\nsub outer(r in 0..4, c in 0..2 window(parent=cc in 0..4)) -> t:\n    u = call inner[a](r)\n    w = load b[c]\n    t = multiply(u, w)\nfor i in 0..4:\n    p = call outer[a, b](i, 0)\n    q = call outer[a, b](i, 1)\n    z = call outer[m, b](i, 1)\n    v = p + q + z\n    y[i] = v";

    /// The subroutine each call of `body` names, in order.
    fn targets(body: &Body) -> Vec<Arc<Subroutine>> {
        body.of_kinds(&[Kind::Call])
            .iter()
            .map(|here| <&Call>::try_from(&**here.stmt()).unwrap().target.0.clone())
            .collect()
    }

    /// The message the reader refuses `text` with.
    fn refusal(text: &str) -> String {
        parse_body(text)
            .err()
            .unwrap_or_else(|| panic!("{text} parsed"))
            .to_string()
    }

    #[test]
    fn every_statement_reads_back() {
        for text in STATEMENTS {
            let body = parse_body(text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(body_text(&body).unwrap(), *text);
            assert_eq!(parse_body(&body_text(&body).unwrap()).unwrap(), body);
        }
    }

    /// No kind goes without a text: the statements above, with the calls, hold one of every kind, in every layer.
    #[test]
    fn every_kind_has_a_text() {
        let covered: BTreeSet<Kind> = STATEMENTS
            .iter()
            .chain([&CALLS])
            .flat_map(|text| parse_body(text).unwrap().iter_stmts())
            .map(|stmt| stmt.kind())
            .collect();
        assert_eq!(covered, Kind::ALL.iter().copied().collect());
    }

    #[test]
    fn a_body_is_its_statements_one_under_another() {
        let text = "let f32 v0 = 0.125\nfor a0 in 0..4:\n    for a1 in 0..8:\n        in0 = load x[a0, a1]\n        v1 = 2*in0 + v0\n        acc0 <- add(acc0, v1) over a1\n    y[a0] = acc0";
        let body = parse_body(text).unwrap();
        assert_eq!(body_text(&body).unwrap(), text);
        let kinds: Vec<Kind> = body.iter_stmts().iter().map(|stmt| stmt.kind()).collect();
        assert_eq!(
            kinds,
            [
                Kind::Let,
                Kind::Loop,
                Kind::Loop,
                Kind::Load,
                Kind::Affine,
                Kind::Accum,
                Kind::Write
            ]
        );
        assert_eq!(parse_body("").unwrap(), Body::empty());
        assert_eq!(body_text(&Body::empty()).unwrap(), "");
    }

    #[test]
    fn the_reader_takes_any_consistent_indentation_and_blank_lines() {
        assert_eq!(
            parse_body("\n  for a0 in 0..4:\n\n      y[a0] = v\n").unwrap(),
            parse_body("for a0 in 0..4:\n    y[a0] = v").unwrap()
        );
    }

    #[test]
    fn errors_name_their_line() {
        for (text, line, message) in [
            ("y[a0] = v\n    z[a0] = v", 2, "unexpected indentation"),
            (
                "for a0 in 0..4:\n        y[a0] = v\n    z[a0] = v",
                3,
                "unexpected indentation",
            ),
            ("v = 1", 1, "no term"),
            ("y[a0] = v\nv1 = multiply(a, b", 2, "expected"),
            ("v = select:\n    x where a0 < 1", 2, "expected 'when'"),
            ("for a0 in 2..4:", 1, "names its step"),
            ("else:", 1, "expected"),
            (
                "a, b = multiply(x, y)",
                1,
                "only a load binds several names",
            ),
            ("v = x + ", 1, "unexpected end"),
            ("y[a0] = v extra", 1, "unexpected text"),
        ] {
            let err = refusal(text);
            assert!(
                err.starts_with(&format!("line {line}: ")) && err.contains(message),
                "{text}: {err}"
            );
        }
    }

    /// Subroutines compare by object, so reading a text back keeps which calls share one, and the text itself.
    #[test]
    fn a_call_names_a_subroutine_defined_once_ahead_of_it() {
        let body = parse_body(CALLS).unwrap();
        assert_eq!(body_text(&body).unwrap(), CALLS);
        let [p, q, z] = <[_; 3]>::try_from(targets(&body)).unwrap();
        assert!(Arc::ptr_eq(&p, &q) && !Arc::ptr_eq(&p, &z));
        assert_eq!((&*p.name, &*z.name), ("outer", "outer"));
        assert_eq!(
            (
                strs(&p.buffers),
                strs(&z.buffers),
                strs(&p.params()),
                &*p.result
            ),
            (vec!["a", "b"], vec!["m", "b"], vec!["r", "c"], "t")
        );
        let window = p.axes[1].window.as_ref().unwrap();
        assert_eq!(&*window.parent.as_ref().unwrap().name, "cc");
        assert_eq!(
            body_text(&parse_body(&body_text(&body).unwrap()).unwrap()).unwrap(),
            CALLS
        );
    }

    #[test]
    fn a_call_that_does_not_fit_its_subroutine_is_refused() {
        for (text, line, message) in [
            (
                "p = call missing[a](i)",
                1,
                "no subroutine 'missing' is defined above",
            ),
            (
                "sub f(r in 0..4) -> s:\n    x = load a[r]\np = call f[a](0)",
                1,
                "must define its result",
            ),
            (
                "sub f(r in 0..4) -> x:\n    x = load a[r]\np = call f[a](0, 1)",
                3,
                "do not match its parameters",
            ),
        ] {
            let err = refusal(text);
            assert!(
                err.starts_with(&format!("line {line}: ")) && err.contains(message),
                "{text}: {err}"
            );
        }
    }

    #[test]
    fn two_subroutines_of_one_name_have_no_text() {
        // Each call is read from a text of its own, so the two definitions named `f` are two objects.
        let call = |text: &str| parse_statement(text).unwrap();
        let body = Body::new(vec![
            call("sub f(r in 0..4) -> y:\n    x = load a[r]\n    y = exp(x)\np = call f[a](a0)"),
            call("sub f(r in 0..4) -> y:\n    x = load b[r]\n    y = log(x)\nq = call f[b](a0)"),
        ]);
        let err = body_text(&body).unwrap_err().to_string();
        assert!(err.starts_with("two subroutines are named 'f'"), "{err}");
    }

    #[test]
    fn a_compact_body_prints_a_shared_definition_once_and_reads_back() {
        // Two calls of one subroutine, the second four columns over; the caller's `k` shadows the definition's.
        let compact = parse_body(
            "
            sub project(row in 0..3, col in 0..8) -> sum:
                for k in 0..16:
                    xv = load x[row, k]
                    wv = load weight[col, k]
                    product = multiply(xv, wv)
                    sum <- add(sum, product) over k
            for k in 0..3:
                for c in 0..8:
                    left = call project[x, weight](k, c)
                    right = call project[x, weight](k, (c + 4) % 8)
                    value = add(left, right)
                    out[k, c] = value
            ",
        )
        .unwrap();
        let rendered = body_text(&compact).unwrap();
        assert_eq!(
            rendered
                .matches("sub project(row in 0..3, col in 0..8) -> sum:")
                .count(),
            1
        );
        assert!(rendered.contains("left = call project[x, weight](k, c)"));
        assert!(rendered.contains("right = call project[x, weight](k,"));
        assert!(rendered.lines().count() < 18);
        let back = parse_body(&rendered).unwrap();
        assert_eq!(body_text(&back).unwrap(), rendered);
        let [left, right] = <[_; 2]>::try_from(targets(&back)).unwrap();
        assert!(Arc::ptr_eq(&left, &right));
    }
}
