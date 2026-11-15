//! One test per fragment rule and per slab layout: a fragment gathered by its row of the lane-to-element table,
//! stored through it, computed on and folded along its rows, and Volta's paired slabs; then the byte values (8-bit
//! floats, e2m1 codes and the pair table). Each counts the instructions the rule writes in the list, and ptxas
//! assembles every kernel at `-O0` and at its default level for its own target wherever it is installed.

use crate::inst::Module;
use crate::{Interface, Param, ParamKind, Target};
use emmy_ir::dtype::DType;
use emmy_ir::expr::Expression;

use super::common::{EXACT, Params, listing, lowered_in};
use DType::{F16, F32};

const GRID: &str =
    "Grid(axes=(r in 0..4, t in 0..32), decode=(block_idx.x, thread_idx.x), block_threads=32):";

const HALVES: Params = &[
    ("a", ParamKind::Input, Some(F16)),
    ("b", ParamKind::Input, Some(F16)),
    ("w", ParamKind::Input, Some(F32)),
    ("o", ParamKind::Output, Some(F32)),
    ("h", ParamKind::Output, Some(F16)),
];

const FRAGMENTS: &str = "
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fb, role=b, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
";

fn kernel(statements: &str, params: Params, sm: u32) -> emmy_ir::Result<Module> {
    let interface = Interface {
        ..super::common::interface(params)
    };
    let body = format!("{GRID}\n{}", super::common::indented(statements));
    lowered_in(&body, &interface, Target { sm }, EXACT)
}

/// How many instructions of `module` start with `prefix`, past the guard they run under.
fn count(module: &Module, prefix: &str) -> usize {
    listing(module)
        .iter()
        .map(|line| match line.strip_prefix('@') {
            Some(guarded) => guarded.split_once(' ').map_or("", |(_, inst)| inst),
            None => line,
        })
        .filter(|line| line.starts_with(prefix))
        .count()
}

/// How many loads of `module` run under a guard.
fn guarded_loads(module: &Module) -> usize {
    listing(module)
        .iter()
        .filter(|line| line.starts_with('@') && line.contains(" ld."))
        .count()
}

/// A fragment gathered from global memory reads the elements its row of the table gives each lane. A's register
/// holds two elements side by side along K, aligned by the even column and row stride: one 32-bit load each. B
/// with K along its rows strides its two elements a row apart: a load an element, joined by one move. A clamp of
/// the operand's axis holds each row to the last in range, and a fragment wholly past it reads nothing: its
/// loads are guarded, with a zero-fill of K's guard of each element where there is one.
#[test]
fn a_fragment_is_gathered_by_its_row_of_the_table() {
    let module = kernel(
        &format!(
            "{FRAGMENTS}
        FragmentLoad(frag=fa, src_buffer=a, src_index=(r * 512), role=a, ldm=32, src_dtype=f16, dtype=f16)
        FragmentLoad(frag=fb, src_buffer=b, src_index=(r * 256), role=b, ldm=8, src_dtype=f16, dtype=f16)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8)
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&module, "ld.global.nc.b32"), 4);
    assert_eq!(count(&module, "ld.global.nc.b16"), 4);
    assert_eq!(count(&module, "mov.b32 %r"), 2, "{:#?}", listing(&module));
    // Two rows of two floats side by side: one store of a pair each.
    assert_eq!(count(&module, "st.global.v2.f32"), 2);

    let clamped = kernel(
        &format!(
            "{FRAGMENTS}
        FragmentLoad(frag=fa, src_buffer=a, src_index=(r * 512), role=a, ldm=27, clamp=(0, 5), k_zero=(0, 11), src_dtype=f16, dtype=f16)
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    // Every element under its own K guard: a load each, none of them paired.
    assert_eq!(
        count(&clamped, "ld.global.nc.b16"),
        8,
        "{:#?}",
        listing(&clamped)
    );
    assert_eq!(guarded_loads(&clamped), 8);
    // A fragment wholly past the bound reads nothing, by the guards and not by a branch.
    assert_eq!(count(&clamped, "bra"), 0);
    assert!(count(&clamped, "min.s32") >= 1);
}

/// An f32 source gathered into a fragment of halves rounds each element once.
#[test]
fn a_fragment_of_halves_from_floats_rounds_each_element() {
    let module = kernel(
        &format!(
            "{FRAGMENTS}
        FragmentLoad(frag=fa, src_buffer=w, src_index=(r * 512), role=a, ldm=32, src_dtype=f32, dtype=f16)
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&module, "ld.global.nc.f32"), 8);
    assert_eq!(count(&module, "cvt.rn.f16.f32"), 8);
}

/// A guard of Volta's interleaved accumulator is decided by the rows the lane table gives each register over the
/// 32 lanes: a 16-row tile at row 16 of an 8-row output stores nothing and tests nothing; at row 0 a register's
/// rows fall on both sides of the bound over the lanes, so each store keeps its guard.
#[test]
fn a_volta_store_past_its_rows_is_not_written() {
    let store = |row: i64| {
        kernel(
            &format!(
                "
        RegFragment(name=fv, role=c, dtype=f32, fragment_layout=m8n8k4)
        RegStore(dst_buffer=o, dst_index=({row} * 16), frag=fv, shape=(16, 16, 4), ldm=16, ldn=1, m_guard=({row}, 8), fragment_layout=m8n8k4, volta_interleaved=true)
        "
            ),
            HALVES,
            80,
        )
        .unwrap()
    };
    let past = store(16);
    assert_eq!(count(&past, "st.global"), 0, "{:#?}", listing(&past));
    assert_eq!(count(&past, "setp"), 0);
    let first = store(0);
    assert_eq!(count(&first, "st.global"), 4, "{:#?}", listing(&first));
}

/// Two e4m3 elements side by side along K are one 16-bit load and one packed conversion into halves, for a
/// fragment of halves and of bfloat16 alike: every e4m3 value is a bfloat16, so the halves widened and packed
/// again are exact.
#[test]
fn two_adjacent_e4m3_elements_are_one_load_for_halves_and_bfloat16() {
    let params: Params = &[
        ("q", ParamKind::Input, Some(DType::F8E4M3)),
        ("o", ParamKind::Output, Some(F32)),
    ];
    for (dtype, packs) in [("f16", 0), ("bf16", 2)] {
        let module = kernel(
            &format!(
                "
        RegFragment(name=fb, role=b, dtype={dtype})
        FragmentLoad(frag=fb, src_buffer=q, src_index=(r * 256), role=b, ldm=16, b_trans=true, src_dtype=f8e4m3, dtype={dtype})
        "
            ),
            params,
            120,
        )
        .unwrap();
        assert_eq!(
            count(&module, "ld.global.nc.b16"),
            2,
            "{:#?}",
            listing(&module)
        );
        assert_eq!(count(&module, "ld.global.nc.u8"), 0);
        assert_eq!(count(&module, "cvt.rn.f16x2.e4m3x2"), 2);
        assert_eq!(count(&module, "cvt.rn.bf16x2.f32"), packs);
    }
}

/// A store under a guard of rows keeps its pairs, each under its row's predicate, and a guard the ranges decide is
/// no branch; a guard of columns stores each element under its own. An epilogue runs its statements once an element, at the element's coordinates, and the
/// value is rounded once into a buffer of halves.
#[test]
fn a_store_is_guarded_and_runs_its_epilogue_once_an_element() {
    let rows = kernel(
        &format!(
            "{FRAGMENTS}
        RegStore(dst_buffer=o, dst_index=(r * 40), frag=fc, shape=(16, 8, 16), ldm=8, m_guard=(0, 5))
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    // The lane's second row (8 and on) is never below 5: only the first is stored, under its predicate.
    assert_eq!(count(&rows, "st.global.v2.f32"), 1);
    assert_eq!(count(&rows, "bra"), 1);
    // A guard the ranges decide is no branch: the lane's first row is always in, its second always out.
    let decided = kernel(
        &format!(
            "{FRAGMENTS}
        RegStore(dst_buffer=o, dst_index=(r * 40), frag=fc, shape=(16, 8, 16), ldm=8, m_guard=(0, 8))
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(
        count(&decided, "st.global.v2.f32"),
        1,
        "{:#?}",
        listing(&decided)
    );
    assert_eq!(count(&decided, "bra"), 0);
    let columns = kernel(
        &format!(
            "{FRAGMENTS}
        RegStore(dst_buffer=o, dst_index=(r * 40), frag=fc, shape=(16, 8, 16), ldm=8, n_guard=(0, 6))
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&columns, "st.global.f32"), 4);
    let epilogue = kernel(
        &format!(
            "{FRAGMENTS}
        RegStore(dst_buffer=h, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8, epilogue_params=(acc), epilogue_result=v):
            epilogue:
                f32 bias = load w[__N__]
                f32 v = acc + bias
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&epilogue, "ld.global.nc.f32"), 4);
    assert_eq!(count(&epilogue, "add.rn.f32"), 4);
    assert_eq!(count(&epilogue, "cvt.rn.f16.f32"), 4);
    assert_eq!(count(&epilogue, "st.global.v2.b16"), 2);
}

/// An operation per element computes once a register, at the element's own row and column; the fold of a row
/// combines the lane's elements of it, then the lanes that share the row by the table's two shuffles.
#[test]
fn an_operation_per_element_and_a_fold_along_rows_stay_in_registers() {
    let module = kernel(
        &format!(
            "{FRAGMENTS}
        FragmentApply(out=fh, op=where, args=(FragmentArg(kind=coord, expressions=(__frow < __fcol)), FragmentArg(kind=uniform, expressions=(-1e+30)), FragmentArg(kind=frag, names=(fc))), row_base=r * 16, col_base=0)
        FragmentRowReduce(top=top, bot=bot, frags=(fh), op=maximum)
        FragmentApply(out=fd, op=subtract, args=(FragmentArg(kind=frag, names=(fh)), FragmentArg(kind=row, names=(top, bot))))
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fd, shape=(16, 8, 16), ldm=8)
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&module, "selp.f32"), 4);
    assert_eq!(count(&module, "shfl.sync.bfly.b32"), 4);
    // Two elements of each row in the lane, then two shuffles of each of the two rows.
    assert_eq!(count(&module, "max.f32"), 2 + 4);
    assert_eq!(count(&module, "sub.rn.f32"), 4);
}

/// A store whose fragment rows run along the output's columns (`row_dim=1, col_dim=0`) steps an element's row by one
/// and its column by the output's row stride: no two of a lane's elements are neighbours, so each is stored alone.
#[test]
fn a_transposed_store_steps_by_both_strides_of_its_output() {
    let mut interface = Interface {
        ..super::common::interface(HALVES)
    };
    interface.params[3].shape = vec![Expression::int(16), Expression::int(64)];
    let statements = format!(
        "{FRAGMENTS}
        RegStore(dst_buffer=o, dst_index=(0, r * 16), frag=fc, shape=(16, 8, 16), row_dim=1, col_dim=0)
        "
    );
    let body = format!("{GRID}\n{}", super::common::indented(&statements));
    let module = lowered_in(&body, &interface, Target { sm: 80 }, EXACT).unwrap();
    let listed = listing(&module);
    assert_eq!(count(&module, "st.global.f32"), 4, "{listed:#?}");
    assert_eq!(count(&module, "st.global.v2"), 0, "{listed:#?}");
    // The lane's two columns are a row of 64 elements apart, and its second row eight elements on.
    let offsets: Vec<&str> = listed
        .iter()
        .filter(|line| line.starts_with("st.global.f32"))
        .filter_map(|line| {
            line.split('+')
                .nth(1)
                .and_then(|rest| rest.split(']').next())
        })
        .collect();
    assert_eq!(offsets, ["256", "32", "288"], "{listed:#?}");
}

/// An operand gathered from global memory is read at each element's own row and column, and a half one is widened
/// to the accumulator's f32 before the operation.
#[test]
fn a_gathered_operand_is_read_at_each_element_and_widened() {
    let module = kernel(
        &format!(
            "{FRAGMENTS}
        FragmentApply(out=fc, op=add, args=(FragmentArg(kind=frag, names=(fc)), FragmentArg(kind=gmem, expressions=(__frow * 8 + __fcol), buffer=a)), in_place=true, row_base=r * 16, col_base=0)
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8)
        "
        ),
        HALVES,
        80,
    )
    .unwrap();
    let listed = listing(&module);
    assert_eq!(count(&module, "ld.global.nc.b16"), 4, "{listed:#?}");
    assert_eq!(count(&module, "cvt.f32.f16"), 4, "{listed:#?}");
    assert_eq!(count(&module, "add.rn.f32"), 4, "{listed:#?}");
}

/// A clamp of a gathered operand's index that the lane's own range keeps inside the bound folds away: each element
/// reads at its row and column with no comparison and no select. A clamp the range does not decide stays.
#[test]
fn a_gathered_clamp_the_lane_range_decides_is_no_select() {
    let gathered = |bound: i64| {
        let index = format!("(__frow < {bound} ? __frow : {}) * 8 + __fcol", bound - 1);
        kernel(
            &format!(
                "{FRAGMENTS}
        FragmentApply(out=fc, op=add, args=(FragmentArg(kind=frag, names=(fc)), FragmentArg(kind=gmem, expressions=({index}), buffer=a)), in_place=true, row_base=r * 16, col_base=0)
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8)
        "
            ),
            HALVES,
            80,
        )
        .unwrap()
    };
    // Rows `16r` to `16r + 15` with `r < 4`: every one below 64, some at or past 40.
    assert_eq!(count(&gathered(64), "selp"), 0);
    assert!(count(&gathered(40), "selp") > 0);
}

/// Volta's paired slabs: a store of eight halves under the crosswise layout is two stores of four, and the paired
/// drain reads each element of its two fragments where the layout puts it: where the slab position steps by the
/// layout's period, the eight elements are one aligned run, read at once.
#[test]
fn volta_s_paired_slab_is_stored_and_read_through_its_layout() {
    let module_at = |at: &str| {
        kernel(
            &format!(
                "
        Smem(name=sa, extents=(64, 32), dtype=f16, align=16)
        RegFragment(name=fa0, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fa1, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        f16 x0, x1, x2, x3, x4, x5, x6, x7 = load a[r * 256 + t * 8]
        f16 sa[t / 4, 8 * (t % 4)] = (x0, x1, x2, x3, x4, x5, x6, x7) swizzle=V70A
        Sync
        FragmentLoad(frag=fa0, src_buffer=sa, src_index=({at}), role=a, fragment_layout=m8n8k4, ldm=32, staged=true, slab_layout=V70A, pair_frag=fa1, src_dtype=f16, dtype=f16)
        MmaSyncPtx(c_frag=fc, a_frag=fa0, b_frag=fa1, shape=(8, 8, 4))
        "
            ),
            HALVES,
            80,
        )
        .unwrap()
    };
    let module = module_at("0, 4");
    assert_eq!(
        count(&module, "st.shared.v4.b16"),
        2,
        "{:#?}",
        listing(&module)
    );
    // The eight elements of the two cells are one aligned run of the slab: one 128-bit read. At a K group whose
    // second bit is set the second cell's run comes first, and it is still one read.
    for at in ["0, 4", "32, 8", "r * 32, 12"] {
        let module = module_at(at);
        assert_eq!(count(&module, "ld.shared.v4.b32"), 1, "{at}");
        assert_eq!(count(&module, "ld.shared.b16"), 0, "{at}");
        assert_eq!(count(&module, "mma.sync.aligned.m8n8k4"), 1);
    }
    // A slab position off the layout's period proves nothing: an element a load.
    let module = module_at("r * 8, 4");
    assert_eq!(count(&module, "ld.shared.v4.b32"), 0);
    assert_eq!(count(&module, "ld.shared.b16"), 8);
}

/// A run read in a loop that steps its slab position by whole periods moves its address by one sum a trip: the
/// layout's exchanges stand at the constant position, which only the lane decides, ahead of the loop.
#[test]
fn a_paired_read_in_a_loop_moves_its_address_by_a_sum() {
    let module = kernel(
        "
        Smem(name=sa, extents=(64, 32), dtype=f16, align=16)
        RegFragment(name=fa0, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fa1, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        for s in 0..4:
            FragmentLoad(frag=fa0, src_buffer=sa, src_index=(32 * (s % 2), 8), role=a, fragment_layout=m8n8k4, ldm=32, staged=true, slab_layout=V70A, pair_frag=fa1, src_dtype=f16, dtype=f16)
            MmaSyncPtx(c_frag=fc, a_frag=fa0, b_frag=fa1, shape=(8, 8, 4))
        ",
        HALVES,
        80,
    )
    .unwrap();
    let lines = listing(&module);
    let head = lines.iter().rposition(|line| line.ends_with(':')).unwrap();
    let body = &lines[head..];
    assert_eq!(count(&module, "ld.shared.v4.b32"), 1, "{lines:#?}");
    assert!(
        body.iter()
            .all(|line| !line.starts_with("xor") && !line.starts_with("shr")),
        "{body:#?}"
    );
}

/// A swizzled destination of an async copy or a store takes the exchange of 16-byte chunks over its flat index.
#[test]
fn a_swizzled_slab_exchanges_its_chunks() {
    let module = kernel(
        "
        Smem(name=sa, extents=(16, 64), dtype=f16, align=128)
        CpAsyncCopy(smem=sa, smem_index=(t * 8), src=a, src_index=(r * 256 + t * 8), nbytes=16, swizzle=B128)
        CpAsyncCommit
        CpAsyncWait(group=0)
        f16 x = load a[t]
        f16 sa[512 + t] = x swizzle=B64
        ",
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(count(&module, "xor.b32"), 2, "{:#?}", listing(&module));
    assert_eq!(count(&module, "cp.async.cg.shared.global"), 1);
}

const BYTES: Params = &[
    ("x", ParamKind::Input, Some(F32)),
    ("table", ParamKind::Input, Some(F16)),
    ("q", ParamKind::Output, Some(DType::F8E4M3)),
    ("p", ParamKind::Output, Some(DType::F4E2M1x2)),
    ("o", ParamKind::Output, Some(F32)),
];

/// An 8-bit float is its bits in a byte: a cast into it is the hardware's saturating conversion, a cast back goes
/// through the half of the same value, and a store keeps the byte. The e2m1 encode counts the midpoints below the
/// magnitude. A load of the pair table computes its value and reads nothing.
#[test]
fn byte_floats_convert_store_and_decode() {
    let statements = "
        f32 v = load x[t]
        f8e4m3 e = (f8e4m3)v
        f8e4m3 q[t] = e
        f32 back = (f32)e
        f32 code = to_f4e2m1(v)
        i32 c = (i32)code
        f4e2m1x2 pair = (f4e2m1x2)c
        f4e2m1x2 p[t] = pair
        f16 entry = load table[t]
        f32 wide = (f32)entry
        f32 sum = back + wide
        f32 o[t] = sum
    ";
    let mut interface = super::common::interface(BYTES);
    interface.params[1] = Param {
        pair_table: true,
        ..interface.params[1].clone()
    };
    let body = format!("{GRID}\n{}", super::common::indented(statements));
    let module = lowered_in(&body, &interface, Target { sm: 120 }, EXACT).unwrap();
    assert_eq!(count(&module, "cvt.rn.satfinite.e4m3x2.f32"), 1);
    assert_eq!(count(&module, "cvt.rn.f16x2.e4m3x2"), 1);
    assert_eq!(count(&module, "st.global.u8"), 2);
    assert_eq!(count(&module, "setp.g"), 7, "{:#?}", listing(&module));
    // The table is never read: the one global load is `x`'s.
    assert_eq!(count(&module, "ld.global"), 1);
    // Before the card has the conversion, the encode is computed from the bits of the f32.
    let older = lowered_in(&body, &interface, Target { sm: 80 }, EXACT).unwrap();
    assert_eq!(count(&older, "cvt.rn.satfinite.e4m3x2.f32"), 0);
    assert_eq!(count(&older, "st.global.u8"), 2);
}
