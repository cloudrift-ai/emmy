//! One test per fragment rule and per slab layout: a fragment gathered by its row of the lane-to-element table,
//! stored through it, computed on and folded along its rows, and Volta's paired slabs; then the byte values (8-bit
//! floats, e2m1 codes and the pair table). Each counts the instructions the rule writes in the list, and ptxas
//! assembles every kernel at `-O0` and at its default level for its own target wherever it is installed.

mod common;

use emmy_ir::dtype::DType;
use emmy_ptx::inst::Module;
use emmy_ptx::{Interface, Param, ParamKind, Target};

use DType::{F16, F32};
use common::{EXACT, Params, listing, lowered_in};

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
        arch_specific: false,
        ..common::interface(params, 32)
    };
    let body = format!("{GRID}\n{}", common::indented(statements));
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
/// the operand's axis zeroes the fragment and reads only where some row is in range; a zero-fill of K guards
/// each element.
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
    // A fragment wholly past the bound reads nothing: one branch around the loads.
    assert_eq!(count(&clamped, "bra"), 1);
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

/// A store under a guard of rows keeps its pairs, each under its row's predicate; a guard of columns stores each
/// element under its own. An epilogue runs its statements once an element, at the element's coordinates, and the
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
    assert_eq!(count(&rows, "st.global.v2.f32"), 2);
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

/// Volta's paired slabs: a store of eight halves under the crosswise layout is two stores of four, and the paired
/// drain reads each element of its two fragments where the layout puts it.
#[test]
fn volta_s_paired_slab_is_stored_and_read_through_its_layout() {
    let module = kernel(
        "
        Smem(name=sa, extents=(64, 32), dtype=f16, align=16)
        RegFragment(name=fa0, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fa1, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        f16 x0, x1, x2, x3, x4, x5, x6, x7 = load a[r * 256 + t * 8]
        f16 sa[t / 4, 8 * (t % 4)] = (x0, x1, x2, x3, x4, x5, x6, x7) swizzle=V70A
        Sync
        FragmentLoad(frag=fa0, src_buffer=sa, src_index=(0, 4), role=a, fragment_layout=m8n8k4, ldm=32, staged=true, slab_layout=V70A, pair_frag=fa1, src_dtype=f16, dtype=f16)
        MmaSyncPtx(c_frag=fc, a_frag=fa0, b_frag=fa1, shape=(8, 8, 4))
        ",
        HALVES,
        80,
    )
    .unwrap();
    assert_eq!(
        count(&module, "st.shared.v4.b16"),
        2,
        "{:#?}",
        listing(&module)
    );
    assert_eq!(count(&module, "ld.shared.b16"), 8);
    assert_eq!(count(&module, "mma.sync.aligned.m8n8k4"), 1);
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
    let mut interface = common::interface(BYTES, 32);
    interface.params[1] = Param {
        pair_table: true,
        ..interface.params[1].clone()
    };
    let body = format!("{GRID}\n{}", common::indented(statements));
    let module = lowered_in(&body, &interface, Target { sm: 120 }, EXACT).unwrap();
    assert_eq!(count(&module, "cvt.rn.satfinite.e4m3x2.f32"), 1);
    assert_eq!(count(&module, "cvt.rn.f16x2.e4m3x2"), 1);
    assert_eq!(count(&module, "st.global.u8"), 2);
    assert_eq!(count(&module, "setp.g"), 7, "{:#?}", listing(&module));
    // The table is never read: the one global load is `x`'s.
    assert_eq!(count(&module, "ld.global"), 1);
    let refused = lowered_in(&body, &interface, Target { sm: 80 }, EXACT).unwrap_err();
    assert!(refused.to_string().contains("8-bit floats"), "{refused}");
}
