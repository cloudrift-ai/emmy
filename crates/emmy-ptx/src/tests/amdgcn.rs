//! The AMD printer: the same instruction list as LLVM IR. The rules are tested once, on the instruction list; here
//! are the module around the code, how the printer writes a register, a branch and a guard, and what it refuses.
//! Every kernel of the other rule tests is printed and assembled for `gfx942` too (`common::lowered_in`).

use emmy_ir::Error;
use emmy_ir::dtype::DType;

use super::common::{A_O, EXACT, FUSED, Params, amdgcn_ir, indented, interface};
use crate::ParamKind::{Input, Int, Output};
use crate::{Interface, Precision};

const GRID: &str =
    "Grid(axes=(i in 0..4096), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):";

fn ir(body: &str, params: Params, precision: Precision) -> emmy_ir::Result<String> {
    let body = format!("{GRID}\n{}", indented(body));
    amdgcn_ir(&body, &interface(params), precision)
}

/// The lines the printer wrote for the statement `statement`: from its comment to the next one.
fn lines<'a>(ir: &'a str, statement: &str) -> Vec<&'a str> {
    let comment = format!("; {statement}");
    ir.lines()
        .map(str::trim)
        .skip_while(|line| *line != comment)
        .skip(1)
        .take_while(|line| !line.starts_with(';') && *line != "ret void")
        .collect()
}

/// The module names its target, takes a buffer as a pointer of global memory with the launch's alignment and a
/// size as 32 bits, holds every register as a stack slot, and states the block's thread bound.
#[test]
fn a_module_is_one_kernel_entry_with_a_slot_for_every_register() {
    let params: Params = &[
        ("a", Input, Some(DType::F32)),
        ("o", Output, Some(DType::F32)),
        ("n", Int, None),
    ];
    let text = ir("f32 x = load a[i]\nf32 o[i] = x", params, EXACT).unwrap();
    assert!(
        text.starts_with("target triple = \"amdgcn-amd-amdhsa\"\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "define protected amdgpu_kernel void @k(ptr addrspace(1) noundef align 16 %a0, \
             ptr addrspace(1) noundef align 16 %a1, i32 noundef %a2) #0 {\nentry:\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("\t%rd0 = alloca i64, align 8, addrspace(5)\n"),
        "{text}"
    );
    assert!(
        text.contains("\t%f0 = alloca float, align 4, addrspace(5)\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "attributes #0 = { nounwind \"target-cpu\"=\"gfx942\" \"uniform-work-group-size\"=\"true\" \
             \"amdgpu-flat-work-group-size\"=\"1,256\" }"
        ),
        "{text}"
    );
    assert!(
        text.contains("declare i32 @llvm.amdgcn.workitem.id.x()\n"),
        "{text}"
    );
}

/// An instruction reads its operands from their slots and stores its result in its own; a multiply-add is the
/// intrinsic, written only where the list holds one, and no float instruction carries a flag that lets `llc` fuse
/// or reorder.
#[test]
fn an_instruction_loads_its_operands_and_stores_its_result() {
    let body = "f32 x = load a[i]\nf32 v = x * x\nf32 w = v + x\nf32 o[i] = w";
    let exact = ir(body, A_O, EXACT).unwrap();
    assert_eq!(
        lines(&exact, "f32 v = x * x"),
        [
            "%t20 = load float, ptr addrspace(5) %f0",
            "%t21 = load float, ptr addrspace(5) %f0",
            "%t22 = fmul float %t20, %t21",
            "store float %t22, ptr addrspace(5) %f1",
        ]
    );
    assert!(!exact.contains("llvm.fma"), "{exact}");
    let fused = ir(body, A_O, FUSED).unwrap();
    assert!(fused.contains("call float @llvm.fma.f32(float"), "{fused}");
    assert!(!fused.contains("fmul"), "{fused}");
}

/// A label opens a block the one before falls into, a guarded branch forks to the block after it, and a loop's
/// counter is a slot stepped in place.
#[test]
fn a_loop_is_blocks_around_a_slot_stepped_in_place() {
    let text = ir(
        "
        let f32 acc = 0.0
        for j in 0..n:
            f32 x = load a[i * 4 + j]
            f32 acc = acc + x
        f32 o[i] = acc
        ",
        &[
            ("a", Input, Some(DType::F32)),
            ("o", Output, Some(DType::F32)),
            ("n", Int, None),
        ],
        EXACT,
    )
    .unwrap();
    assert!(text.contains("\tbr label %L"), "{text}");
    assert!(text.contains("\nL"), "{text}");
    let forks = text.lines().filter(|line| line.contains("br i1 ")).count();
    assert_eq!(
        forks, 2,
        "the test ahead of the loop and the one at its bottom:\n{text}"
    );
}

/// A conversion to an integer saturates as PTX's does, and flush to zero is the kernel's attribute.
#[test]
fn a_conversion_saturates_and_flush_to_zero_is_an_attribute() {
    let ftz = Precision {
        fuse: false,
        ftz: true,
    };
    let text = ir(
        "f32 x = load a[i]\ni32 n = (i32)x\nf32 v = (f32)n\nf32 w = v + x\nf32 o[i] = w",
        A_O,
        ftz,
    )
    .unwrap();
    assert!(
        text.contains("call i32 @llvm.fptosi.sat.i32.f32(float"),
        "{text}"
    );
    assert!(text.contains("sitofp i32"), "{text}");
    assert!(
        text.contains("\"denormal-fp-math-f32\"=\"preserve-sign,preserve-sign\""),
        "{text}"
    );
}

fn rows(statements: &str) -> emmy_ir::Result<String> {
    let body = format!(
        "Grid(axes=(r in 0..64, t in 0..64), decode=(block_idx.x, thread_idx.x), block_threads=64):\n{}",
        indented(statements)
    );
    amdgcn_ir(&body, &Interface { ..interface(A_O) }, EXACT)
}

/// A shared buffer is an array of the block's local memory, addressed in 32 bits, and a barrier of the whole
/// block is AMD's.
#[test]
fn a_shared_buffer_is_local_memory_and_a_barrier_is_the_blocks() {
    let text = rows(
        "
        Smem(name=s, extents=(8, 8), dtype=f32)
        f32 x = load a[r * 64 + t]
        f32 s[t] = x
        Sync
        f32 y = load s[63 - t]
        f32 o[r * 64 + t] = y
        ",
    )
    .unwrap();
    assert!(
        text.contains("@k_shared_0 = internal addrspace(3) global [256 x i8] undef, align 4\n"),
        "{text}"
    );
    assert!(
        text.contains("ptrtoint ptr addrspace(3) @k_shared_0 to i32"),
        "{text}"
    );
    assert!(text.contains("to ptr addrspace(3)\n"), "{text}");
    assert_eq!(lines(&text, "Sync"), ["call void @llvm.amdgcn.s.barrier()"]);
}

/// What only an NVIDIA card has is refused by name: a statement by the rule that asks the target for its
/// instruction, an instruction every NVIDIA card has by the printer.
#[test]
fn an_nvidia_instruction_is_refused_by_name() {
    let shuffle = rows(
        "
        f32 acc = load a[r * 64 + t]
        WarpShuffle(state=(acc), state_b=(acc__o), length=32):
            combine_states:
                f32 acc = acc + acc__o
        f32 o[r * 64 + t] = acc
        ",
    )
    .unwrap_err();
    assert!(matches!(shuffle, Error::NotImplemented(_)), "{shuffle}");
    let message = shuffle.to_string();
    assert!(
        message.starts_with("no LLVM IR for `") && message.ends_with("on gfx942"),
        "{message}"
    );
    let fast = ir(
        "f32 x = load a[i]\nf32 v = exp_fast(x)\nf32 o[i] = v",
        A_O,
        EXACT,
    )
    .unwrap_err();
    assert!(matches!(fast, Error::NotImplemented(_)), "{fast}");
    assert!(fast.to_string().contains("ex2.approx"), "{fast}");
}
