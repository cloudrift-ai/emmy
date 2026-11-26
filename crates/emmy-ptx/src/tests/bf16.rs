//! The bfloat16 conversions before sm_80: the integer instructions that stand in for `cvt`, run here over the edge
//! values against what `cvt.rn.bf16.f32` and `cvt.f32.bf16` give, and a kernel that converts assembled for sm_70.

use std::collections::HashMap;

use crate::Target;
use crate::inst::{Bank, Cmp, Inst, Item, Opcode, Operand, Reg, Round, Ty};
use crate::lower::bf16::legalize;
use emmy_ir::dtype::DType;

use super::common::{EXACT, listing, lowered};
use crate::ParamKind::{Input, Output};

fn reg(bank: Bank, number: u32) -> Reg {
    Reg { bank, number }
}

/// Run the straight-line `items` over the registers in `regs`, as the card would: every value its bits.
fn run(items: &[Item], regs: &mut HashMap<Reg, u64>) {
    for item in items {
        let Item::Inst(inst) = item else { continue };
        let read = |operand: &Operand, regs: &HashMap<Reg, u64>| match operand {
            Operand::Reg(r) => regs[r],
            Operand::Int(value) => *value as u64,
            other => panic!("operand {other:?}"),
        };
        let a = read(&inst.operands[0], regs);
        let b = inst.operands.get(1).map(|o| read(o, regs));
        let value = match inst.opcode {
            Opcode::Mov => a & 0xffff_ffff,
            Opcode::Shl => (a << b.unwrap()) & 0xffff_ffff,
            Opcode::Shr => a >> b.unwrap(),
            Opcode::And => a & b.unwrap(),
            Opcode::Or => a | b.unwrap(),
            Opcode::Add => (a + b.unwrap()) & 0xffff_ffff,
            Opcode::Setp(Cmp::Gt) => u64::from(a > b.unwrap()),
            Opcode::Selp => {
                if read(&inst.operands[2], regs) != 0 {
                    a
                } else {
                    b.unwrap()
                }
            }
            Opcode::Cvt(Ty::U16) => a & 0xffff,
            Opcode::Cvt(Ty::U32) => a & 0xffff,
            other => panic!("instruction {other:?}"),
        };
        regs.insert(inst.results[0], value);
    }
}

fn narrowed(bits: u32) -> u16 {
    let (f, h) = (reg(Bank::F32, 0), reg(Bank::B16, 0));
    let cvt = Inst::new(Opcode::Cvt(Ty::F32), Ty::Bf16, vec![h], vec![f.into()]).rounded(Round::Rn);
    let mut counts = vec![
        (Bank::F32, 1),
        (Bank::B16, 1),
        (Bank::B32, 0),
        (Bank::Pred, 0),
    ];
    let items = legalize(vec![Item::Inst(cvt)], &mut counts, 70);
    let mut regs = HashMap::from([(f, u64::from(bits))]);
    run(&items, &mut regs);
    regs[&h] as u16
}

fn widened(half: u16) -> u32 {
    let (h, f) = (reg(Bank::B16, 0), reg(Bank::F32, 0));
    let cvt = Inst::new(Opcode::Cvt(Ty::Bf16), Ty::F32, vec![f], vec![h.into()]);
    let mut counts = vec![
        (Bank::F32, 1),
        (Bank::B16, 1),
        (Bank::B32, 0),
        (Bank::Pred, 0),
    ];
    let items = legalize(vec![Item::Inst(cvt)], &mut counts, 70);
    let mut regs = HashMap::from([(h, u64::from(half))]);
    run(&items, &mut regs);
    regs[&f] as u32
}

#[test]
fn a_narrowing_rounds_to_nearest_even_as_cvt_does() {
    let cases: [(u32, u16); 22] = [
        (0x0000_0000, 0x0000), // +0
        (0x8000_0000, 0x8000), // -0
        (0x3f80_0000, 0x3f80), // 1
        (0xbf80_0000, 0xbf80), // -1
        (0x3f80_7fff, 0x3f80), // below the tie
        (0x3f80_8000, 0x3f80), // a tie to the even below
        (0x3f81_8000, 0x3f82), // a tie to the even above
        (0x3f80_8001, 0x3f81), // above the tie
        (0xbf81_8000, 0xbf82), // a negative tie
        (0x7f7f_ffff, 0x7f80), // the largest f32 rounds to infinity
        (0x7f7f_7fff, 0x7f7f), // the largest bf16 stays
        (0xff7f_ffff, 0xff80), // and to -infinity
        (0x7f80_0000, 0x7f80), // +inf
        (0xff80_0000, 0xff80), // -inf
        (0x0000_0001, 0x0000), // the smallest subnormal rounds to zero
        (0x0000_8000, 0x0000), // a subnormal tie to even zero
        (0x0001_8000, 0x0002), // a subnormal tie to even above
        (0x0001_0000, 0x0001), // the smallest bf16 subnormal
        (0x007f_ffff, 0x0080), // the largest subnormal rounds into the normals
        (0x8001_8000, 0x8002), // a negative subnormal tie
        (0x7fc0_0000, 0x7fc0), // the quiet NaN
        (0xffc0_0000, 0xffc0), // a negative quiet NaN keeps its sign
    ];
    for (bits, half) in cases {
        assert_eq!(narrowed(bits), half, "{bits:#010x}");
    }
}

#[test]
fn a_nan_stays_a_nan_with_its_sign() {
    // A signalling NaN whose payload lies only in the dropped bits would round to infinity in integer arithmetic.
    for bits in [
        0x7f80_0001u32,
        0x7fbf_ffff,
        0xff80_0001,
        0x7fff_ffff,
        0xffff_ffff,
    ] {
        let half = narrowed(bits);
        assert!(
            half & 0x7f80 == 0x7f80 && half & 0x7f != 0,
            "{bits:#010x} -> {half:#06x}"
        );
        assert_eq!(half & 0x8000, (bits >> 16) as u16 & 0x8000);
        assert_ne!(half & 0x40, 0, "the quiet bit");
    }
}

#[test]
fn a_widening_is_the_high_half() {
    for half in [
        0x0000u16, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f7f, 0x7f80, 0xff80, 0x7fc0, 0xffff,
    ] {
        assert_eq!(widened(half), u32::from(half) << 16);
    }
}

const GRID: &str =
    "Grid(axes=(i in 0..4096), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):";

const SELECTION: &str = "
        bf16 x = load a[i]
        init bf16 m = 0
        bf16 m <- maximum(m, x)
        bf16 o[i] = m
        ";

const PARAMS: super::common::Params = &[
    ("a", Input, Some(DType::BF16)),
    ("o", Output, Some(DType::BF16)),
];

/// Before sm_80 a kernel that converts holds no bf16 instruction, and ptxas assembles it for sm_70 wherever it still
/// takes that target; from sm_80 the conversions are the `cvt` instructions themselves.
#[test]
fn a_volta_kernel_converts_without_cvt_and_assembles() {
    let body = format!("{GRID}\n{}", super::common::indented(SELECTION));
    let volta = listing(&lowered(&body, PARAMS, Target { sm: 70 }, EXACT).unwrap());
    assert!(
        volta.iter().all(|line| !line.contains("bf16")),
        "{volta:#?}"
    );
    assert!(volta.iter().any(|line| line.starts_with("setp.gt.u32")));
    for sm in [80, 120] {
        let ampere = listing(&lowered(&body, PARAMS, Target { sm }, EXACT).unwrap());
        assert!(ampere.iter().any(|line| line.starts_with("cvt.f32.bf16")));
        assert!(
            ampere
                .iter()
                .any(|line| line.starts_with("cvt.rn.bf16.f32"))
        );
        assert!(ampere.iter().all(|line| !line.starts_with("setp.gt.u32")));
    }
}

/// An integer converted to bf16 and back goes through f32 before sm_90, which first has the `cvt`: before sm_80 no
/// bf16 instruction is left, from sm_80 the f32 step is the float `cvt`.
#[test]
fn a_volta_integer_conversion_goes_through_f32() {
    let body = format!(
        "{GRID}\n{}",
        super::common::indented(
            "
        i32 x = load a[i]
        bf16 b = (bf16)x
        i32 back = (i32)b
        i32 o[i] = back
        "
        )
    );
    let params: super::common::Params = &[
        ("a", Input, Some(DType::I32)),
        ("o", Output, Some(DType::I32)),
    ];
    let volta = listing(&lowered(&body, params, Target { sm: 70 }, EXACT).unwrap());
    assert!(
        volta.iter().all(|line| !line.contains("bf16")),
        "{volta:#?}"
    );
    assert!(
        volta.iter().any(|line| line.starts_with("cvt.rn.f32.s32")),
        "{volta:#?}"
    );
    let ampere = listing(&lowered(&body, params, Target { sm: 80 }, EXACT).unwrap());
    assert!(
        ampere
            .iter()
            .any(|line| line.starts_with("cvt.rn.bf16.f32")),
        "{ampere:#?}"
    );
    assert!(
        ampere
            .iter()
            .all(|line| !line.contains("bf16.s32") && !line.contains("s32.bf16")),
        "{ampere:#?}"
    );
    let hopper = listing(&lowered(&body, params, Target { sm: 90 }, EXACT).unwrap());
    assert!(
        hopper
            .iter()
            .any(|line| line.starts_with("cvt.rn.bf16.s32")),
        "{hopper:#?}"
    );
}
