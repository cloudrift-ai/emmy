//! The bfloat16 conversions on a card before sm_80, which has no `cvt` to or from bf16: a bf16 is the high half of
//! an f32's bits. A widening places the half there, exactly. A narrowing rounds the f32's bits to nearest, ties to
//! even, in integer arithmetic, which is what `cvt.rn.bf16.f32` computes for every value but a NaN; a NaN keeps its
//! sign and the top of its payload, with the quiet bit set.

use std::collections::HashMap;

use crate::inst::{Bank, Cmp, Inst, Item, Opcode, Operand, Reg, Ty};

/// Rewrite every scalar conversion between bf16 and f32 in `items` into integer instructions, taking the registers
/// it needs past the counts in `registers`.
pub(crate) fn legalize(items: Vec<Item>, registers: &mut [(Bank, u32)]) -> Vec<Item> {
    let mut counts: HashMap<Bank, u32> = registers.iter().copied().collect();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::Inst(inst) if is_widening(&inst) => widen(inst, &mut counts, &mut out),
            Item::Inst(inst) if is_narrowing(&inst) => narrow(inst, &mut counts, &mut out),
            other => out.push(other),
        }
    }
    for (bank, count) in registers.iter_mut() {
        *count = counts[bank];
    }
    out
}

fn is_widening(inst: &Inst) -> bool {
    inst.opcode == Opcode::Cvt(Ty::Bf16) && inst.ty == Ty::F32
}

fn is_narrowing(inst: &Inst) -> bool {
    inst.opcode == Opcode::Cvt(Ty::F32) && inst.ty == Ty::Bf16
}

struct Emit<'a> {
    counts: &'a mut HashMap<Bank, u32>,
    out: &'a mut Vec<Item>,
    guard: Option<crate::inst::Guard>,
}

impl Emit<'_> {
    fn reg(&mut self, bank: Bank) -> Reg {
        let count = self.counts.entry(bank).or_insert(0);
        *count += 1;
        Reg {
            bank,
            number: *count - 1,
        }
    }

    /// `opcode` at `ty` over `operands` into `result`, under the conversion's own guard.
    fn put(&mut self, result: Reg, opcode: Opcode, ty: Ty, operands: Vec<Operand>) {
        let inst = Inst {
            guard: self.guard,
            ..Inst::new(opcode, ty, vec![result], operands)
        };
        self.out.push(Item::Inst(inst));
    }

    fn word(&mut self, opcode: Opcode, ty: Ty, operands: Vec<Operand>) -> Reg {
        let reg = self.reg(Bank::B32);
        self.put(reg, opcode, ty, operands);
        reg
    }
}

/// `cvt.f32.bf16 f, h`: the half's bits, zero-extended and shifted into the high half, are the f32.
fn widen(inst: Inst, counts: &mut HashMap<Bank, u32>, out: &mut Vec<Item>) {
    let mut e = Emit {
        counts,
        out,
        guard: inst.guard,
    };
    let half = inst.operands[0].clone();
    let low = e.word(Opcode::Cvt(Ty::U16), Ty::U32, vec![half]);
    let bits = e.word(Opcode::Shl, Ty::B32, vec![low.into(), Operand::Int(16)]);
    e.put(inst.results[0], Opcode::Mov, Ty::B32, vec![bits.into()]);
}

/// `cvt.rn.bf16.f32 h, f`: add `0x7fff` and the lowest kept bit, then keep the high half — round to nearest, ties
/// to even, an overflow carried into the exponent and on to infinity. A NaN (magnitude above the infinity's bits)
/// takes its high half with the quiet bit set instead, so a payload in the dropped bits never rounds to infinity.
fn narrow(inst: Inst, counts: &mut HashMap<Bank, u32>, out: &mut Vec<Item>) {
    let mut e = Emit {
        counts,
        out,
        guard: inst.guard,
    };
    let bits = e.word(Opcode::Mov, Ty::B32, vec![inst.operands[0].clone()]);
    let high = e.word(Opcode::Shr, Ty::U32, vec![bits.into(), Operand::Int(16)]);
    let odd = e.word(Opcode::And, Ty::B32, vec![high.into(), Operand::Int(1)]);
    let bias = e.word(Opcode::Add, Ty::U32, vec![odd.into(), Operand::Int(0x7fff)]);
    let sum = e.word(Opcode::Add, Ty::U32, vec![bits.into(), bias.into()]);
    let rounded = e.word(Opcode::Shr, Ty::U32, vec![sum.into(), Operand::Int(16)]);
    let quiet = e.word(Opcode::Or, Ty::B32, vec![high.into(), Operand::Int(0x40)]);
    let magnitude = e.word(
        Opcode::And,
        Ty::B32,
        vec![bits.into(), Operand::Int(0x7fff_ffff)],
    );
    let is_nan = e.reg(Bank::Pred);
    let compare = vec![magnitude.into(), Operand::Int(0x7f80_0000)];
    e.put(is_nan, Opcode::Setp(Cmp::Gt), Ty::U32, compare);
    let pick = vec![quiet.into(), rounded.into(), is_nan.into()];
    let picked = e.word(Opcode::Selp, Ty::B32, pick);
    e.put(
        inst.results[0],
        Opcode::Cvt(Ty::U32),
        Ty::U16,
        vec![picked.into()],
    );
}
