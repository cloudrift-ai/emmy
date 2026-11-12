//! The verifier, run on every compile, for what ptxas does not check: every register is defined on every path to
//! each read of it, every label a branch names is bound once, and an alignment fact stands behind every vector
//! access. A register read undefined assembles, and gives a right answer at one optimization level and a wrong one
//! at another.

use std::collections::HashMap;

use emmy_ir::{Error, Result};

use crate::inst::{Base, Inst, Item, Label, Module, Opcode, Operand, Reg};
use crate::print::instruction_text;

/// A set of registers, one bit each.
#[derive(Clone, PartialEq)]
struct Defined(Vec<u64>);

impl Defined {
    fn new(size: usize, full: bool) -> Defined {
        Defined(vec![if full { u64::MAX } else { 0 }; size.div_ceil(64)])
    }

    fn insert(&mut self, at: usize) {
        self.0[at / 64] |= 1 << (at % 64);
    }

    fn contains(&self, at: usize) -> bool {
        self.0[at / 64] & (1 << (at % 64)) != 0
    }

    fn intersect(&mut self, other: &Defined) {
        for (word, theirs) in self.0.iter_mut().zip(&other.0) {
            *word &= theirs;
        }
    }
}

/// A run of instructions control enters only at its first and leaves only after its last.
struct Block {
    insts: Vec<usize>,
    /// The blocks control may go to next.
    next: Vec<usize>,
}

fn refusal(message: String) -> Error {
    Error::Value(format!("PTX verifier: {message}"))
}

pub fn verify(module: &Module) -> Result<()> {
    let labels = bound_labels(module)?;
    let blocks = blocks(module, &labels)?;
    definitions(module, &blocks)?;
    vector_accesses(module)
}

/// The item each label is bound at; an error for a label bound twice.
fn bound_labels(module: &Module) -> Result<HashMap<Label, usize>> {
    let mut labels = HashMap::new();
    for (at, item) in module.items.iter().enumerate() {
        if let Item::Label(label) = item
            && labels.insert(*label, at).is_some()
        {
            return Err(refusal(format!("the label L{} is bound twice", label.0)));
        }
    }
    Ok(labels)
}

fn instruction(module: &Module, at: usize) -> Option<&Inst> {
    match module.items.get(at) {
        Some(Item::Inst(inst)) => Some(inst),
        _ => None,
    }
}

/// The blocks of the code: one starts at the entry, at every label and after every branch.
fn blocks(module: &Module, labels: &HashMap<Label, usize>) -> Result<Vec<Block>> {
    let mut starts = vec![0];
    for (at, item) in module.items.iter().enumerate() {
        match item {
            Item::Label(_) => starts.push(at),
            Item::Inst(inst) if matches!(inst.opcode, Opcode::Bra | Opcode::Ret) => {
                starts.push(at + 1)
            }
            _ => {}
        }
    }
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|start| *start < module.items.len());
    let block_at = |item: usize| {
        starts
            .partition_point(|start| *start <= item)
            .saturating_sub(1)
    };
    let mut blocks = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(module.items.len());
        let insts: Vec<usize> = (*start..end)
            .filter(|at| instruction(module, *at).is_some())
            .collect();
        let last = insts.last().and_then(|at| instruction(module, *at));
        let mut next = Vec::new();
        let falls_through = match last {
            Some(inst) if inst.opcode == Opcode::Ret => inst.guard.is_some(),
            Some(inst) if inst.opcode == Opcode::Bra => {
                let target = inst
                    .target()
                    .ok_or_else(|| refusal("a branch names no label".into()))?;
                let bound = labels
                    .get(&target)
                    .ok_or_else(|| refusal(format!("the label L{} is never bound", target.0)))?;
                next.push(block_at(*bound));
                inst.guard.is_some()
            }
            _ => true,
        };
        if falls_through && index + 1 < starts.len() {
            next.push(index + 1);
        }
        blocks.push(Block { insts, next });
    }
    Ok(blocks)
}

/// Every register read is defined on every path from the entry. A definition under a guard defines its register
/// on some paths only, so it counts for none.
fn definitions(module: &Module, blocks: &[Block]) -> Result<()> {
    let mut offsets = HashMap::new();
    let mut size = 0;
    for (bank, count) in &module.registers {
        offsets.insert(*bank, size);
        size += *count as usize;
    }
    let position = |reg: Reg| -> Result<usize> {
        let count = module.registers.iter().find(|(bank, _)| *bank == reg.bank);
        match (offsets.get(&reg.bank), count) {
            (Some(offset), Some((_, count))) if reg.number < *count => {
                Ok(offset + reg.number as usize)
            }
            _ => Err(refusal(format!("the register {reg:?} is not declared"))),
        }
    };
    // What is defined where each block starts: nothing at the entry, and for any other block what every block
    // that reaches it leaves defined, found by narrowing from everything.
    let mut entering: Vec<Defined> = (0..blocks.len())
        .map(|index| Defined::new(size, index != 0))
        .collect();
    let leaving = |block: &Block, entering: &Defined| -> Result<Defined> {
        let mut defined = entering.clone();
        for at in &block.insts {
            if let Some(inst) = instruction(module, *at)
                && inst.guard.is_none()
            {
                for reg in &inst.results {
                    defined.insert(position(*reg)?);
                }
            }
        }
        Ok(defined)
    };
    let mut pending: Vec<usize> = (0..blocks.len()).rev().collect();
    while let Some(index) = pending.pop() {
        let out = leaving(&blocks[index], &entering[index])?;
        for next in &blocks[index].next {
            let mut narrowed = entering[*next].clone();
            narrowed.intersect(&out);
            if narrowed != entering[*next] {
                entering[*next] = narrowed;
                pending.push(*next);
            }
        }
    }
    for (block, entering) in blocks.iter().zip(&entering) {
        let mut defined = entering.clone();
        for at in &block.insts {
            let Some(inst) = instruction(module, *at) else {
                continue;
            };
            for reg in inst.reads() {
                if !defined.contains(position(reg)?) {
                    return Err(refusal(format!(
                        "`{}` reads %{}{} on a path that does not define it",
                        instruction_text(inst),
                        reg.bank.prefix(),
                        reg.number
                    )));
                }
            }
            if inst.guard.is_none() {
                for reg in &inst.results {
                    defined.insert(position(*reg)?);
                }
            }
        }
    }
    Ok(())
}

/// A load or a store of several registers at once reads or writes that many elements in one access, which the
/// hardware takes only at an address aligned to the whole access: the base register must be proven so aligned, and
/// the constant offset a multiple of it.
fn vector_accesses(module: &Module) -> Result<()> {
    for inst in module.instructions() {
        let lanes = match inst.opcode {
            Opcode::Ld(_) => inst.results.len(),
            Opcode::St(_) => inst.operands.iter().map(Operand::lanes).max().unwrap_or(1),
            _ => 1,
        };
        if lanes < 2 {
            continue;
        }
        let bytes = lanes as i64 * inst.ty.bytes();
        let aligned = inst.operands.iter().any(|operand| match operand {
            Operand::Addr(Base::Reg(reg), offset) => {
                let proven = module.alignment.get(reg).copied().unwrap_or(1);
                i64::from(proven) % bytes == 0 && offset % bytes == 0
            }
            _ => false,
        });
        if !aligned {
            return Err(refusal(format!(
                "`{}` is a {bytes}-byte access with no alignment proven of its address",
                instruction_text(inst)
            )));
        }
    }
    Ok(())
}
