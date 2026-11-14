//! The compiler from Kernel IR to NVIDIA PTX. A translation: it reads a kernel body that is already typed and flat,
//! writes one instruction list through a builder, checks it, and prints it. It makes no decision of its own; the
//! kernel's two precision decisions reach it as input. ptxas assembles the text it returns.

mod builder;
mod inst;
mod lower;
mod print;
#[cfg(test)]
mod tests;
mod verify;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::expr::Expression;
use emmy_ir::{Name, Result};

use crate::inst::{Module, Opcode};

/// What a kernel parameter is to the launch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamKind {
    /// A buffer the kernel reads.
    Input,
    /// A buffer the kernel writes.
    Output,
    /// A host-built tensor-map descriptor.
    Descriptor,
    /// A symbolic size, passed as a 32-bit integer.
    Int,
}

/// One kernel parameter, in the launch's argument order.
#[derive(Clone, PartialEq, Debug)]
pub struct Param {
    /// The name the body reads it by: a buffer's own name, also where the launch binds its page table.
    pub name: Name,
    pub kind: ParamKind,
    /// The element type of a buffer.
    pub dtype: Option<DType>,
    /// Where the parameter is a table of the buffer's pages and not the buffer: the dimension the pages cut and the
    /// extent of a page along it.
    pub page: Option<Page>,
    /// The bytes the launch gives a buffer's first element as alignment: what stands behind every access of
    /// several elements.
    pub align: u32,
    /// The extents of a buffer, one a dimension, a symbolic one as an expression over the sizes the launch passes:
    /// what flattens the index of a statement that names an element by one coordinate a dimension. Empty where no
    /// statement of the kernel does.
    pub shape: Vec<Expression>,
    /// Whether the buffer is the e2m1 pair table: a load of it computes its value from the index and reads nothing.
    pub pair_table: bool,
}

/// How a paged buffer is cut: its pages are equal, each the buffer's shape with dimension `axis` cut to `size`, and
/// the table holds the address of each, in order along that dimension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Page {
    pub axis: usize,
    pub size: i64,
}

/// What the launch and the code must agree on: the parameters in call order and the shared-memory pool. The thread
/// bound is the grid's, and the instruction set the module targets is what its instructions need.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Interface {
    pub params: Vec<Param>,
    /// The byte offset of each shared buffer in the launch-sized pool; empty when the buffers are static arrays.
    pub shared_offsets: Vec<(Name, u64)>,
    /// The pool's size in bytes, 0 when the buffers are static arrays.
    pub shared_pool_bytes: u64,
    /// Whether the launch places at most one block on each multiprocessor: a static grid of no more blocks than
    /// the card has multiprocessors.
    pub single_wave: bool,
}

/// One kernel: its entry name, its body and its interface.
#[derive(Clone, Debug)]
pub struct Kernel<'a> {
    pub name: &'a str,
    pub body: &'a Body,
    pub interface: &'a Interface,
}

/// The card the kernel is assembled for, by compute capability: `120` is `sm_120`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target {
    pub sm: u32,
}

/// The first compute capability whose target needs the later PTX version.
const SM_BLACKWELL: u32 = 100;

impl Target {
    /// The target as ptxas spells it, with the suffix of the arch-specific instruction set when the kernel needs it.
    pub fn text(self, arch_specific: bool) -> String {
        format!("sm_{}{}", self.sm, if arch_specific { "a" } else { "" })
    }

    /// The lowest PTX version that names the target and every instruction the compiler writes: an older toolkit's
    /// ptxas refuses a version past its own.
    pub fn version(self) -> (u32, u32) {
        if self.sm >= SM_BLACKWELL {
            (8, 7)
        } else {
            (7, 8)
        }
    }
}

/// A kernel's two whole-kernel precision decisions.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Precision {
    /// A multiply read once by an add, a subtract or an accumulate in its own scope becomes one fused multiply-add.
    pub fuse: bool,
    /// Subnormal f32 operands and results are flushed to zero.
    pub ftz: bool,
}

/// The instruction list of `kernel` for `target`, verified. An error names the statement the input check refuses,
/// or the statement or operation that has no rule.
pub(crate) fn lower(kernel: &Kernel, target: Target, precision: Precision) -> Result<Module> {
    let module = lower::lower(kernel, target, precision)?;
    verify::verify(&module)?;
    Ok(module)
}

/// A kernel's PTX: the text, and whether the kernel waits on the grid launched ahead of it, so it may be launched as
/// a programmatic dependent launch.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ptx {
    pub text: String,
    pub waits_on_grid: bool,
}

/// The PTX of `kernel` for `target`.
pub fn compile(kernel: &Kernel, target: Target, precision: Precision) -> Result<Ptx> {
    let module = lower(kernel, target, precision)?;
    let waits_on_grid = module
        .instructions()
        .any(|inst| inst.opcode == Opcode::GridWait);
    Ok(Ptx {
        text: print::print(&module),
        waits_on_grid,
    })
}
