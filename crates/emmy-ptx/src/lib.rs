//! The compiler from Kernel IR to NVIDIA PTX. A translation: it reads a kernel body that is already typed and flat,
//! writes one instruction list through a builder, checks it, and prints it. It makes no decision of its own; the
//! kernel's two precision decisions reach it as input. ptxas assembles the text it returns.

pub mod builder;
pub mod inst;
mod lower;
pub mod print;
pub mod verify;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::{Name, Result};

use crate::inst::Module;

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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Param {
    pub name: Name,
    pub kind: ParamKind,
    /// The element type of a buffer.
    pub dtype: Option<DType>,
    /// Whether the parameter is a page table of the buffer and not the buffer.
    pub paged: bool,
}

/// What the launch and the code must agree on: the parameters in call order, the per-block thread bound, the
/// shared-memory pool, and whether the kernel needs the arch-specific instruction set.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Interface {
    pub params: Vec<Param>,
    /// The most threads a block of the kernel launches with.
    pub threads: u32,
    /// The byte offset of each shared buffer in the launch-sized pool; empty when the buffers are static arrays.
    pub shared_offsets: Vec<(Name, u64)>,
    /// The pool's size in bytes, 0 when the buffers are static arrays.
    pub shared_pool_bytes: u64,
    pub arch_specific: bool,
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
pub fn lower(kernel: &Kernel, target: Target, precision: Precision) -> Result<Module> {
    let module = lower::lower(kernel, target, precision)?;
    verify::verify(&module)?;
    Ok(module)
}

/// The PTX text of `kernel` for `target`.
pub fn compile(kernel: &Kernel, target: Target, precision: Precision) -> Result<String> {
    Ok(print::print(&lower(kernel, target, precision)?))
}
