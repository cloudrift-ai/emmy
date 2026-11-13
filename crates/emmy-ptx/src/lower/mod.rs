//! The rules: one per statement type, each writing the statement's instructions through the builder. The input is
//! taken as typed and flat: a value without a dtype, an operand of another type than its reader, or an index of
//! several coordinates is refused by the statement's name, and so is a statement or an operation with no rule yet.

mod coop;
mod fragment;
mod hopper;
mod int;
mod math;
mod memory;
mod tensor;
mod value;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expr, Expression, SimplifyCtx};
use emmy_ir::normalize::{inner_ctx, launch_ctx};
use emmy_ir::statement::{
    Accum, Assign, Cond, Grid, Init, Let, Loop, Select, Statement, StridedLoop,
};
use emmy_ir::text::statement_text;
use emmy_ir::{Error, Name, Result};

use self::coop::Threads;
use self::memory::{Buffer, Carried, Elem};
use crate::builder::Builder;
use crate::inst::{Bank, Base, Inst, Module, Opcode, Operand, ParamDecl, Reg, Shared, Space, Ty};
use crate::{Kernel, ParamKind, Precision, Target};

/// The most threads a block can hold.
const MAX_BLOCK_THREADS: u32 = 1024;

/// The first compute capability with the hardware's hyperbolic tangent.
const SM_TURING: u32 = 75;

/// The first compute capability with a half-precision minimum and maximum, and with async copies.
const SM_AMPERE: u32 = 80;

/// The first compute capability whose kernels wait on the grid ahead of them.
const SM_HOPPER: u32 = 90;

/// The alignment the launch gives a tensor-map descriptor.
const DESCRIPTOR_ALIGN: u32 = 64;

/// What an instruction asks of the target: the compute capabilities that have it, whether only their
/// arch-specific instruction set does, and the first PTX version that names it. The cards are what ptxas of CUDA
/// 13.0 takes and refuses; so is every version from 7.8 on, the module's own lowest, and an earlier one is the PTX
/// reference's.
pub(crate) struct Needs {
    /// The instruction, as an error names it.
    pub what: &'static str,
    pub sm: std::ops::RangeInclusive<u32>,
    pub arch_specific: bool,
    pub version: (u32, u32),
}

/// The last compute capability there is to ask for.
pub(crate) const SM_LAST: u32 = u32::MAX;

/// The alignment of the shared pool when no buffer in it asks for more: what a 16-byte access needs.
const POOL_ALIGN: u32 = 16;

/// The type of a value the rules compute: the dtypes a scalar kernel holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum VTy {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    I64,
    Bool,
    /// An 8-bit float, or a byte of two 4-bit e2m1 codes, as its bits in the low byte of a 32-bit register: what a
    /// buffer of them stores and a cast converts. No arithmetic reads one.
    F8E4M3,
    F8E5M2,
    F4x2,
}

impl VTy {
    fn of(dtype: DType) -> Option<VTy> {
        Some(match dtype {
            DType::F32 => VTy::F32,
            DType::F16 => VTy::F16,
            DType::BF16 => VTy::Bf16,
            DType::I32 => VTy::I32,
            DType::U32 => VTy::U32,
            DType::I64 => VTy::I64,
            DType::Bool => VTy::Bool,
            DType::F8E4M3 => VTy::F8E4M3,
            DType::F8E5M2 => VTy::F8E5M2,
            DType::F4E2M1x2 => VTy::F4x2,
            _ => return None,
        })
    }

    /// Whether the value is the bits of an 8-bit float or of a pair of 4-bit codes.
    pub(super) fn is_byte_float(self) -> bool {
        matches!(self, VTy::F8E4M3 | VTy::F8E5M2 | VTy::F4x2)
    }

    fn is_int(self) -> bool {
        matches!(self, VTy::I32 | VTy::U32 | VTy::I64)
    }

    fn is_float(self) -> bool {
        matches!(self, VTy::F32 | VTy::F16 | VTy::Bf16)
    }

    /// The type an arithmetic instruction computes this value at.
    fn arith(self) -> Ty {
        match self {
            VTy::F32 => Ty::F32,
            VTy::F16 => Ty::F16,
            VTy::Bf16 => Ty::Bf16,
            VTy::I32 => Ty::S32,
            VTy::U32 => Ty::U32,
            VTy::I64 => Ty::S64,
            VTy::Bool => Ty::Pred,
            VTy::F8E4M3 | VTy::F8E5M2 | VTy::F4x2 => Ty::U32,
        }
    }

    /// The type a move, a select, a load or a store holds this value at: a half is its 16 bits.
    fn held(self) -> Ty {
        match self {
            VTy::F16 | VTy::Bf16 => Ty::B16,
            other => other.arith(),
        }
    }

    fn bank(self) -> Bank {
        self.arith().bank()
    }

    /// The bytes one element takes in a buffer: a bool is a byte.
    fn bytes(self) -> i64 {
        self.held().bytes()
    }

    fn text(self) -> &'static str {
        match self {
            VTy::F32 => "f32",
            VTy::F16 => "f16",
            VTy::Bf16 => "bf16",
            VTy::I32 => "i32",
            VTy::U32 => "u32",
            VTy::I64 => "i64",
            VTy::Bool => "bool",
            VTy::F8E4M3 => "f8e4m3",
            VTy::F8E5M2 => "f8e5m2",
            VTy::F4x2 => "f4e2m1x2",
        }
    }
}

/// A value where it is read: a register or a constant, with its type.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Val {
    pub op: Operand,
    pub ty: VTy,
}

impl Val {
    fn reg(reg: Reg, ty: VTy) -> Val {
        Val { op: reg.into(), ty }
    }
}

pub(crate) struct Lowerer {
    b: Builder,
    target: Target,
    precision: Precision,
    buffers: HashMap<Name, Buffer>,
    /// The shared declarations of the module, and where each shared buffer starts in the launch-sized pool when
    /// the interface gives one.
    shared: Vec<Shared>,
    pool: HashMap<Name, u64>,
    /// The names in scope, one table per open body, the innermost last.
    names: Vec<HashMap<Name, Val>>,
    /// How many statements read each name.
    uses: HashMap<Name, u32>,
    /// The names some statement updates in place: each is a state from where it is defined.
    updated: HashSet<Name>,
    /// The products held back for the one add, subtract or accumulate that reads them: the type and the operands.
    products: HashMap<Name, (VTy, Operand, Operand)>,
    carried: HashMap<Name, Carried>,
    /// What is decided of the threads of a block where the statement being lowered stands.
    threads: Threads,
    /// The value of a register as an expression over the thread index alone, where it is one.
    by_thread: HashMap<Reg, Expression>,
    /// How many bodies are open in the grid's own body.
    grid_depth: usize,
    /// The PTX version the module states: the target's own, raised by each instruction that needs a later one.
    version: (u32, u32),
    /// Whether the interface asks for the target's arch-specific instruction set.
    arch_specific: bool,
    /// The register holding the address of each tensor-map descriptor the launch passes.
    descriptors: HashMap<Name, Reg>,
    /// The extents of each buffer the kernel or the interface states them of, for a statement that indexes one by
    /// several coordinates.
    extents: HashMap<Name, Vec<Expression>>,
    /// The inputs that are the e2m1 pair table: a load of one computes its value.
    pair_tables: HashSet<Name>,
    /// The first line of the statement being lowered, for an error to name.
    statement: String,
}

/// How many statements read each name, and the names a statement updates in place.
fn survey(body: &Body, uses: &mut HashMap<Name, u32>, updated: &mut HashSet<Name>) {
    for stmt in body.stmts() {
        for name in stmt.deps() {
            *uses.entry(name).or_insert(0) += 1;
        }
        updated.extend(stmt.updates());
        for nested in stmt.nested() {
            survey(&nested, uses, updated);
        }
    }
}

pub(crate) fn lower(kernel: &Kernel, target: Target, precision: Precision) -> Result<Module> {
    let interface = kernel.interface;
    if !(1..=MAX_BLOCK_THREADS).contains(&interface.threads) {
        return Err(Error::Value(format!(
            "a block of {} threads: the bound is 1 to {MAX_BLOCK_THREADS}",
            interface.threads
        )));
    }
    let mut lowerer = Lowerer {
        b: Builder::new(),
        target,
        precision,
        buffers: HashMap::new(),
        shared: Vec::new(),
        pool: interface.shared_offsets.iter().copied().collect(),
        names: vec![HashMap::new()],
        uses: HashMap::new(),
        updated: HashSet::new(),
        products: HashMap::new(),
        carried: HashMap::new(),
        threads: Threads::new(interface.threads as usize),
        by_thread: HashMap::new(),
        grid_depth: 0,
        version: target.version(),
        arch_specific: interface.arch_specific,
        descriptors: HashMap::new(),
        extents: HashMap::new(),
        pair_tables: HashSet::new(),
        statement: format!("the interface of {}", kernel.name),
    };
    survey(kernel.body, &mut lowerer.uses, &mut lowerer.updated);
    if !lowerer.pool.is_empty() {
        lowerer.declare_pool(kernel.body)?;
    }
    // A kernel waits for the grid launched ahead of it before it reaches any memory that grid may write.
    if target.sm >= SM_HOPPER {
        let wait = Inst::new(Opcode::GridWait, Ty::Pred, Vec::new(), Vec::new());
        lowerer.b.emit(wait);
    }
    let params = lowerer.parameters(kernel)?;
    lowerer.body(kernel.body, &SimplifyCtx::empty())?;
    let code = lowerer.b.finish()?;
    Ok(Module {
        version: lowerer.version,
        target: target.text(interface.arch_specific),
        entry: kernel.name.to_string(),
        params,
        shared: lowerer.shared,
        max_threads: interface.threads,
        registers: code.registers,
        items: code.items,
        alignment: code.alignment,
    })
}

impl Lowerer {
    fn no_rule(&self, what: impl std::fmt::Display) -> Error {
        Error::NoRule(format!("no PTX rule for {what}: `{}`", self.statement))
    }

    fn refuse(&self, why: impl std::fmt::Display) -> Error {
        Error::Value(format!("{why}: `{}`", self.statement))
    }

    /// An error where the target lacks the instruction `needs` describes; the module's version is raised to the
    /// one that names it.
    fn requires(&mut self, needs: &Needs) -> Result<()> {
        let (what, sm) = (needs.what, self.target.sm);
        if !needs.sm.contains(&sm) {
            let suffix = if needs.arch_specific { "a" } else { "" };
            let (first, last) = (needs.sm.start(), needs.sm.end());
            let has = match *last {
                SM_LAST => format!("sm_{first}{suffix} or later"),
                last if last / 10 == first / 10 => format!("sm_{first}{suffix}"),
                last => format!("sm_{first}{suffix} to sm_{last}{suffix}"),
            };
            return Err(self.no_rule(format_args!("{what} on sm_{sm}, which needs {has}")));
        }
        if needs.arch_specific && !self.arch_specific {
            return Err(self.refuse(format_args!(
                "{what} needs the arch-specific instruction set, sm_{sm}a, which the interface does not ask for"
            )));
        }
        self.version = self.version.max(needs.version);
        Ok(())
    }

    /// The parameter declarations, with each parameter loaded into a register: a buffer's pointer, a size's value.
    fn parameters(&mut self, kernel: &Kernel) -> Result<Vec<ParamDecl>> {
        kernel
            .interface
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| {
                let source = Operand::Addr(Base::Param(index as u32), 0);
                if param.paged {
                    return Err(self.no_rule(format_args!("the paged buffer {}", param.name)));
                }
                match param.kind {
                    ParamKind::Input | ParamKind::Output => {
                        let elem = self.elem(param.dtype)?;
                        let align = param.align;
                        if !align.is_power_of_two() || i64::from(align) < elem.bytes() {
                            let name = param.name;
                            return Err(self.refuse(format_args!(
                                "{name} is aligned to {align} bytes, which is no alignment of its elements"
                            )));
                        }
                        let base = self.b.reg(Bank::B64);
                        let load = Opcode::Ld(Space::Param);
                        self.b
                            .emit(Inst::new(load, Ty::U64, vec![base], vec![source]));
                        self.b.set_alignment(base, align);
                        let buffer = Buffer {
                            space: Space::Global,
                            base,
                            offset: 0,
                            elem,
                            writable: param.kind == ParamKind::Output,
                        };
                        self.buffers.insert(param.name, buffer);
                        if !param.shape.is_empty() {
                            self.extents.insert(param.name, param.shape.clone());
                        }
                        if param.pair_table {
                            self.pair_tables.insert(param.name);
                        }
                        Ok(ParamDecl::Pointer { align })
                    }
                    ParamKind::Int => {
                        let reg = self.b.reg(Bank::B32);
                        let load = Opcode::Ld(Space::Param);
                        self.b
                            .emit(Inst::new(load, Ty::U32, vec![reg], vec![source]));
                        self.bind(param.name, Val::reg(reg, VTy::I32));
                        Ok(ParamDecl::U32)
                    }
                    ParamKind::Descriptor => {
                        let address = self.b.reg(Bank::B64);
                        let load = Opcode::Ld(Space::Param);
                        self.b
                            .emit(Inst::new(load, Ty::U64, vec![address], vec![source]));
                        self.descriptors.insert(param.name, address);
                        Ok(ParamDecl::Pointer {
                            align: DESCRIPTOR_ALIGN,
                        })
                    }
                }
            })
            .collect()
    }

    /// The value type of a statement's dtype: the input check refuses a value that carries none.
    fn vty(&self, dtype: Option<DType>) -> Result<VTy> {
        let dtype = dtype.ok_or_else(|| self.refuse("a value carries no dtype"))?;
        VTy::of(dtype).ok_or_else(|| self.no_rule(format_args!("the dtype {dtype}")))
    }

    /// The element type of a buffer of `dtype`.
    fn elem(&self, dtype: Option<DType>) -> Result<Elem> {
        let dtype = dtype.ok_or_else(|| self.refuse("a value carries no dtype"))?;
        Elem::of(dtype).ok_or_else(|| self.no_rule(format_args!("the dtype {dtype}")))
    }

    /// Bind `name` to `val` in the innermost body. A name some statement updates in place is a state from here on.
    fn bind(&mut self, name: Name, val: Val) {
        let val = match val.op {
            Operand::Reg(reg) if self.b.is_state(reg) => val,
            _ if self.updated.contains(&name) => {
                let state = self.b.state(val.ty.bank());
                let ty = val.ty.held();
                self.b
                    .emit(Inst::new(Opcode::Mov, ty, vec![state], vec![val.op]));
                Val::reg(state, val.ty)
            }
            _ => val,
        };
        if let Some(scope) = self.names.last_mut() {
            scope.insert(name, val);
        }
    }

    /// A register for the value `name` is defined with: a state when some statement updates the name in place.
    fn fresh(&mut self, name: Name, bank: Bank) -> Reg {
        if self.updated.contains(&name) {
            self.b.state(bank)
        } else {
            self.b.reg(bank)
        }
    }

    /// The register of the state `name` is, which a statement updates in place at `ty`.
    fn state(&self, name: Name, ty: VTy) -> Result<Reg> {
        match self.find(name) {
            Some((_, val)) if val.ty != ty => {
                let (held, read) = (val.ty.text(), ty.text());
                Err(self.refuse(format_args!("{name} is {held}, updated at {read}")))
            }
            Some((_, val)) => val
                .op
                .reg()
                .filter(|reg| self.b.is_state(*reg))
                .ok_or_else(|| self.refuse(format_args!("{name} is no state"))),
            None => Err(self.refuse(format_args!("{name} is updated before it is defined"))),
        }
    }

    /// The value a name in scope holds, and how many bodies deep it was bound.
    fn find(&self, name: Name) -> Option<(usize, &Val)> {
        self.names
            .iter()
            .enumerate()
            .rev()
            .find_map(|(depth, scope)| scope.get(&name).map(|val| (depth, val)))
    }

    /// The value `name` holds where it is read. A product held back for a reader that turned out not to fuse it
    /// is computed here.
    fn lookup(&mut self, name: Name) -> Result<Val> {
        if let Some((ty, a, b)) = self.products.remove(&name) {
            let product = self.float_inst(Opcode::Mul, ty, vec![a, b]);
            let val = Val::reg(self.b.value(product, ty.bank()), ty);
            self.bind(name, val.clone());
            return Ok(val);
        }
        match self.find(name) {
            Some((_, val)) if matches!(val.op, Operand::Tuple(_)) => Err(self.refuse(
                format_args!("{name} is a fragment, read one register at a time"),
            )),
            Some((_, val)) => Ok(val.clone()),
            None => Err(self.refuse(format_args!("{name} is read before it is defined"))),
        }
    }

    /// `val` as the value of a name of its own: a state's register is copied, since the state steps on.
    fn named(&mut self, val: Val) -> Val {
        match val.op {
            Operand::Reg(reg) if self.b.is_state(reg) => {
                let copy = self.b.reg(reg.bank);
                let ty = val.ty.held();
                self.b
                    .emit(Inst::new(Opcode::Mov, ty, vec![copy], vec![reg.into()]));
                Val::reg(copy, val.ty)
            }
            _ => val,
        }
    }

    /// The constant a state of `ty` is seeded `value` with, as a move takes it.
    fn seed_constant(&self, ty: VTy, value: f64) -> Result<Operand> {
        match ty {
            VTy::F32 => Ok(Operand::f32(value as f32)),
            VTy::F16 | VTy::Bf16 => Ok(Operand::Int(i64::from(value::half_bits(ty, value)))),
            // An integer state holds a seed its type holds: the bounded stand-in of an unbounded neutral element
            // is a float's.
            VTy::I32 | VTy::U32 | VTy::I64 if int::holds_seed(ty, value) => {
                Ok(Operand::Int(value as i64))
            }
            _ => Err(self.no_rule(format_args!("a state of {} seeded {value}", ty.text()))),
        }
    }

    /// A new state named `name`, holding `value`.
    fn init(&mut self, name: Name, ty: VTy, value: f64) -> Result<()> {
        let constant = self.seed_constant(ty, value)?;
        let reg = self.b.state(ty.bank());
        self.b
            .emit(Inst::new(Opcode::Mov, ty.held(), vec![reg], vec![constant]));
        self.bind(name, Val::reg(reg, ty));
        Ok(())
    }

    /// Seed every accumulator the loop over `body` folds, ahead of the loop: each holds its combine's neutral
    /// element, at the bounded stand-in a kernel initializes with.
    fn seed(&mut self, body: &Body) -> Result<()> {
        let mut seen = HashSet::new();
        for stmt in body.stmts() {
            let Statement::Accum(accum) = &**stmt else {
                continue;
            };
            if !seen.insert(accum.name) {
                continue;
            }
            let ty = self.vty(accum.dtype)?;
            let identity = accum.op.seed().ok_or_else(|| {
                self.no_rule(format_args!("an accumulator of {}", accum.op.name()))
            })?;
            self.init(accum.name, ty, identity)?;
        }
        Ok(())
    }

    fn body(&mut self, body: &Body, ctx: &SimplifyCtx) -> Result<()> {
        let mut ctx = ctx.clone();
        let stmts = body.stmts();
        for (at, stmt) in stmts.iter().enumerate() {
            let text = statement_text(stmt).unwrap_or_else(|_| stmt.ty().name().to_string());
            self.statement = text.lines().next().unwrap_or_default().to_string();
            self.b.comment(self.statement.clone());
            self.rule(stmt, &stmts[at + 1..], &mut ctx)?;
        }
        Ok(())
    }

    /// `body` as a scope of its own: the names it binds leave with it.
    fn scope(&mut self, body: &Body, ctx: &SimplifyCtx) -> Result<()> {
        self.names.push(HashMap::new());
        let done = self.body(body, ctx);
        self.names.pop();
        done
    }

    fn rule(
        &mut self,
        stmt: &Arc<Statement>,
        rest: &[Arc<Statement>],
        ctx: &mut SimplifyCtx,
    ) -> Result<()> {
        match &**stmt {
            Statement::Grid(grid) => self.grid(stmt, grid, ctx),
            Statement::Loop(node) => self.loop_(stmt, node, ctx),
            Statement::StridedLoop(node) => self.strided_loop(stmt, node, ctx),
            Statement::Cond(cond) => self.cond(cond, rest, ctx),
            Statement::Load(load) => self.load(load, ctx),
            Statement::Write(write) => self.write(write, ctx),
            Statement::Assign(assign) => self.assign(assign, rest, ctx),
            Statement::Let(binding) => self.let_(binding, ctx),
            Statement::Select(select) => self.select(select, ctx),
            Statement::Accum(accum) => self.accum(accum, ctx),
            Statement::Init(init) => self.init_(init),
            Statement::Carry(carry) => self.carry(carry),
            Statement::Pre(pre) => self.pre(pre),
            Statement::ZeroPrologue(zero) => self.zero_prologue(zero, ctx),
            Statement::Smem(smem) => self.smem(smem),
            Statement::CpAsyncCopy(copy) => self.cp_async(copy, ctx),
            Statement::CpAsyncCommit(_) => self.cp_async_group(Opcode::CpAsyncCommit, None),
            Statement::CpAsyncWait(wait) => {
                self.cp_async_group(Opcode::CpAsyncWait, Some(wait.group))
            }
            Statement::Sync(sync) => self.sync(sync),
            Statement::WarpShuffle(shuffle) => self.warp_shuffle(shuffle, ctx),
            Statement::WarpBroadcast(broadcast) => self.warp_broadcast(broadcast),
            Statement::TreeHalve(tree) => self.tree_halve(tree, ctx),
            Statement::RegFragment(fragment) => self.reg_fragment(fragment),
            Statement::LdmatrixLoad(load) => self.ldmatrix(load, ctx),
            Statement::MmaSyncPtx(mma) => self.mma_sync(mma),
            Statement::FragmentPromote(promote) => self.fragment_promote(promote, ctx),
            Statement::FragmentRepack(repack) => self.fragment_repack(repack, ctx),
            Statement::BlockScaleLoad(load) => self.block_scale_load(load, ctx),
            Statement::WgmmaDescriptor(descriptor) => self.wgmma_descriptor(descriptor, ctx),
            Statement::WgmmaMma(mma) => self.wgmma(mma),
            Statement::WgmmaFence(_) => self.wgmma_group(Opcode::WgmmaFence, None),
            Statement::WgmmaCommit(_) => self.wgmma_group(Opcode::WgmmaCommit, None),
            Statement::WgmmaWait(wait) => self.wgmma_group(Opcode::WgmmaWait, Some(wait.group)),
            Statement::TmaDescriptor(descriptor) => self.tma_descriptor(descriptor),
            Statement::TmaLoad(load) => self.tma_load(load, ctx),
            Statement::MbarrierInit(init) => self.mbarrier_init(init, ctx),
            Statement::MbarrierArriveExpectTx(arrive) => {
                self.mbarrier_arrive(arrive.mbar, &arrive.slot, Some(arrive.bytes_), ctx)
            }
            Statement::MbarrierArrive(arrive) => {
                self.mbarrier_arrive(arrive.mbar, &arrive.slot, None, ctx)
            }
            Statement::MbarrierWait(wait) => self.mbarrier_wait(wait, ctx),
            Statement::SetMaxNReg(set) => self.set_max_registers(set),
            Statement::FragmentLoad(load) => self.fragment_gather(load, ctx),
            Statement::RegStore(store) => self.reg_store(store, ctx),
            Statement::FragmentApply(apply) => self.fragment_apply(apply, ctx),
            Statement::FragmentRowReduce(reduce) => self.fragment_row_reduce(reduce),
            other => Err(self.no_rule(format_args!("the statement type {}", other.ty()))),
        }
    }

    /// A grid binds each axis to its decode over the built-ins, under the launch's ranges, and runs its body once.
    fn grid(&mut self, stmt: &Statement, grid: &Grid, ctx: &SimplifyCtx) -> Result<()> {
        if grid.axes.len() != grid.decode.len() {
            return Err(self.refuse("a grid decodes every axis"));
        }
        let threads = grid.block_threads + grid.aux_threads;
        if threads != self.threads.count() as i64 {
            let bound = self.threads.count();
            return Err(self.refuse(format_args!(
                "a grid of {threads} threads a block under an interface of {bound}"
            )));
        }
        let launch = launch_ctx(ctx, stmt).unwrap_or_else(|| ctx.clone());
        self.names.push(HashMap::new());
        self.grid_depth = self.names.len();
        let done = (|| {
            for (axis, decode) in grid.axes.iter().zip(grid.decode.iter()) {
                let val = self.int(decode, &launch)?;
                self.note_by_thread(&val, decode);
                self.bind(axis.name, val);
            }
            self.lane_coordinates(&grid.body, &launch)?;
            self.body(&grid.body, &inner_ctx(&launch, stmt))
        })();
        self.names.pop();
        done
    }

    fn loop_(&mut self, stmt: &Statement, node: &Loop, ctx: &SimplifyCtx) -> Result<()> {
        if node.unroll {
            return Err(self.refuse("a loop asks to be unrolled, which a kernel pass does"));
        }
        if node.seed {
            self.seed(&node.body)?;
        }
        let carried: Vec<Name> = node.carries().into_iter().map(|(name, _)| name).collect();
        self.declare_carried(&carried, &node.body, ctx)?;
        let (end, enters) = match node.axis.extent.as_static() {
            Some(extent) if extent <= 0 => return Ok(()),
            Some(extent) => (Operand::Int(extent), true),
            None => {
                let extent = self.int(&node.axis.extent.expression, ctx)?;
                (self.as_int(&extent, VTy::I32), false)
            }
        };
        let uneven = self.differs_by_thread(&[&node.axis.extent.expression]);
        let open = self
            .b
            .open_loop(Operand::Int(0), end, Operand::Int(1), enters);
        self.names.push(HashMap::new());
        self.bind(node.axis.name, Val::reg(open.var, VTy::I32));
        self.threads.uneven += usize::from(uneven);
        let done = self.body(&node.body, &inner_ctx(ctx, stmt));
        self.threads.uneven -= usize::from(uneven);
        self.names.pop();
        done?;
        self.commit_carried(&carried);
        self.b.close_loop(open)
    }

    /// A strided loop with a constant start and step over a static extent runs the same trips in every thread, so
    /// it is a counter from zero to a constant with the axis computed from it; any other steps the axis itself.
    fn strided_loop(
        &mut self,
        stmt: &Statement,
        node: &StridedLoop,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        if node.unroll {
            return Err(self.refuse("a loop asks to be unrolled, which a kernel pass does"));
        }
        if node.seed {
            self.seed(&node.body)?;
        }
        let inner = inner_ctx(ctx, stmt);
        let mut uneven = false;
        let uniform = match (
            node.start.int_literal(),
            node.step.int_literal(),
            node.axis.extent.as_static(),
            &node.end,
        ) {
            (Some(start), Some(step), Some(extent), None) if step > 0 => {
                Some((start, step, extent))
            }
            _ => None,
        };
        self.names.push(HashMap::new());
        let open = match uniform {
            Some((start, step, extent)) => {
                let trips = (extent - start + step - 1).div_euclid(step);
                if trips <= 0 {
                    self.names.pop();
                    return Ok(());
                }
                let open =
                    self.b
                        .open_loop(Operand::Int(0), Operand::Int(trips), Operand::Int(1), true);
                let axis = self.scaled_counter(open.var, step, start);
                self.bind(node.axis.name, axis);
                open
            }
            None => {
                let start = self.int(&node.start, ctx)?;
                let step = self.int(&node.step, ctx)?;
                let last = node.end.as_ref().unwrap_or(&node.axis.extent.expression);
                let end = self.int(last, ctx)?;
                uneven = self.differs_by_thread(&[&node.start, &node.step, last]);
                let multiple = int::gcd(self.multiple(&start.op), self.multiple(&step.op));
                let (start, step, end) = (
                    self.as_int(&start, VTy::I32),
                    self.as_int(&step, VTy::I32),
                    self.as_int(&end, VTy::I32),
                );
                let open = self.b.open_loop(start, end, step, false);
                self.b.set_multiple(open.var, multiple);
                self.bind(node.axis.name, Val::reg(open.var, VTy::I32));
                open
            }
        };
        self.threads.uneven += usize::from(uneven);
        let done = self.body(&node.body, &inner);
        self.threads.uneven -= usize::from(uneven);
        self.names.pop();
        done?;
        self.b.close_loop(open)
    }

    /// A condition runs each side where its predicate holds or does not. What is decided of the threads that
    /// reach each side is kept for the barriers inside; the last condition of a grid's body with no other side is
    /// the grid's mask, and the threads it turns away have left the kernel.
    fn cond(&mut self, cond: &Cond, rest: &[Arc<Statement>], ctx: &SimplifyCtx) -> Result<()> {
        // A condition the body's optimization decided and left standing: its one side is the statement.
        if let Expr::Literal(decided) = &*cond.cond {
            let side = if decided.value.truthy() {
                &cond.body
            } else {
                &cond.else_body
            };
            return self.scope(side, ctx);
        }
        let pred = self.pred(&cond.cond, ctx)?;
        let passes = self.truth_by_thread(&cond.cond);
        let mask =
            rest.is_empty() && cond.else_body.is_empty() && self.names.len() == self.grid_depth;
        let around = self.threads.clone();
        self.threads.enter(passes.as_deref(), true, mask);
        let mut open = self.b.open_if(pred);
        self.scope(&cond.body, ctx)?;
        if !cond.else_body.is_empty() {
            self.threads = around.clone();
            self.threads.enter(passes.as_deref(), false, false);
            self.b.else_(&mut open)?;
            self.scope(&cond.else_body, ctx)?;
        }
        self.threads = around;
        self.b.close_if(open)
    }

    /// `val` in a register: a constant is moved into one.
    fn register(&mut self, val: &Val) -> Reg {
        match val.op {
            Operand::Reg(reg) => reg,
            _ => {
                let inst = Inst::new(Opcode::Mov, val.ty.held(), Vec::new(), vec![val.op.clone()]);
                self.b.value(inst, val.ty.bank())
            }
        }
    }

    /// The value of `expression` at `ty`.
    fn typed(&mut self, expression: &Expression, ty: VTy, ctx: &SimplifyCtx) -> Result<Val> {
        match ty {
            VTy::Bool => Ok(Val::reg(self.pred(expression, ctx)?, VTy::Bool)),
            ty if ty.is_byte_float() => self.byte_float(expression, ty, ctx),
            ty if ty.is_float() => Ok(Val {
                op: self.float(expression, ty, ctx)?,
                ty,
            }),
            ty => {
                let val = self.int(expression, ctx)?;
                Ok(Val {
                    op: self.as_int(&val, ty),
                    ty,
                })
            }
        }
    }

    /// Bind `name` to `value` at `dtype`. An integer's range joins the context, so the expressions that read the
    /// name are sized by it.
    fn define(
        &mut self,
        name: Name,
        value: &Expression,
        dtype: Option<DType>,
        ctx: &mut SimplifyCtx,
    ) -> Result<()> {
        let ty = match dtype {
            Some(dtype) => self.vty(Some(dtype))?,
            None => self
                .infer(value)
                .ok_or_else(|| self.refuse("a value carries no dtype"))?,
        };
        let val = self.typed(value, ty, ctx)?;
        if ty.is_int()
            && let Some(range) = value.range(ctx)
        {
            ctx.ranges.insert(name, range);
        }
        self.note_by_thread(&val, value);
        let val = self.named(val);
        self.bind(name, val);
        Ok(())
    }

    /// Whether the product `name` is read once, by a sibling that adds it, subtracts with it or accumulates it at
    /// its own type: the multiply the kernel's fusion decision makes one instruction with its reader.
    fn fuses(&self, name: Name, dtype: Option<DType>, rest: &[Arc<Statement>]) -> bool {
        let reads = |e: &Expression| e.as_var() == Some(&name);
        self.uses.get(&name) == Some(&1)
            && rest
                .iter()
                .find(|stmt| stmt.deps().contains(&name))
                .is_some_and(|reader| match &**reader {
                    Statement::Assign(assign) => {
                        assign.dtype == dtype
                            && matches!(
                                assign.value.operation(),
                                Some((Operation::Add | Operation::Subtract, [x, y]))
                                    if reads(x) != reads(y)
                            )
                    }
                    Statement::Accum(accum) => {
                        accum.dtype == dtype
                            && reads(&accum.value)
                            && matches!(accum.op.known(), Some(Operation::Add | Operation::Sum))
                    }
                    _ => false,
                })
    }

    fn assign(
        &mut self,
        assign: &Assign,
        rest: &[Arc<Statement>],
        ctx: &mut SimplifyCtx,
    ) -> Result<()> {
        if self.precision.fuse
            && let Some(ty) = assign.dtype.and_then(VTy::of)
            && matches!(ty, VTy::F32 | VTy::F16)
            && let Some((Operation::Multiply, [x, y])) = assign.value.operation()
            && self.fuses(assign.name, assign.dtype, rest)
        {
            let (x, y) = (self.float(x, ty, ctx)?, self.float(y, ty, ctx)?);
            let steps = |operand: &Operand| operand.reg().is_some_and(|reg| self.b.is_state(reg));
            if steps(&x) || steps(&y) {
                let product = self.float_inst(Opcode::Mul, ty, vec![x, y]);
                let val = Val::reg(self.b.value(product, ty.bank()), ty);
                self.bind(assign.name, val);
            } else {
                self.products.insert(assign.name, (ty, x, y));
            }
            return Ok(());
        }
        self.define(assign.name, &assign.value, assign.dtype, ctx)
    }

    fn let_(&mut self, binding: &Let, ctx: &mut SimplifyCtx) -> Result<()> {
        self.define(binding.name, &binding.value, binding.dtype, ctx)
    }

    /// A select is the value of its first branch whose predicate holds, the last branch where none does.
    fn select(&mut self, select: &Select, ctx: &SimplifyCtx) -> Result<()> {
        let Some((last, earlier)) = select.branches.split_last() else {
            return Err(self.refuse("a select has a branch"));
        };
        let mut result = self.lookup(last.value)?;
        let ty = result.ty;
        if ty == VTy::Bool {
            return Err(self.no_rule("a select of bools"));
        }
        for branch in earlier.iter().rev() {
            let value = self.lookup(branch.value)?;
            if value.ty != ty {
                let (name, held) = (branch.value, value.ty.text());
                return Err(self.refuse(format_args!("{name} is {held}, not {}", ty.text())));
            }
            let pred = self.pred(&branch.select, ctx)?;
            let operands = vec![value.op, result.op, pred.into()];
            let inst = Inst::new(Opcode::Selp, ty.held(), Vec::new(), operands);
            result = Val::reg(self.b.value(inst, ty.bank()), ty);
        }
        let result = self.named(result);
        self.bind(select.name, result);
        Ok(())
    }

    fn init_(&mut self, init: &Init) -> Result<()> {
        let ty = self.vty(Some(init.dtype))?;
        self.init(init.name, ty, init.identity.0)
    }

    /// One step of an accumulator: its register is defined again from its own value, or from `base`, and the
    /// statement's value.
    fn accum(&mut self, accum: &Accum, ctx: &SimplifyCtx) -> Result<()> {
        let ty = self.vty(accum.dtype)?;
        let state = match self.find(accum.name) {
            Some((_, val)) if val.ty == ty => val.op.reg().filter(|reg| self.b.is_state(*reg)),
            _ => None,
        };
        let Some(state) = state else {
            let name = accum.name;
            return Err(self.refuse(format_args!("the accumulator {name} has no seed in scope")));
        };
        let left = match accum.base {
            Some(base) => {
                let val = self.lookup(base)?;
                if val.ty != ty {
                    let held = val.ty.text();
                    return Err(self.refuse(format_args!("{base} is {held}, not {}", ty.text())));
                }
                val
            }
            None => Val::reg(state, ty),
        };
        let opcode = match accum.op.known() {
            Some(Operation::Add | Operation::Sum) => Opcode::Add,
            Some(Operation::Multiply | Operation::Prod) => Opcode::Mul,
            Some(Operation::Maximum | Operation::Amax | Operation::Fmax) => Opcode::Max,
            Some(Operation::Minimum | Operation::Fmin) => Opcode::Min,
            _ => return Err(self.no_rule(format_args!("an accumulator of {}", accum.op.name()))),
        };
        let step = if ty.is_float() {
            self.float_supported(opcode, ty)?;
            let product = match opcode {
                Opcode::Add => self.product(&accum.value, ty, ctx)?,
                _ => None,
            };
            match product {
                Some((a, b)) => self.float_inst(Opcode::Fma, ty, vec![a, b, left.op]),
                None => {
                    let value = self.float(&accum.value, ty, ctx)?;
                    self.float_inst(opcode, ty, vec![left.op, value])
                }
            }
        } else if ty.is_int() {
            let value = self.int(&accum.value, ctx)?;
            let operands = vec![self.as_int(&left, ty), self.as_int(&value, ty)];
            Inst::new(opcode, ty.arith(), Vec::new(), operands)
        } else {
            return Err(self.no_rule("an accumulator of bools"));
        };
        self.b.emit(Inst {
            results: vec![state],
            ..step
        });
        Ok(())
    }
}
