//! The rules: one per statement type, each writing the statement's instructions through the builder. The input is
//! taken as typed and flat: a value without a dtype, an operand of another type than its reader, or an index of
//! several coordinates is refused by the statement's name, and so is a statement or an operation with no rule yet.

mod int;
mod value;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expression, SimplifyCtx};
use emmy_ir::normalize::{inner_ctx, launch_ctx};
use emmy_ir::statement::{
    Accum, Assign, Cond, Grid, Init, Let, Load, Loop, Select, Statement, StridedLoop, Write,
};
use emmy_ir::text::statement_text;
use emmy_ir::{Error, Name, Result};

use crate::builder::Builder;
use crate::inst::{Bank, Base, Inst, Module, Opcode, Operand, ParamDecl, Reg, Space, Ty};
use crate::{Kernel, ParamKind, Precision, Target};

/// The most threads a block can hold.
const MAX_BLOCK_THREADS: u32 = 1024;

/// The first compute capability with a half-precision minimum and maximum.
const SM_HALF_MIN_MAX: u32 = 80;

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
            _ => return None,
        })
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

/// A buffer parameter: the register its pointer was loaded into and its element type.
struct Buffer {
    base: Reg,
    ty: VTy,
    writable: bool,
}

pub(crate) struct Lowerer {
    b: Builder,
    target: Target,
    precision: Precision,
    buffers: HashMap<Name, Buffer>,
    /// The names in scope, one table per open body, the innermost last.
    names: Vec<HashMap<Name, Val>>,
    /// How many statements read each name.
    uses: HashMap<Name, u32>,
    /// The products held back for the one add, subtract or accumulate that reads them: the type and the operands.
    products: HashMap<Name, (VTy, Operand, Operand)>,
    /// The first line of the statement being lowered, for an error to name.
    statement: String,
}

fn count_uses(body: &Body, uses: &mut HashMap<Name, u32>) {
    for stmt in body.stmts() {
        for name in stmt.deps() {
            *uses.entry(name).or_insert(0) += 1;
        }
        for nested in stmt.nested() {
            count_uses(&nested, uses);
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
        names: vec![HashMap::new()],
        uses: HashMap::new(),
        products: HashMap::new(),
        statement: format!("the interface of {}", kernel.name),
    };
    count_uses(kernel.body, &mut lowerer.uses);
    let params = lowerer.parameters(kernel)?;
    lowerer.body(kernel.body, &SimplifyCtx::empty())?;
    let code = lowerer.b.finish()?;
    Ok(Module {
        version: target.version(),
        target: target.text(interface.arch_specific),
        entry: kernel.name.to_string(),
        params,
        shared: Vec::new(),
        max_threads: interface.threads,
        registers: code.registers,
        items: code.items,
        alignment: code.alignment,
    })
}

impl Lowerer {
    fn no_rule(&self, what: impl std::fmt::Display) -> Error {
        Error::NotImplemented(format!("no PTX rule for {what}: `{}`", self.statement))
    }

    fn refuse(&self, why: impl std::fmt::Display) -> Error {
        Error::Value(format!("{why}: `{}`", self.statement))
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
                        let ty = self.vty(param.dtype)?;
                        let base = self.b.reg(Bank::B64);
                        let load = Opcode::Ld(Space::Param);
                        self.b
                            .emit(Inst::new(load, Ty::U64, vec![base], vec![source]));
                        let align = ty.bytes() as u32;
                        self.b.set_alignment(base, align);
                        let writable = param.kind == ParamKind::Output;
                        let buffer = Buffer { base, ty, writable };
                        self.buffers.insert(param.name, buffer);
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
                        Err(self.no_rule(format_args!("the descriptor {}", param.name)))
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

    fn bind(&mut self, name: Name, val: Val) {
        if let Some(scope) = self.names.last_mut() {
            scope.insert(name, val);
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
        self.find(name)
            .map(|(_, val)| val.clone())
            .ok_or_else(|| self.refuse(format_args!("{name} is read before it is defined")))
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

    /// A new state named `name`, holding `value`.
    fn init(&mut self, name: Name, ty: VTy, value: f64) -> Result<()> {
        let constant = match ty {
            VTy::F32 => Operand::f32(value as f32),
            VTy::F16 | VTy::Bf16 => Operand::Int(i64::from(value::half_bits(ty, value))),
            VTy::I32 | VTy::U32 | VTy::I64 if value.fract() == 0.0 => Operand::Int(value as i64),
            _ => return Err(self.no_rule(format_args!("a {} state seeded {value}", ty.text()))),
        };
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
            Statement::Cond(cond) => self.cond(cond, ctx),
            Statement::Load(load) => self.load(load, ctx),
            Statement::Write(write) => self.write(write, ctx),
            Statement::Assign(assign) => self.assign(assign, rest, ctx),
            Statement::Let(binding) => self.let_(binding, ctx),
            Statement::Select(select) => self.select(select, ctx),
            Statement::Accum(accum) => self.accum(accum, ctx),
            Statement::Init(init) => self.init_(init),
            other => Err(self.no_rule(format_args!("the statement type {}", other.ty()))),
        }
    }

    /// A grid binds each axis to its decode over the built-ins, under the launch's ranges, and runs its body once.
    fn grid(&mut self, stmt: &Statement, grid: &Grid, ctx: &SimplifyCtx) -> Result<()> {
        if grid.axes.len() != grid.decode.len() {
            return Err(self.refuse("a grid decodes every axis"));
        }
        let launch = launch_ctx(ctx, stmt).unwrap_or_else(|| ctx.clone());
        self.names.push(HashMap::new());
        let done = (|| {
            for (axis, decode) in grid.axes.iter().zip(grid.decode.iter()) {
                let val = self.int(decode, &launch)?;
                self.bind(axis.name, val);
            }
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
        let (end, enters) = match node.axis.extent.as_static() {
            Some(extent) if extent <= 0 => return Ok(()),
            Some(extent) => (Operand::Int(extent), true),
            None => {
                let extent = self.int(&node.axis.extent.expression, ctx)?;
                (self.as_int(&extent, VTy::I32), false)
            }
        };
        let open = self
            .b
            .open_loop(Operand::Int(0), end, Operand::Int(1), enters);
        self.names.push(HashMap::new());
        self.bind(node.axis.name, Val::reg(open.var, VTy::I32));
        let done = self.body(&node.body, &inner_ctx(ctx, stmt));
        self.names.pop();
        done?;
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
                let end = match &node.end {
                    Some(end) => self.int(end, ctx)?,
                    None => self.int(&node.axis.extent.expression, ctx)?,
                };
                let (start, step, end) = (
                    self.as_int(&start, VTy::I32),
                    self.as_int(&step, VTy::I32),
                    self.as_int(&end, VTy::I32),
                );
                let open = self.b.open_loop(start, end, step, false);
                self.bind(node.axis.name, Val::reg(open.var, VTy::I32));
                open
            }
        };
        let done = self.body(&node.body, &inner);
        self.names.pop();
        done?;
        self.b.close_loop(open)
    }

    fn cond(&mut self, cond: &Cond, ctx: &SimplifyCtx) -> Result<()> {
        let pred = self.pred(&cond.cond, ctx)?;
        let mut open = self.b.open_if(pred);
        self.scope(&cond.body, ctx)?;
        if !cond.else_body.is_empty() {
            self.b.else_(&mut open)?;
            self.scope(&cond.else_body, ctx)?;
        }
        self.b.close_if(open)
    }

    /// The pointer register, the element type and whether the kernel may write it, of the buffer `name`.
    fn buffer(&self, name: Name) -> Result<(Reg, VTy, bool)> {
        self.buffers
            .get(&name)
            .map(|buffer| (buffer.base, buffer.ty, buffer.writable))
            .ok_or_else(|| self.refuse(format_args!("{name} is no buffer of the interface")))
    }

    fn load(&mut self, load: &Load, ctx: &SimplifyCtx) -> Result<()> {
        let [name] = &load.names[..] else {
            return Err(self.no_rule("a vector load"));
        };
        if !load.carried.is_empty() {
            return Err(self.no_rule("a load of a staged copy"));
        }
        let ty = self.vty(load.dtype)?;
        let (_, stored, _) = self.buffer(load.input)?;
        if stored != ty {
            let stored = stored.text();
            return Err(self.refuse(format_args!("{} holds {stored}", load.input)));
        }
        let address = self.address(load.input, &load.index, ctx)?;
        // A bool is a byte in memory and a predicate in a register.
        let (held, bank) = match ty {
            VTy::Bool => (Ty::U8, Bank::B32),
            ty => (ty.held(), ty.bank()),
        };
        let reg = self.b.reg(bank);
        let opcode = Opcode::Ld(Space::Global);
        self.b
            .emit(Inst::new(opcode, held, vec![reg], vec![address]));
        let val = match ty {
            VTy::Bool => Val::reg(self.truth(&Val::reg(reg, VTy::U32))?, VTy::Bool),
            ty => Val::reg(reg, ty),
        };
        self.bind(*name, val);
        Ok(())
    }

    fn write(&mut self, write: &Write, ctx: &SimplifyCtx) -> Result<()> {
        let [name] = &write.values[..] else {
            return Err(self.no_rule("a vector store"));
        };
        if write.atomic {
            return Err(self.no_rule("an atomic store"));
        }
        if &*write.swizzle != "NONE" {
            return Err(self.no_rule("a swizzled store"));
        }
        let val = self.lookup(*name)?;
        let (_, stored, writable) = self.buffer(write.output)?;
        if !writable {
            return Err(self.refuse(format_args!("{} is an input", write.output)));
        }
        let declared = write.value_dtype.map(|dtype| self.vty(Some(dtype)));
        if stored != val.ty || declared.transpose()?.is_some_and(|ty| ty != stored) {
            let (stored, held) = (stored.text(), val.ty.text());
            let output = write.output;
            return Err(self.refuse(format_args!("{output} holds {stored}, {name} is {held}")));
        }
        let address = self.address(write.output, &write.index, ctx)?;
        let (held, value) = match val.ty {
            VTy::Bool => (Ty::U8, self.flag(&val)?.op),
            ty => (ty.held(), self.register(&val).into()),
        };
        let opcode = Opcode::St(Space::Global);
        self.b
            .emit(Inst::new(opcode, held, Vec::new(), vec![address, value]));
        Ok(())
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
