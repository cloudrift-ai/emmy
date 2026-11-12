//! Memory: the buffers a kernel reaches (its parameters and its shared declarations), loads and stores of one
//! element or of several at once, atomic adds, async copies into shared memory, the zeroing one kernel does for
//! another, and the cells of a carried state.

use std::collections::BTreeMap;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Builtin, Expr, Expression, Interval, SimplifyCtx};
use emmy_ir::field::NumOr;
use emmy_ir::statement::{Carry, CpAsyncCopy, Load, Pre, Smem, Statement, Write, ZeroPrologue};
use emmy_ir::{Name, Result};

use super::{Lowerer, POOL_ALIGN, SM_AMPERE, VTy, Val};
use crate::inst::{Bank, Base, Cache, Inst, Opcode, Operand, Reg, Shared, Space, Ty};

/// The most bytes the shared arrays of a module may take together; past it the launch sizes one pool.
const STATIC_SHARED_BYTES: u64 = 48 * 1024;

/// The most 32-bit words one kernel zeroes for another: twice as many still fit a signed 32-bit index.
const MAX_ZEROED_WORDS: i64 = 1 << 30;

/// How one element of a buffer sits in memory and in a register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Elem {
    pub dtype: DType,
    /// The type of the value a load gives.
    pub ty: VTy,
    /// The type a load or a store of one element takes.
    pub mem: Ty,
}

impl Elem {
    /// A byte is an integer in a register; a bool is a byte in memory and a predicate in a register.
    pub(super) fn of(dtype: DType) -> Option<Elem> {
        let (ty, mem) = match dtype {
            DType::U8 => (VTy::I32, Ty::U8),
            DType::Bool => (VTy::Bool, Ty::U8),
            dtype => {
                let ty = VTy::of(dtype)?;
                (ty, ty.held())
            }
        };
        Some(Elem { dtype, ty, mem })
    }

    pub(super) fn bytes(self) -> i64 {
        self.mem.bytes()
    }

    fn is_byte(self) -> bool {
        self.dtype == DType::U8
    }
}

/// A buffer: a parameter in global memory, or a shared declaration.
#[derive(Clone, Copy)]
pub(super) struct Buffer {
    pub space: Space,
    /// The register that holds its address: 64 bits in global memory, 32 in shared.
    pub base: Reg,
    /// Where it starts from that address, in bytes: its place in the shared pool.
    pub offset: i64,
    pub elem: Elem,
    pub writable: bool,
}

impl Buffer {
    /// A load of global memory the kernel never writes goes past the coherent cache, as nvcc sends a load through
    /// a `const` pointer it was promised nothing else writes.
    fn load(self) -> Opcode {
        match (self.space, self.writable) {
            (Space::Global, false) => Opcode::LdConst,
            (space, _) => Opcode::Ld(space),
        }
    }
}

/// How an access of several elements is made: by one instruction of that many registers, or as 32-bit words that
/// hold `each` elements apiece.
#[derive(Clone, Copy)]
enum Access {
    Direct,
    Words { words: usize, each: usize },
}

/// The access of `lanes` elements of `bytes` each: 4, 8 or 16 bytes at once, as the hardware loads and stores.
fn access(bytes: i64, lanes: usize) -> Option<Access> {
    match (bytes, lanes) {
        (_, 1) | (4 | 8, 2) | (4, 4) | (2, 2 | 4) | (1, 2 | 4) => Some(Access::Direct),
        (2, 8) => Some(Access::Words { words: 4, each: 2 }),
        (1, 8) => Some(Access::Words { words: 2, each: 4 }),
        (1, 16) => Some(Access::Words { words: 4, each: 4 }),
        _ => None,
    }
}

/// The element type a shared declaration names: a dtype, or the C type the CUDA renderer declares the array with.
/// `unsigned char` is not read as a byte: the renderer declares a slab of packed fp4 pairs with it.
fn shared_dtype(name: &str) -> Option<DType> {
    name.parse().ok().or(match name {
        "float" => Some(DType::F32),
        "__half" => Some(DType::F16),
        "__nv_bfloat16" => Some(DType::BF16),
        "int" => Some(DType::I32),
        "unsigned int" => Some(DType::U32),
        "long long" => Some(DType::I64),
        "bool" => Some(DType::Bool),
        _ => None,
    })
}

/// A carried state: its type, what every cell holds before the first step, and per cell the register a read sees
/// and the register the step defines.
pub(super) struct Carried {
    ty: VTy,
    seed: NumOr<Name>,
    cells: BTreeMap<Vec<i64>, (Reg, Reg)>,
}

impl Lowerer {
    pub(super) fn buffer(&self, name: Name) -> Result<Buffer> {
        self.buffers
            .get(&name)
            .copied()
            .ok_or_else(|| self.refuse(format_args!("{name} is no buffer of the kernel")))
    }

    /// An error where the target is older than `sm`, which `what` needs.
    fn needs(&self, sm: u32, what: &str) -> Result<()> {
        if self.target.sm >= sm {
            return Ok(());
        }
        Err(self.no_rule(format_args!("{what} on sm_{}", self.target.sm)))
    }

    // --- shared memory -----------------------------------------------------------------------------------

    /// The alignment and the size in bytes of a shared declaration.
    fn shared_extent(&self, smem: &Smem) -> Result<(Elem, u32, u64)> {
        let dtype = shared_dtype(&smem.dtype)
            .ok_or_else(|| self.no_rule(format_args!("a shared array of {}", smem.dtype)))?;
        let elem = self.elem(Some(dtype))?;
        let elements: i64 = smem.extents.iter().product();
        let align = smem.align.max(elem.bytes());
        if elements < 0 || !(1..=1 << 16).contains(&align) || align & (align - 1) != 0 {
            return Err(self.refuse("a shared array has a size and a power of two as alignment"));
        }
        Ok((elem, align as u32, (elements * elem.bytes()) as u64))
    }

    /// Declare the launch-sized pool every shared buffer of `body` has its place in. It is aligned to the strictest
    /// of them.
    pub(super) fn declare_pool(&mut self, body: &Body) -> Result<()> {
        let mut align = POOL_ALIGN;
        for stmt in body.iter_stmts() {
            if let Statement::Smem(smem) = &*stmt {
                align = align.max(self.shared_extent(smem)?.1);
            }
        }
        self.shared.push(Shared { align, bytes: None });
        Ok(())
    }

    /// A shared buffer: an array of the module, or its place in the pool. Either way its address is the symbol's
    /// own, read where the buffer is declared, so nothing assumes where the block's shared memory starts.
    pub(super) fn smem(&mut self, smem: &Smem) -> Result<()> {
        let (elem, align, bytes) = self.shared_extent(smem)?;
        let (index, offset) = if self.pool.is_empty() {
            let taken: u64 = self.shared.iter().filter_map(|shared| shared.bytes).sum();
            if taken + bytes > STATIC_SHARED_BYTES {
                return Err(self.refuse(format_args!(
                    "{} bytes of shared arrays pass the {STATIC_SHARED_BYTES} a module may declare, and the \
                     interface gives no pool",
                    taken + bytes
                )));
            }
            self.shared.push(Shared {
                align,
                bytes: Some(bytes),
            });
            (self.shared.len() - 1, 0)
        } else {
            let offset = self.pool.get(&smem.name).copied().ok_or_else(|| {
                self.refuse(format_args!(
                    "{} has no place in the interface's pool",
                    smem.name
                ))
            })?;
            if offset % u64::from(align) != 0 {
                return Err(self.refuse(format_args!(
                    "{} stands at byte {offset} of the pool, off its alignment of {align}",
                    smem.name
                )));
            }
            (0, offset as i64)
        };
        let symbol = Operand::Shared(index as u32);
        let address = Inst::new(Opcode::Mov, Ty::U32, Vec::new(), vec![symbol]);
        let base = self.b.value(address, Bank::B32);
        self.b.set_alignment(base, self.shared[index].align);
        let buffer = Buffer {
            space: Space::Shared,
            base,
            offset,
            elem,
            writable: true,
        };
        self.buffers.insert(smem.name, buffer);
        Ok(())
    }

    // --- loads and stores --------------------------------------------------------------------------------

    /// The elements of `word`, lowest first, into `parts`: two halves by one move, four bytes by shifts and masks.
    fn unpack(&mut self, word: Reg, parts: &[Reg]) {
        if parts.len() == 2 {
            let split = Inst::new(Opcode::Mov, Ty::B32, parts.to_vec(), vec![word.into()]);
            return self.b.emit(split);
        }
        for (at, part) in parts.iter().enumerate() {
            let shifted = match at {
                0 => word,
                _ => {
                    let operands = vec![word.into(), Operand::Int(8 * at as i64)];
                    let shift = Inst::new(Opcode::Shr, Ty::U32, Vec::new(), operands);
                    self.b.value(shift, Bank::B32)
                }
            };
            let operands = vec![shifted.into(), Operand::Int(0xff)];
            self.b
                .emit(Inst::new(Opcode::And, Ty::B32, vec![*part], operands));
        }
    }

    /// One 32-bit word of `parts`, lowest first: two halves by one move, four bytes by masks, shifts and sums.
    fn pack(&mut self, parts: &[Reg]) -> Reg {
        if parts.len() == 2 {
            let halves = Operand::Tuple(parts.to_vec());
            let join = Inst::new(Opcode::Mov, Ty::B32, Vec::new(), vec![halves]);
            return self.b.value(join, Bank::B32);
        }
        let mut value = |opcode, left: Reg, right: i64| {
            let operands = vec![left.into(), Operand::Int(right)];
            self.b
                .value(Inst::new(opcode, Ty::B32, Vec::new(), operands), Bank::B32)
        };
        let bytes: Vec<Reg> = parts
            .iter()
            .enumerate()
            .map(|(at, part)| {
                let byte = value(Opcode::And, *part, 0xff);
                match at {
                    0 => byte,
                    _ => value(Opcode::Shl, byte, 8 * at as i64),
                }
            })
            .collect();
        let mut word = bytes[0];
        for byte in &bytes[1..] {
            let operands = vec![word.into(), (*byte).into()];
            let join = Inst::new(Opcode::Or, Ty::B32, Vec::new(), operands);
            word = self.b.value(join, Bank::B32);
        }
        word
    }

    /// A load of one element, or of several in a row by one access. A staged copy is loaded again into the
    /// registers its first load declared.
    pub(super) fn load(&mut self, load: &Load, ctx: &mut SimplifyCtx) -> Result<()> {
        let buffer = self.buffer(load.input)?;
        let elem = buffer.elem;
        let dtype = load
            .dtype
            .ok_or_else(|| self.refuse("a value carries no dtype"))?;
        if dtype != elem.dtype {
            let (input, stored) = (load.input, elem.dtype);
            return Err(self.refuse(format_args!("{input} holds {stored}")));
        }
        let lanes = load.names.len();
        let again = !load.carried.is_empty();
        let access = access(elem.bytes(), lanes)
            .filter(|_| elem.ty != VTy::Bool || (lanes == 1 && !again))
            .ok_or_else(|| self.no_rule(format_args!("a load of {lanes} elements of {dtype}")))?;
        let address = self.address(load.input, &load.index, ctx)?;
        let mut regs = Vec::with_capacity(lanes);
        for name in load.names.iter() {
            regs.push(match (again, elem.ty) {
                (true, ty) => self.state(*name, ty)?,
                (false, VTy::Bool) => self.b.reg(Bank::B32),
                (false, ty) => self.fresh(*name, ty.bank()),
            });
        }
        match access {
            Access::Direct => {
                let inst = Inst::new(buffer.load(), elem.mem, regs.clone(), vec![address]);
                self.b.emit(inst);
            }
            Access::Words { words, each } => {
                let packed: Vec<Reg> = (0..words).map(|_| self.b.reg(Bank::B32)).collect();
                let inst = Inst::new(buffer.load(), Ty::B32, packed.clone(), vec![address]);
                self.b.emit(inst);
                for (word, parts) in packed.iter().zip(regs.chunks(each)) {
                    self.unpack(*word, parts);
                }
            }
        }
        for (name, reg) in load.names.iter().zip(regs) {
            if elem.is_byte() {
                ctx.ranges.insert(*name, Interval::new(0, 0xff));
            }
            if again {
                continue;
            }
            let val = match elem.ty {
                VTy::Bool => Val::reg(self.truth(&Val::reg(reg, VTy::U32))?, VTy::Bool),
                ty => Val::reg(reg, ty),
            };
            self.bind(*name, val);
        }
        Ok(())
    }

    /// A store of one value, or of several in a row by one access; an atomic store adds its value to the cell.
    pub(super) fn write(&mut self, write: &Write, ctx: &SimplifyCtx) -> Result<()> {
        if &*write.swizzle != "NONE" {
            return Err(self.no_rule("a swizzled store"));
        }
        let buffer = self.buffer(write.output)?;
        let (elem, output) = (buffer.elem, write.output);
        if !buffer.writable {
            return Err(self.refuse(format_args!("{output} is an input")));
        }
        let lanes = write.values.len();
        let access = access(elem.bytes(), lanes)
            .filter(|_| !write.atomic || lanes == 1)
            .ok_or_else(|| {
                let stored = elem.dtype;
                self.no_rule(format_args!("a store of {lanes} elements of {stored}"))
            })?;
        let mut vals = Vec::with_capacity(lanes);
        for name in write.values.iter() {
            let val = self.lookup(*name)?;
            // A byte is stored from an integer of any width; every other value has its buffer's type.
            let fits = val.ty == elem.ty || (elem.is_byte() && val.ty.is_int());
            if !fits || write.value_dtype.is_some_and(|dtype| dtype != elem.dtype) {
                let (stored, held) = (elem.dtype, val.ty.text());
                return Err(self.refuse(format_args!("{output} holds {stored}, {name} is {held}")));
            }
            vals.push(val);
        }
        let address = self.address(output, &write.index, ctx)?;
        let mut regs = Vec::with_capacity(lanes);
        for val in &vals {
            let val = match val.ty {
                VTy::Bool => self.flag(val)?,
                _ if elem.is_byte() => Val {
                    op: self.as_int(val, VTy::I32),
                    ty: VTy::I32,
                },
                _ => val.clone(),
            };
            regs.push(self.register(&val));
        }
        if write.atomic {
            let ty = match elem.dtype {
                DType::F32 | DType::I32 | DType::U32 => elem.ty.arith(),
                DType::I64 => Ty::U64,
                stored => return Err(self.no_rule(format_args!("an atomic add of {stored}"))),
            };
            let held = self.b.reg(ty.bank());
            let operands = vec![address, regs[0].into()];
            let add = Inst::new(Opcode::AtomAdd(buffer.space), ty, vec![held], operands);
            self.b.emit(add);
            return Ok(());
        }
        let (ty, value) = match access {
            Access::Direct if lanes == 1 => (elem.mem, regs[0].into()),
            Access::Direct => (elem.mem, Operand::Tuple(regs)),
            Access::Words { each, .. } => {
                let words = regs.chunks(each).map(|parts| self.pack(parts)).collect();
                (Ty::B32, Operand::Tuple(words))
            }
        };
        let store = Opcode::St(buffer.space);
        self.b
            .emit(Inst::new(store, ty, Vec::new(), vec![address, value]));
        Ok(())
    }

    /// Zero the 32-bit words of another launch's accumulator, every thread of the grid a stride of them:
    /// `for (i = thread of the grid; i < words; i += threads of the grid) dst[i] = 0`. The start and the step are
    /// held to the word count, so the index stays in 32 bits whatever the grid.
    pub(super) fn zero_prologue(&mut self, zero: &ZeroPrologue, ctx: &SimplifyCtx) -> Result<()> {
        let buffer = self.buffer(zero.dst)?;
        if !buffer.writable || buffer.space != Space::Global {
            return Err(self.refuse(format_args!("{} is no output", zero.dst)));
        }
        if !(0..=MAX_ZEROED_WORDS).contains(&zero.words) {
            return Err(self.no_rule(format_args!("a zeroing of {} words", zero.words)));
        }
        let builtin = |builtin| Expression::new(Expr::Builtin(builtin));
        let (mul, add) = (Operation::Multiply, Operation::Add);
        let block = Expression::binary(
            mul,
            builtin(Builtin::BlockIdxX),
            builtin(Builtin::BlockDimX),
        );
        let thread = Expression::binary(add, block, builtin(Builtin::ThreadIdxX));
        let grid = Expression::binary(mul, builtin(Builtin::BlockDimX), builtin(Builtin::GridDimX));
        let mut held = |value: Expression| -> Result<Operand> {
            let bounded = vec![value, Expression::int(zero.words)];
            let val = self.int(&Expression::apply(Operation::Minimum, bounded), ctx)?;
            Ok(self.as_int(&val, VTy::I32))
        };
        let (start, step) = (held(thread)?, held(grid)?);
        let nothing = self.register(&Val::int(0));
        let open = self
            .b
            .open_loop(start, Operand::Int(zero.words), step, false);
        let offset = self.scaled(&Val::reg(open.var, VTy::I32), Ty::B32.bytes());
        let operands = vec![buffer.base.into(), offset];
        let address = Inst::new(Opcode::Add, Ty::S64, Vec::new(), operands);
        let address = Operand::Addr(Base::Reg(self.b.value(address, Bank::B64)), 0);
        let store = Opcode::St(Space::Global);
        let operands = vec![address, nothing.into()];
        self.b.emit(Inst::new(store, Ty::B32, Vec::new(), operands));
        self.b.close_loop(open)
    }

    // --- async copies ------------------------------------------------------------------------------------

    /// One async copy of 4, 8 or 16 bytes from global to shared memory. Under `valid`, a 16-byte copy reads its
    /// bytes where the condition holds and writes zeros where it does not.
    pub(super) fn cp_async(&mut self, copy: &CpAsyncCopy, ctx: &SimplifyCtx) -> Result<()> {
        self.needs(SM_AMPERE, "an async copy")?;
        if &*copy.swizzle != "NONE" || copy.lane_index.is_some() {
            return Err(self.no_rule("an async copy into a swizzled or lane-indexed slab"));
        }
        let cache = match (copy.nbytes, &copy.valid) {
            (16, _) => Cache::Global,
            (4 | 8, None) => Cache::All,
            (bytes @ (4 | 8), Some(_)) => {
                return Err(self.no_rule(format_args!(
                    "an async copy of {bytes} bytes under a condition"
                )));
            }
            (bytes, _) => return Err(self.refuse(format_args!("an async copy of {bytes} bytes"))),
        };
        let (to, from) = (self.buffer(copy.smem)?, self.buffer(copy.src)?);
        if to.space != Space::Shared || from.space != Space::Global {
            return Err(self.refuse("an async copy goes from a parameter to a shared buffer"));
        }
        let to = self.address(copy.smem, &copy.smem_index, ctx)?;
        let from = self.address(copy.src, &copy.src_index, ctx)?;
        let mut operands = vec![to, from, Operand::Int(copy.nbytes)];
        if let Some(valid) = &copy.valid {
            let pred = self.pred(valid, ctx)?;
            let read = vec![Operand::Int(copy.nbytes), Operand::Int(0), pred.into()];
            let read = Inst::new(Opcode::Selp, Ty::U32, Vec::new(), read);
            operands.push(self.b.value(read, Bank::B32).into());
        }
        let copy = Inst::new(Opcode::CpAsync(cache), Ty::B32, Vec::new(), operands);
        self.b.emit(copy);
        Ok(())
    }

    /// Close the copies issued so far into a group, or wait until at most `group` groups are still in flight.
    pub(super) fn cp_async_group(&mut self, opcode: Opcode, group: Option<i64>) -> Result<()> {
        self.needs(SM_AMPERE, "an async copy")?;
        let operands = group.map(Operand::Int).into_iter().collect();
        self.b
            .emit(Inst::new(opcode, Ty::B32, Vec::new(), operands));
        Ok(())
    }

    // --- carried states ----------------------------------------------------------------------------------

    /// The cell of the carried state `name` an index of literals names.
    fn cell_index(&self, name: Name, index: &[Expression]) -> Result<Vec<i64>> {
        let literals: Option<Vec<i64>> = index.iter().map(Expression::int_literal).collect();
        literals.ok_or_else(|| {
            self.no_rule(format_args!(
                "a cell of the carried state {name} at an index known only where the kernel runs"
            ))
        })
    }

    /// The registers of one cell of the carried state `name`: the one a read sees and the one a step defines,
    /// both seeded where the cell is first met.
    fn cell(&mut self, name: Name, index: &[Expression], ctx: &SimplifyCtx) -> Result<(Reg, Reg)> {
        let key = self.cell_index(name, index)?;
        let Some(state) = self.carried.get(&name) else {
            return Err(self.refuse(format_args!("{name} is no carried state of a loop around")));
        };
        if let Some(cell) = state.cells.get(&key) {
            return Ok(*cell);
        }
        let (ty, seed) = (state.ty, state.seed.clone());
        let (now, next) = (self.b.state(ty.bank()), self.b.state(ty.bank()));
        match seed {
            NumOr::Num(value) => {
                let constant = self.seed_constant(ty, value.as_f64())?;
                self.b
                    .emit(Inst::new(Opcode::Mov, ty.held(), vec![now], vec![constant]));
            }
            NumOr::Value(seeds) => {
                let buffer = self.buffer(seeds)?;
                if buffer.elem.ty != ty || buffer.elem.mem != ty.held() {
                    let stored = buffer.elem.dtype;
                    return Err(self.refuse(format_args!("{seeds} holds {stored}")));
                }
                let address = self.address(seeds, index, ctx)?;
                self.b.emit(Inst::new(
                    buffer.load(),
                    ty.held(),
                    vec![now],
                    vec![address],
                ));
            }
        }
        self.b.emit(Inst::new(
            Opcode::Mov,
            ty.held(),
            vec![next],
            vec![now.into()],
        ));
        if let Some(state) = self.carried.get_mut(&name) {
            state.cells.insert(key, (now, next));
        }
        Ok((now, next))
    }

    /// Seed, ahead of the loop that carries them, every cell of the states `names` that the loop's `body` defines
    /// or reads.
    pub(super) fn declare_carried(
        &mut self,
        names: &[Name],
        body: &Body,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let stmts = body.iter_stmts();
        for name in names {
            let defined = stmts.iter().find_map(|stmt| match &**stmt {
                Statement::Carry(carry) if carry.name == *name => Some(carry),
                _ => None,
            });
            let Some(carry) = defined else { continue };
            let state = Carried {
                ty: self.vty(carry.dtype)?,
                seed: carry.seed.clone(),
                cells: BTreeMap::new(),
            };
            self.carried.insert(*name, state);
            for stmt in &stmts {
                match &**stmt {
                    Statement::Carry(carry) if carry.name == *name => {
                        self.cell(*name, &carry.index, ctx)?;
                    }
                    Statement::Pre(pre) if pre.carrier == *name => {
                        self.cell(*name, &pre.index, ctx)?;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// The end of a step of the loop that carries `names`: every cell a read sees takes what the step defined. A
    /// cell the step skipped keeps its value.
    pub(super) fn commit_carried(&mut self, names: &[Name]) {
        for name in names {
            let Some(state) = self.carried.get(name) else {
                continue;
            };
            let ty = state.ty.held();
            for (now, next) in state.cells.values() {
                let step = Inst::new(Opcode::Mov, ty, vec![*now], vec![(*next).into()]);
                self.b.emit(step);
            }
        }
    }

    /// The next value of one cell of a carried state, which reads see from the next step on.
    pub(super) fn carry(&mut self, carry: &Carry) -> Result<()> {
        let (_, next) = self.cell(carry.name, &carry.index, &SimplifyCtx::empty())?;
        let ty = self.carried[&carry.name].ty;
        let val = self.lookup(carry.value)?;
        if val.ty != ty {
            let (value, held) = (carry.value, val.ty.text());
            return Err(self.refuse(format_args!("{value} is {held}, not {}", ty.text())));
        }
        self.b
            .emit(Inst::new(Opcode::Mov, ty.held(), vec![next], vec![val.op]));
        Ok(())
    }

    /// A read of one cell of a carried state: what the step before left, the seed on the first.
    pub(super) fn pre(&mut self, pre: &Pre) -> Result<()> {
        let (now, _) = self.cell(pre.carrier, &pre.index, &SimplifyCtx::empty())?;
        let ty = self.carried[&pre.carrier].ty;
        let val = self.named(Val::reg(now, ty));
        self.bind(pre.name, val);
        Ok(())
    }
}
