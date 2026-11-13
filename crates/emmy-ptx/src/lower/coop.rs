//! Cooperation between the threads of a block: barriers, with what is decided of the threads that reach one; warp
//! shuffles; the tree reduction over a shared slab; and the combine program each of those runs, its states updated
//! in place.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Builtin, Expr, Expression, Num, SimplifyCtx};
use emmy_ir::statement::{Statement, Sync, TreeHalve, WarpBroadcast, WarpShuffle};
use emmy_ir::text::statement_text;
use emmy_ir::{Name, Result};

use super::{Lowerer, VTy, Val};
use crate::inst::{Bank, Inst, Opcode, Operand, Reg, Shuffle, Space, Ty};

/// The lanes of a warp.
const WARP: i64 = 32;

/// The barriers a kernel may name beside the block's own.
const NAMED_BARRIERS: std::ops::RangeInclusive<i64> = 1..=15;

/// The thread index, as the variable of an expression over it alone.
const THREAD: &str = "thread_idx.x";

/// The deepest expression over the thread index a register is remembered by.
const MAX_HEIGHT: u32 = 24;

/// What is decided of the threads of one block where a statement stands.
#[derive(Clone)]
pub(super) struct Threads {
    /// The threads that have not left the kernel.
    live: Vec<bool>,
    /// The threads every decided condition around the statement lets through.
    here: Vec<bool>,
    /// How many conditions around the statement are not decided.
    open: usize,
    /// How many loops around the statement run a trip count that differs between threads.
    pub uneven: usize,
}

impl Threads {
    pub(super) fn new(count: usize) -> Threads {
        Threads {
            live: vec![true; count],
            here: vec![true; count],
            open: 0,
            uneven: 0,
        }
    }

    /// The threads of a block.
    pub(super) fn count(&self) -> usize {
        self.live.len()
    }

    /// Enter the side of a condition that runs where its truth is `side`. `passes` is that truth per thread, where
    /// the condition is decided. The threads a grid's `mask` turns away have left the kernel.
    pub(super) fn enter(&mut self, passes: Option<&[bool]>, side: bool, mask: bool) {
        let Some(passes) = passes else {
            self.open += usize::from(!mask);
            return;
        };
        for (thread, pass) in passes.iter().enumerate() {
            self.here[thread] &= *pass == side;
            self.live[thread] &= !mask || *pass;
        }
    }

    /// The threads that may reach the statement, lowest first.
    fn reaching(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.count()).filter(|thread| self.live[*thread] && self.here[*thread])
    }
}

/// A fold over the lanes of a warp: the states and the names the other lane's take, the combine program, the lanes
/// of a group, whether it is a prefix, and the type it runs at.
struct Fold<'a> {
    states: &'a [Name],
    others: &'a [Name],
    program: &'a [Arc<Statement>],
    length: i64,
    prefix: bool,
    ty: VTy,
}

/// Who writes a slot of a tree reduction.
enum Owner {
    /// Every thread has a slot of its own.
    Thread,
    /// The lanes of a warp share one slot, the warp's.
    Warp,
}

fn thread_index() -> Expression {
    Expression::new(Expr::Builtin(Builtin::ThreadIdxX))
}

/// The halvings of a power of two: 8 gives 4, 2, 1.
fn halvings(length: i64) -> impl Iterator<Item = i64> {
    std::iter::successors(Some(length / 2), |step| Some(step / 2)).take_while(|step| *step > 0)
}

impl Lowerer {
    // --- what is decided of the threads of a block -------------------------------------------------------

    /// `expression` over the thread index alone: every name it reads replaced by what defines it. `None` where it
    /// reads anything else: the block's index, a size, a loop's axis, a loaded value, a state.
    fn over_thread(&self, expression: &Expression) -> Option<Expression> {
        let mut defined = HashMap::new();
        for name in expression.free_vars() {
            let value = match &self.find(name)?.1.op {
                Operand::Int(value) => Expression::int(*value),
                Operand::Reg(reg) => self.by_thread.get(reg)?.clone(),
                _ => return None,
            };
            defined.insert(name, value);
        }
        let mut closed = true;
        let threads = self.threads.count() as i64;
        let over = expression.substitute(&defined).rebuild(&mut |node| {
            let known = match &*node {
                Expr::Builtin(Builtin::ThreadIdxX) => Some(Expression::var(THREAD)),
                Expr::Builtin(Builtin::BlockDimX) => Some(Expression::int(threads)),
                Expr::Builtin(Builtin::WarpSize) => Some(Expression::int(WARP)),
                Expr::Builtin(_) => {
                    closed = false;
                    None
                }
                _ => None,
            };
            known.unwrap_or(node)
        });
        (closed && over.height() <= MAX_HEIGHT).then_some(over)
    }

    /// Remember the register of `val` by `expression`, where that is one over the thread index alone.
    pub(super) fn note_by_thread(&mut self, val: &Val, expression: &Expression) {
        if let Operand::Reg(reg) = val.op
            && !self.b.is_state(reg)
            && !matches!(val.ty, VTy::F32 | VTy::F16 | VTy::Bf16)
            && let Some(over) = self.over_thread(expression)
        {
            self.by_thread.insert(reg, over);
        }
    }

    /// The value of `expression` in every thread of a block, where it is decided by the thread index alone.
    fn per_thread(&self, expression: &Expression) -> Option<Vec<Num>> {
        let over = self.over_thread(expression)?;
        let thread = Name::from(THREAD);
        let mut at = HashMap::new();
        (0..self.threads.count() as i64)
            .map(|index| {
                at.insert(thread, Num::Int(index));
                over.eval(&at).ok()
            })
            .collect()
    }

    pub(super) fn truth_by_thread(&self, condition: &Expression) -> Option<Vec<bool>> {
        let values = self.per_thread(condition)?;
        Some(values.into_iter().map(Num::truthy).collect())
    }

    /// Whether one of `expressions` is decided to differ between two threads that reach the statement.
    pub(super) fn differs_by_thread(&self, expressions: &[&Expression]) -> bool {
        let varying = expressions.iter().filter(|e| e.literal_value().is_none());
        varying.into_iter().any(|expression| {
            let Some(values) = self.per_thread(expression) else {
                return false;
            };
            let mut reaching = self.threads.reaching().map(|thread| values[thread]);
            let first = reaching.next();
            reaching.any(|value| Some(value) != first)
        })
    }

    // --- barriers ----------------------------------------------------------------------------------------

    /// A barrier of the whole block, or of `count` threads at the named barrier `id`. It is refused where the
    /// conditions and loops around it decide that the threads meant to meet at it do not all reach it.
    fn barrier(&mut self, id: i64, count: Option<i64>) -> Result<()> {
        let threads = &self.threads;
        if threads.uneven > 0 {
            return Err(self.refuse(
                "a barrier stands in a loop whose trips differ between the threads of a block",
            ));
        }
        let (reaching, live) = (
            threads.reaching().count(),
            threads.live.iter().filter(|live| **live).count(),
        );
        let operands = match (id, count) {
            (0, _) if reaching != live => {
                return Err(self.refuse(format_args!(
                    "a barrier of the whole block is reached by {reaching} of its {live} threads"
                )));
            }
            (0, _) => vec![Operand::Int(0)],
            (id, Some(count)) if NAMED_BARRIERS.contains(&id) => {
                if count <= 0 || count % WARP != 0 {
                    return Err(self.refuse(format_args!(
                        "a barrier of {count} threads: ptxas takes whole warps"
                    )));
                }
                if threads.open == 0 && count != reaching as i64 {
                    return Err(self.refuse(format_args!(
                        "a barrier of {count} threads is reached by {reaching}"
                    )));
                }
                vec![Operand::Int(id), Operand::Int(count)]
            }
            _ => return Err(self.refuse("a named barrier is one of 1 to 15, with a thread count")),
        };
        self.b
            .emit(Inst::new(Opcode::BarSync, Ty::Pred, Vec::new(), operands));
        Ok(())
    }

    pub(super) fn sync(&mut self, sync: &Sync) -> Result<()> {
        if !sync.warp {
            return self.barrier(sync.barrier_id, sync.count);
        }
        if sync.barrier_id != 0 {
            return Err(self.refuse("a barrier of a warp names no barrier"));
        }
        // Every lane of the warp: a lane the block does not hold never ran, so it is not waited for.
        let every = vec![Operand::Int(-1)];
        self.b
            .emit(Inst::new(Opcode::BarWarpSync, Ty::Pred, Vec::new(), every));
        Ok(())
    }

    // --- the combine program -----------------------------------------------------------------------------

    /// Run a combine program: a statement that defines one of `states` moves its value into that state's register,
    /// under `guard` when one is given, and any other is a temporary of the program. The order is the program's: a
    /// state is read before the statement that updates it.
    fn combine(
        &mut self,
        program: &[Arc<Statement>],
        states: &[Name],
        ty: VTy,
        guard: Option<Reg>,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let mut ctx = ctx.clone();
        let around = self.statement.clone();
        for (at, stmt) in program.iter().enumerate() {
            let text = statement_text(stmt).unwrap_or_else(|_| stmt.ty().name().to_string());
            self.statement = text.lines().next().unwrap_or_default().to_string();
            self.b.comment(self.statement.clone());
            match &**stmt {
                Statement::Assign(assign) if states.contains(&assign.name) => {
                    if assign.dtype.is_some_and(|dtype| VTy::of(dtype) != Some(ty)) {
                        let ty = ty.text();
                        return Err(self.refuse(format_args!("a state combined at {ty}")));
                    }
                    let state = self.state(assign.name, ty)?;
                    let before = self.b.lines();
                    let val = self.typed(&assign.value, ty, &ctx)?;
                    // The instruction that computed the value for this statement alone defines the state itself.
                    let computed = assign.value.operation().is_some() && self.b.lines() > before;
                    let stepped = match val.op {
                        Operand::Reg(reg) if computed => self.b.retarget(reg, state, guard),
                        _ => false,
                    };
                    if !stepped {
                        let step = Inst::new(Opcode::Mov, ty.held(), vec![state], vec![val.op]);
                        self.b.emit(match guard {
                            Some(guard) => step.guarded(guard, false),
                            None => step,
                        });
                    }
                }
                _ => self.rule(stmt, &program[at + 1..], &mut ctx)?,
            }
        }
        self.statement = around;
        Ok(())
    }

    // --- warp shuffles -----------------------------------------------------------------------------------

    /// The mask of the lanes that run here together. It is read once, ahead of a statement's shuffles: every
    /// shuffle under it then waits for the same lanes.
    fn active_mask(&mut self) -> Operand {
        let mask = self.b.reg(Bank::B32);
        self.b.emit(Inst::new(
            Opcode::Activemask,
            Ty::B32,
            vec![mask],
            Vec::new(),
        ));
        mask.into()
    }

    /// The value of `ty` in `from`, read from the lane the shuffle picks. A shuffle moves 32 bits: a float goes
    /// through its bits, a half through a 32-bit register.
    pub(super) fn shuffled(
        &mut self,
        shuffle: Shuffle,
        (from, ty): (Reg, VTy),
        (lane, clamp): (i64, i64),
        mask: &Operand,
    ) -> Result<Reg> {
        let moved = |this: &mut Lowerer, opcode, ty: Ty, operand: Reg| {
            let to = this.b.reg(ty.bank());
            this.b
                .emit(Inst::new(opcode, ty, vec![to], vec![operand.into()]));
            to
        };
        let bits = match ty {
            VTy::F32 => moved(self, Opcode::Mov, Ty::B32, from),
            VTy::F16 | VTy::Bf16 => moved(self, Opcode::Cvt(Ty::U16), Ty::U32, from),
            VTy::I32 | VTy::U32 => from,
            _ => return Err(self.no_rule(format_args!("a shuffle of {}", ty.text()))),
        };
        let theirs = self.b.reg(Bank::B32);
        let operands = vec![
            bits.into(),
            Operand::Int(lane),
            Operand::Int(clamp),
            mask.clone(),
        ];
        let inst = Inst::new(Opcode::Shfl(shuffle), Ty::B32, vec![theirs], operands);
        self.b.emit(inst);
        Ok(match ty {
            VTy::F32 => {
                let to = self.b.reg(Bank::F32);
                let back = Inst::new(Opcode::Mov, Ty::B32, vec![to], vec![theirs.into()]);
                self.b.emit(back);
                to
            }
            VTy::F16 | VTy::Bf16 => moved(self, Opcode::Cvt(Ty::U32), Ty::U16, theirs),
            _ => theirs,
        })
    }

    /// The lane count of a warp statement: a power of two up to the warp.
    fn lanes(&self, length: i64) -> Result<i64> {
        if (1..=WARP).contains(&length) && length & (length - 1) == 0 {
            return Ok(length);
        }
        Err(self.refuse(format_args!(
            "{length} lanes are no power of two up to a warp"
        )))
    }

    /// Fold the `length` lanes of each aligned group into every one of them. At each step a lane reads every state
    /// of the lane a stride away into the fold's other names, then runs the combine program over its own states. A
    /// butterfly halves the stride; a prefix doubles it and reads the lane below, combining only where one stands
    /// in the group, so each lane ends with the fold of the lanes up to itself. `mask` names the lanes that take
    /// part, the lanes that run here together when it is not given.
    fn fold_lanes(&mut self, fold: &Fold, mask: Option<Operand>, ctx: &SimplifyCtx) -> Result<()> {
        let (ty, length) = (fold.ty, fold.length);
        if fold.states.len() != fold.others.len() {
            return Err(self.refuse("every state has the name its other lane's value takes"));
        }
        let mut strides: Vec<i64> = halvings(length).collect();
        if fold.prefix {
            strides.reverse();
        }
        if strides.is_empty() {
            return Ok(());
        }
        // A group of `length` lanes: the shuffle's clamp keeps a lane from reading past its own.
        let segment = (WARP - length) << 8;
        let mask = mask.unwrap_or_else(|| self.active_mask());
        for stride in strides {
            self.names.push(HashMap::new());
            let done = (|| {
                for (state, other) in fold.states.iter().zip(fold.others) {
                    let from = (self.state(*state, ty)?, ty);
                    let theirs = match fold.prefix {
                        true => self.shuffled(Shuffle::Up, from, (stride, segment), &mask)?,
                        false => self.shuffled(Shuffle::Bfly, from, (stride, WARP - 1), &mask)?,
                    };
                    self.bind(*other, Val::reg(theirs, ty));
                }
                let guard = match fold.prefix {
                    true => {
                        let (lane, least) = (Expression::int(length), Expression::int(stride));
                        let lane = Expression::binary(Operation::Remainder, thread_index(), lane);
                        let below = Expression::binary(Operation::GreaterEqual, lane, least);
                        Some(self.pred(&below, ctx)?)
                    }
                    false => None,
                };
                self.combine(fold.program, fold.states, ty, guard, ctx)
            })();
            self.names.pop();
            done?;
        }
        Ok(())
    }

    pub(super) fn warp_shuffle(&mut self, node: &WarpShuffle, ctx: &SimplifyCtx) -> Result<()> {
        let fold = Fold {
            states: &node.state,
            others: &node.state_b,
            program: &node.combine_states,
            length: self.lanes(node.length)?,
            prefix: node.prefix,
            ty: self.vty(Some(node.dtype))?,
        };
        self.fold_lanes(&fold, None, ctx)
    }

    /// Every lane of each aligned group takes the states of the group's last lane.
    pub(super) fn warp_broadcast(&mut self, node: &WarpBroadcast) -> Result<()> {
        let length = self.lanes(node.length)?;
        let mask = self.active_mask();
        let pick = (length - 1, (WARP - length) << 8 | (WARP - 1));
        for name in node.state.iter() {
            let Some((_, val)) = self.find(*name) else {
                return Err(self.refuse(format_args!("{name} is updated before it is defined")));
            };
            let ty = val.ty;
            let state = self.state(*name, ty)?;
            let last = self.shuffled(Shuffle::Idx, (state, ty), pick, &mask)?;
            self.b.emit(Inst::new(
                Opcode::Mov,
                ty.held(),
                vec![state],
                vec![last.into()],
            ));
        }
        Ok(())
    }

    // --- the tree reduction ------------------------------------------------------------------------------

    /// Who writes each slot `slot` names, over the threads that reach the statement: its own thread, or the lanes
    /// of one warp together. Anything else, and a slot the thread index alone does not decide, is refused: two
    /// threads that read, combine and write one slot race.
    fn slot_owner(&self, slot: &Expression) -> Result<Owner> {
        let Some(slots) = self.per_thread(slot) else {
            return Err(self.no_rule(format_args!(
                "a tree reduction whose slot, {slot}, the thread index alone does not decide"
            )));
        };
        let reaching: Vec<usize> = self.threads.reaching().collect();
        let taken: Option<Vec<i64>> = reaching.iter().map(|thread| slots[*thread].int()).collect();
        let Some(taken) = taken else {
            return Err(self.refuse(format_args!("the slot {slot} is no integer")));
        };
        let distinct: HashSet<i64> = taken.iter().copied().collect();
        if distinct.len() == taken.len() {
            return Ok(Owner::Thread);
        }
        let by_warp = reaching.iter().zip(&taken);
        if by_warp
            .into_iter()
            .all(|(thread, slot)| *slot == *thread as i64 / WARP)
        {
            return Ok(Owner::Warp);
        }
        Err(self.no_rule(format_args!(
            "a tree reduction where threads share the slot {slot} other than by warp"
        )))
    }

    /// Reduce a shared slab of `length` slots per state into its first, then give every thread that first slot as
    /// its state. Where each thread has a slot of its own, every halving step has the threads of the low half
    /// read their slot and the one a stride above, combine them and write their own, then a barrier. Where the
    /// lanes of a warp share a slot, the block's first lanes take a slot each into registers, fold them with a
    /// butterfly, and the first of them writes the result: one writer a slot, either way.
    pub(super) fn tree_halve(&mut self, node: &TreeHalve, ctx: &SimplifyCtx) -> Result<()> {
        let ty = self.vty(Some(node.dtype))?;
        let length = node.length;
        if length < 1 || length & (length - 1) != 0 {
            return Err(self.refuse(format_args!("{length} slots are no power of two")));
        }
        if node.bufs.len() != node.state.len() || node.bufs.len() != node.state_b.len() {
            return Err(self.refuse("every state has a slab and the name its other slot takes"));
        }
        let mut outer = Vec::with_capacity(node.state.len());
        for (name, slab) in node.state.iter().zip(node.bufs.iter()) {
            let buffer = self.buffer(*slab)?;
            if buffer.space != Space::Shared || buffer.elem.dtype != node.dtype {
                let (stored, read) = (buffer.elem.dtype, node.dtype);
                return Err(self.refuse(format_args!("{slab} holds {stored}, read at {read}")));
            }
            outer.push(self.state(*name, ty)?);
        }
        let (scale, slot, root) = match &node.inner {
            Some((within, scale)) => {
                let scaled = Expression::binary(
                    Operation::Multiply,
                    node.tid_var.clone(),
                    Expression::int(*scale),
                );
                let slot = Expression::binary(Operation::Add, scaled, within.clone());
                (*scale, slot, within.clone())
            }
            None => (1, node.tid_var.clone(), Expression::int(0)),
        };
        match self.slot_owner(&slot)? {
            Owner::Thread => {
                // Each thread's own slot of every slab, addressed once for all the steps.
                let mut slots = Vec::with_capacity(node.bufs.len());
                for slab in node.bufs.iter() {
                    slots.push(self.address(*slab, std::slice::from_ref(&slot), ctx)?);
                }
                for stride in halvings(length) {
                    let low = Expression::binary(
                        Operation::Less,
                        node.tid_var.clone(),
                        Expression::int(stride),
                    );
                    let pred = self.pred(&low, ctx)?;
                    let open = self.b.open_if(pred);
                    self.names.push(HashMap::new());
                    let done = self.tree_step(node, &slots, stride * scale, ty, ctx);
                    self.names.pop();
                    done?;
                    self.b.close_if(open)?;
                    self.barrier(node.barrier_id, node.barrier_count)?;
                }
            }
            Owner::Warp if node.inner.is_none() && node.barrier_id == 0 && length <= WARP => {
                self.tree_in_first_lanes(node, ty, ctx)?;
                self.barrier(0, None)?;
            }
            Owner::Warp => {
                return Err(self.no_rule(
                    "a tree reduction over slots a warp shares, past a warp of them or at a named barrier",
                ));
            }
        }
        for (state, slab) in outer.into_iter().zip(node.bufs.iter()) {
            let first = self.address(*slab, std::slice::from_ref(&root), ctx)?;
            self.load_slot(&first, 0, Some(state), ty);
        }
        Ok(())
    }

    /// Load the slot at `address`, `above` slots further, into `into` or a new state.
    fn load_slot(&mut self, address: &Operand, above: i64, into: Option<Reg>, ty: VTy) -> Reg {
        let address = match address {
            Operand::Addr(base, offset) => Operand::Addr(*base, offset + above * ty.bytes()),
            other => other.clone(),
        };
        let reg = into.unwrap_or_else(|| self.b.state(ty.bank()));
        let load = Opcode::Ld(Space::Shared);
        self.b
            .emit(Inst::new(load, ty.held(), vec![reg], vec![address]));
        reg
    }

    /// One halving step in a thread of the low half: its slot and the one `above` it into the state names and
    /// their others, the combine, and its slot written back.
    fn tree_step(
        &mut self,
        node: &TreeHalve,
        slots: &[Operand],
        above: i64,
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let mut own = Vec::with_capacity(slots.len());
        for ((slot, state), other) in slots.iter().zip(&*node.state).zip(&*node.state_b) {
            let mine = self.load_slot(slot, 0, None, ty);
            let theirs = self.load_slot(slot, above, None, ty);
            self.bind(*state, Val::reg(mine, ty));
            self.bind(*other, Val::reg(theirs, ty));
            own.push(mine);
        }
        self.combine(&node.combine_states, &node.state, ty, None, ctx)?;
        for (mine, slot) in own.into_iter().zip(slots) {
            let store = Opcode::St(Space::Shared);
            let operands = vec![slot.clone(), mine.into()];
            self.b
                .emit(Inst::new(store, ty.held(), Vec::new(), operands));
        }
        Ok(())
    }

    /// The slab folded in registers: each of the block's first `length` lanes takes a slot of its own, a butterfly
    /// over exactly those lanes folds them, and the first writes the first slot. No slot is read by one lane and
    /// written by another, which the CUDA renderer's form, with the whole first warp reading, does.
    fn tree_in_first_lanes(&mut self, node: &TreeHalve, ty: VTy, ctx: &SimplifyCtx) -> Result<()> {
        let compare = |op, value| Expression::binary(op, thread_index(), Expression::int(value));
        let folding = self.pred(&compare(Operation::Less, node.length), ctx)?;
        let open = self.b.open_if(folding);
        self.names.push(HashMap::new());
        let done = (|| {
            let mut own = Vec::with_capacity(node.bufs.len());
            for (slab, state) in node.bufs.iter().zip(node.state.iter()) {
                let slot = self.address(*slab, &[thread_index()], ctx)?;
                let mine = self.load_slot(&slot, 0, None, ty);
                self.bind(*state, Val::reg(mine, ty));
                own.push(mine);
            }
            let fold = Fold {
                states: &node.state,
                others: &node.state_b,
                program: &node.combine_states,
                length: node.length,
                prefix: false,
                ty,
            };
            let lanes = Operand::Int((1 << node.length) - 1);
            self.fold_lanes(&fold, Some(lanes), ctx)?;
            let first = self.pred(&compare(Operation::Equal, 0), ctx)?;
            let writes = self.b.open_if(first);
            for (mine, slab) in own.into_iter().zip(node.bufs.iter()) {
                let address = self.address(*slab, &[Expression::int(0)], ctx)?;
                let store = Opcode::St(Space::Shared);
                let operands = vec![address, mine.into()];
                self.b
                    .emit(Inst::new(store, ty.held(), Vec::new(), operands));
            }
            self.b.close_if(writes)
        })();
        self.names.pop();
        done?;
        self.b.close_if(open)
    }
}
