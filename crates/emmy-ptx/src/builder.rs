//! The builder: the one way instructions reach a kernel's list. It hands out registers, keeps the value table of
//! every open scope and what is known of each register (a number it is a multiple of, the alignment of an address),
//! and is the only writer of labels and branches: a loop or a condition is opened and closed through it.

use std::collections::{HashMap, HashSet};

use emmy_ir::{Error, Result};

use crate::inst::{Bank, Cmp, Inst, Item, Label, Opcode, Operand, Reg, Round, Ty};

/// A pure instruction without its result: two with one key compute one value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Key {
    opcode: Opcode,
    ty: Ty,
    round: Option<Round>,
    ftz: bool,
    sat: bool,
    operands: Vec<Operand>,
}

/// An open condition: the label its false side goes to, and the label its end is, once it has an else side.
#[derive(Debug)]
#[must_use = "an open condition is closed with `close_if`"]
pub struct If {
    skip: Label,
    end: Option<Label>,
    depth: usize,
}

/// An open loop: its variable, its bound and step, and its labels.
#[derive(Debug)]
#[must_use = "an open loop is closed with `close_loop`"]
pub struct Loop {
    pub var: Reg,
    end: Operand,
    step: Operand,
    head: Label,
    done: Option<Label>,
    depth: usize,
}

/// What a finished builder holds: the code, the register count of each bank, and the alignment proven of each
/// address register.
pub struct Code {
    pub items: Vec<Item>,
    pub registers: Vec<(Bank, u32)>,
    pub alignment: HashMap<Reg, u32>,
}

#[derive(Default)]
pub struct Builder {
    items: Vec<Item>,
    counts: HashMap<Bank, u32>,
    labels: u32,
    /// The value table of each open scope, the innermost last.
    tables: Vec<HashMap<Key, Reg>>,
    /// Where each open scope that is a loop starts in the list, beside its table: the place a value the loop does
    /// not change is computed ahead of it. `None` for the kernel's scope and a condition's side.
    heads: Vec<Option<usize>>,
    /// The depth of the scope each register is defined in, the deepest where it is defined more than once.
    defined_at: HashMap<Reg, usize>,
    /// The variable of each open scope that is a loop counting from zero by one, beside its table.
    counters: Vec<Option<Reg>>,
    /// The multiples of a loop's counter that addresses read, by counter, factor and bank (64 bits for a global
    /// address, 32 for a shared one), each made when the first address reads it.
    multiples_of: HashMap<(Reg, i64, Bank), Reg>,
    /// The registers an instruction may define again: an accumulator, a seeded state.
    states: HashSet<Reg>,
    multiples: HashMap<Reg, i64>,
    alignment: HashMap<Reg, u32>,
    /// The register the instruction appended last defines, where no table keeps it.
    loose: Option<Reg>,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            tables: vec![HashMap::new()],
            heads: vec![None],
            counters: vec![None],
            ..Builder::default()
        }
    }

    /// A register no instruction has named yet.
    pub fn reg(&mut self, bank: Bank) -> Reg {
        let count = self.counts.entry(bank).or_insert(0);
        let reg = Reg {
            bank,
            number: *count,
        };
        *count += 1;
        reg
    }

    pub fn comment(&mut self, text: String) {
        self.items.push(Item::Comment(text));
    }

    /// A register that is defined again where its state steps: an accumulator, a seeded state. The value table
    /// keeps nothing computed from one, since the same instruction later reads another value.
    pub fn state(&mut self, bank: Bank) -> Reg {
        let reg = self.reg(bank);
        self.states.insert(reg);
        reg
    }

    pub fn is_state(&self, reg: Reg) -> bool {
        self.states.contains(&reg)
    }

    /// Append `inst` as written: a load, a store, or a definition of a state.
    pub fn emit(&mut self, inst: Inst) {
        self.loose = None;
        let depth = self.tables.len();
        for reg in &inst.results {
            let at = self.defined_at.entry(*reg).or_insert(depth);
            *at = (*at).max(depth);
        }
        self.items.push(Item::Inst(inst));
    }

    /// How many lines the code holds so far.
    pub fn lines(&self) -> usize {
        self.items.len()
    }

    /// Have the instruction appended last define `to` and not `from`, under `guard` when one is given: a value
    /// computed for a state alone steps the state in place. `false`, with nothing changed, where that instruction
    /// does not define `from` or a table keeps `from` for other readers.
    pub fn retarget(&mut self, from: Reg, to: Reg, guard: Option<Reg>) -> bool {
        match self.items.last_mut() {
            Some(Item::Inst(inst)) if self.loose == Some(from) && inst.guard.is_none() => {
                inst.results = vec![to];
                inst.guard = guard.map(|pred| crate::inst::Guard {
                    pred,
                    negated: false,
                });
                self.loose = None;
                true
            }
            _ => false,
        }
    }

    /// The register holding the value of a pure instruction: the one an open scope already computed it into, else
    /// a new register the instruction is appended for. A value made twice is made once.
    pub fn value(&mut self, inst: Inst, bank: Bank) -> Reg {
        let keyed = inst.opcode.is_pure() && !inst.reads().iter().any(|reg| self.is_state(*reg));
        let key = Key {
            opcode: inst.opcode,
            ty: inst.ty,
            round: inst.round,
            ftz: inst.ftz,
            sat: inst.sat,
            operands: inst.operands.clone(),
        };
        if keyed && let Some(reg) = self.tables.iter().rev().find_map(|table| table.get(&key)) {
            return *reg;
        }
        let reg = self.reg(bank);
        let inst = Inst {
            results: vec![reg],
            ..inst
        };
        if keyed && let Some(depth) = self.invariant_depth(&inst) {
            self.hoist(depth, inst, key);
            return reg;
        }
        self.emit(inst);
        if keyed && let Some(table) = self.tables.last_mut() {
            table.insert(key, reg);
        } else {
            self.loose = Some(reg);
        }
        reg
    }

    /// The depth of the outermost scope a pure integer value can be computed in, where that is outside the loops
    /// it stands in: the loops around it that define none of its operands. `None` where no loop can be left. A
    /// condition is never left, since not every thread runs its side, and a float value is never moved: the body's
    /// optimization placed every value the kernel names, so what moves here is what the compiler makes itself
    /// (address parts, widenings, the terms of an index).
    fn invariant_depth(&self, inst: &Inst) -> Option<usize> {
        let integer = matches!(
            inst.ty,
            Ty::B32 | Ty::B64 | Ty::S32 | Ty::U32 | Ty::S64 | Ty::U64
        );
        if !integer
            || inst.guard.is_some()
            || !matches!(inst.results[..], [reg] if matches!(reg.bank, Bank::B32 | Bank::B64))
        {
            return None;
        }
        let depth = self.tables.len();
        let operands = inst
            .reads()
            .iter()
            .map(|reg| self.defined_at.get(reg).copied().unwrap_or(depth))
            .max();
        let floor = operands.unwrap_or(1);
        let mut at = depth;
        while at > floor && self.heads[at - 1].is_some() {
            at -= 1;
        }
        (at < depth).then_some(at)
    }

    /// Compute `inst` in the scope at `depth`, ahead of the loop it holds that is open, and keep it in that scope's
    /// table.
    fn hoist(&mut self, depth: usize, inst: Inst, key: Key) {
        let place = self.heads[depth].expect("the scope left is a loop");
        for head in self.heads[depth..].iter_mut().flatten() {
            *head += 1;
        }
        self.defined_at.insert(inst.results[0], depth);
        self.tables[depth - 1].insert(key, inst.results[0]);
        self.items.insert(place, Item::Inst(inst));
    }

    /// A number every value `reg` takes is a multiple of; 1 when nothing is known.
    pub fn multiple(&self, reg: Reg) -> i64 {
        self.multiples.get(&reg).copied().unwrap_or(1)
    }

    pub fn set_multiple(&mut self, reg: Reg, multiple: i64) {
        if multiple > 1 {
            self.multiples.insert(reg, multiple);
        }
    }

    /// The bytes the address in `reg` is proven a multiple of; 1 when nothing is known.
    pub fn alignment(&self, reg: Reg) -> u32 {
        self.alignment.get(&reg).copied().unwrap_or(1)
    }

    pub fn set_alignment(&mut self, reg: Reg, bytes: u32) {
        self.alignment.insert(reg, bytes.max(1));
    }

    fn label(&mut self) -> Label {
        self.labels += 1;
        Label(self.labels)
    }

    fn branch(&mut self, to: Label, guard: Option<(Reg, bool)>) {
        let inst = Inst::new(Opcode::Bra, Ty::Pred, Vec::new(), vec![Operand::Label(to)]);
        self.emit(match guard {
            Some((pred, negated)) => inst.guarded(pred, negated),
            None => inst,
        });
    }

    fn compare(&mut self, cmp: Cmp, left: Operand, right: Operand) -> Reg {
        let pred = self.reg(Bank::Pred);
        self.emit(Inst::new(
            Opcode::Setp(cmp),
            Ty::S32,
            vec![pred],
            vec![left, right],
        ));
        pred
    }

    /// Open the side of a condition that runs where `pred` holds. What the side computes is forgotten when it
    /// closes: not every thread ran it.
    pub fn open_if(&mut self, pred: Reg) -> If {
        let skip = self.label();
        self.branch(skip, Some((pred, true)));
        self.push_scope(None, None);
        If {
            skip,
            end: None,
            depth: self.tables.len(),
        }
    }

    /// Close the true side of `open` and open the side that runs where its predicate does not hold.
    pub fn else_(&mut self, open: &mut If) -> Result<()> {
        self.leave(open.depth)?;
        let end = self.label();
        self.branch(end, None);
        self.items.push(Item::Label(open.skip));
        open.end = Some(end);
        self.push_scope(None, None);
        Ok(())
    }

    pub fn close_if(&mut self, open: If) -> Result<()> {
        self.leave(open.depth)?;
        self.items.push(Item::Label(open.end.unwrap_or(open.skip)));
        Ok(())
    }

    /// Open a loop of a signed 32-bit variable from `start` while it is below `end`, by `step`. `enters` says the
    /// first trip is known to run, so no test stands ahead of the loop: with a zero start and a constant end, that
    /// is the counter from zero to a constant ptxas unrolls. The test stands at the bottom either way.
    pub fn open_loop(&mut self, start: Operand, end: Operand, step: Operand, enters: bool) -> Loop {
        let place = self.items.len();
        let var = self.reg(Bank::B32);
        self.emit(Inst::new(
            Opcode::Mov,
            Ty::U32,
            vec![var],
            vec![start.clone()],
        ));
        let done = (!enters).then(|| {
            let done = self.label();
            let past = self.compare(Cmp::Ge, var.into(), end.clone());
            self.branch(done, Some((past, false)));
            done
        });
        let head = self.label();
        self.items.push(Item::Label(head));
        let counts = start == Operand::Int(0) && step == Operand::Int(1);
        self.push_scope(Some(place), counts.then_some(var));
        // The variable steps at the bottom of every trip, so nothing computed from it leaves the loop.
        self.defined_at.insert(var, self.tables.len());
        Loop {
            var,
            end,
            step,
            head,
            done,
            depth: self.tables.len(),
        }
    }

    /// Mark the loop just opened, before anything is written in it, as one ptxas keeps as it is (`.pragma
    /// "nounroll"` at its head): the kernel lowering already wrote its copies.
    pub fn keep_rolled(&mut self) {
        self.items.push(Item::NoUnroll);
    }

    pub fn close_loop(&mut self, open: Loop) -> Result<()> {
        self.leave(open.depth)?;
        let mut stepped: Vec<_> = self
            .multiples_of
            .iter()
            .filter(|((var, _, _), _)| *var == open.var)
            .map(|(key, reg)| (*key, *reg))
            .collect();
        stepped.sort_by_key(|(_, reg)| (reg.bank, reg.number));
        for (key, reg) in stepped {
            self.multiples_of.remove(&key);
            let ty = if key.2 == Bank::B64 { Ty::S64 } else { Ty::S32 };
            let step = vec![reg.into(), Operand::Int(key.1)];
            self.emit(Inst::new(Opcode::Add, ty, vec![reg], step));
        }
        self.emit(Inst::new(
            Opcode::Add,
            Ty::S32,
            vec![open.var],
            vec![open.var.into(), open.step],
        ));
        let again = self.compare(Cmp::Lt, open.var.into(), open.end);
        self.branch(open.head, Some((again, false)));
        if let Some(done) = open.done {
            self.items.push(Item::Label(done));
        }
        Ok(())
    }

    /// Run `inst`, which defines the predicate `done`, again until that predicate holds: the one loop that is an
    /// instruction waiting, with no body.
    pub fn repeat_until(&mut self, done: Reg, inst: Inst) {
        let head = self.label();
        self.items.push(Item::Label(head));
        self.emit(inst);
        self.branch(head, Some((done, true)));
    }

    fn push_scope(&mut self, head: Option<usize>, counter: Option<Reg>) {
        self.tables.push(HashMap::new());
        self.heads.push(head);
        self.counters.push(counter);
    }

    /// The counter `var` of an open loop that counts from zero by one, times `factor`, as a register of `bank`
    /// (`B64` for a global address, `B32` for a shared one) that steps by `factor` beside it: made ahead of the loop
    /// when an address first reads it. ptxas folds a 64-bit term of it into `[base + constant]` across the trips it
    /// unrolls, which it does not for the widening of the 32-bit counter, since `var + 1` may wrap. Stepping the
    /// multiple itself, not the counter scaled in the trip, is what lets ptxas issue every load of a trip it keeps
    /// rolled ahead of the arithmetic, into registers of their own, as it does for cicc's stepped pointers. `None`
    /// for any other register.
    pub fn counter_multiple(&mut self, var: Reg, factor: i64, bank: Bank) -> Option<Reg> {
        if let Some(reg) = self.multiples_of.get(&(var, factor, bank)) {
            return Some(*reg);
        }
        let at = self
            .counters
            .iter()
            .position(|counter| *counter == Some(var))?;
        let place = self.heads[at]?;
        let reg = self.reg(bank);
        for head in self.heads[at..].iter_mut().flatten() {
            *head += 1;
        }
        let ty = if bank == Bank::B64 { Ty::U64 } else { Ty::U32 };
        let start = Inst::new(Opcode::Mov, ty, vec![reg], vec![Operand::Int(0)]);
        self.items.insert(place, Item::Inst(start));
        self.defined_at.insert(reg, at + 1);
        self.multiples.insert(reg, factor.saturating_abs().max(1));
        self.multiples_of.insert((var, factor, bank), reg);
        Some(reg)
    }

    /// Drop the value table of the scope opened at `depth`: it must be the innermost open one.
    fn leave(&mut self, depth: usize) -> Result<()> {
        if self.tables.len() != depth {
            return Err(Error::Value(format!(
                "a scope opened at depth {depth} is closed at depth {}",
                self.tables.len()
            )));
        }
        self.tables.pop();
        self.heads.pop();
        self.counters.pop();
        Ok(())
    }

    /// The finished code, closed with `ret`. An error when a scope is still open.
    pub fn finish(mut self) -> Result<Code> {
        self.leave(1)?;
        self.emit(Inst::new(Opcode::Ret, Ty::Pred, Vec::new(), Vec::new()));
        let registers = Bank::ALL
            .iter()
            .map(|bank| (*bank, self.counts.get(bank).copied().unwrap_or(0)))
            .collect();
        Ok(Code {
            items: self.items,
            registers,
            alignment: self.alignment,
        })
    }
}
