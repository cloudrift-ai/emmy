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
    /// The registers an instruction may define again: an accumulator, a seeded state.
    states: HashSet<Reg>,
    multiples: HashMap<Reg, i64>,
    alignment: HashMap<Reg, u32>,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            tables: vec![HashMap::new()],
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
        self.items.push(Item::Inst(inst));
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
            operands: inst.operands.clone(),
        };
        if keyed && let Some(reg) = self.tables.iter().rev().find_map(|table| table.get(&key)) {
            return *reg;
        }
        let reg = self.reg(bank);
        self.emit(Inst {
            results: vec![reg],
            ..inst
        });
        if keyed && let Some(table) = self.tables.last_mut() {
            table.insert(key, reg);
        }
        reg
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
        self.tables.push(HashMap::new());
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
        self.tables.push(HashMap::new());
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
        let var = self.reg(Bank::B32);
        self.emit(Inst::new(Opcode::Mov, Ty::U32, vec![var], vec![start]));
        let done = (!enters).then(|| {
            let done = self.label();
            let past = self.compare(Cmp::Ge, var.into(), end.clone());
            self.branch(done, Some((past, false)));
            done
        });
        let head = self.label();
        self.items.push(Item::Label(head));
        self.tables.push(HashMap::new());
        Loop {
            var,
            end,
            step,
            head,
            done,
            depth: self.tables.len(),
        }
    }

    pub fn close_loop(&mut self, open: Loop) -> Result<()> {
        self.leave(open.depth)?;
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

    /// Drop the value table of the scope opened at `depth`: it must be the innermost open one.
    fn leave(&mut self, depth: usize) -> Result<()> {
        if self.tables.len() != depth {
            return Err(Error::Value(format!(
                "a scope opened at depth {depth} is closed at depth {}",
                self.tables.len()
            )));
        }
        self.tables.pop();
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
