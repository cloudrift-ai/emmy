//! The instruction list: what a kernel is between the rules and the printer. Typed instructions and labels in one
//! flat list, opcodes and modifiers as enums, so nothing is a string until it is printed.

use std::collections::HashMap;

/// A register bank, as PTX declares registers: by width, with the floats and the predicates apart. Signedness is
/// on the instruction, never on the register.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Bank {
    Pred,
    B16,
    B32,
    B64,
    F32,
}

impl Bank {
    pub const ALL: [Bank; 5] = [Bank::Pred, Bank::B16, Bank::B32, Bank::B64, Bank::F32];

    /// The name every register of the bank starts with.
    pub fn prefix(self) -> &'static str {
        match self {
            Bank::Pred => "p",
            Bank::B16 => "h",
            Bank::B32 => "r",
            Bank::B64 => "rd",
            Bank::F32 => "f",
        }
    }

    /// The type a `.reg` declaration gives the bank.
    pub fn declared(self) -> &'static str {
        match self {
            Bank::Pred => "pred",
            Bank::B16 => "b16",
            Bank::B32 => "b32",
            Bank::B64 => "b64",
            Bank::F32 => "f32",
        }
    }
}

/// A virtual register: ptxas allocates the real ones.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Reg {
    pub bank: Bank,
    pub number: u32,
}

/// The type an instruction computes at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Pred,
    B16,
    B32,
    B64,
    U8,
    S32,
    U32,
    S64,
    U64,
    F16,
    Bf16,
    F32,
}

impl Ty {
    pub fn text(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B16 => "b16",
            Ty::B32 => "b32",
            Ty::B64 => "b64",
            Ty::U8 => "u8",
            Ty::S32 => "s32",
            Ty::U32 => "u32",
            Ty::S64 => "s64",
            Ty::U64 => "u64",
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::F32 => "f32",
        }
    }

    /// The bank a value of this type lives in. A byte is held zero-extended in a 32-bit register, and a half in
    /// a 16-bit one.
    pub fn bank(self) -> Bank {
        match self {
            Ty::Pred => Bank::Pred,
            Ty::B16 | Ty::F16 | Ty::Bf16 => Bank::B16,
            Ty::B32 | Ty::U8 | Ty::S32 | Ty::U32 => Bank::B32,
            Ty::B64 | Ty::S64 | Ty::U64 => Bank::B64,
            Ty::F32 => Bank::F32,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Ty::F16 | Ty::Bf16 | Ty::F32)
    }

    /// The bytes one value takes in memory.
    pub fn bytes(self) -> i64 {
        match self {
            Ty::Pred | Ty::U8 => 1,
            Ty::B16 | Ty::F16 | Ty::Bf16 => 2,
            Ty::B32 | Ty::S32 | Ty::U32 | Ty::F32 => 4,
            Ty::B64 | Ty::S64 | Ty::U64 => 8,
        }
    }
}

/// A comparison, as `setp` spells it. The ordered float forms are false on a NaN; `Neu` is true on one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cmp {
    Eq,
    Ne,
    Neu,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    pub fn text(self) -> &'static str {
        match self {
            Cmp::Eq => "eq",
            Cmp::Ne => "ne",
            Cmp::Neu => "neu",
            Cmp::Lt => "lt",
            Cmp::Le => "le",
            Cmp::Gt => "gt",
            Cmp::Ge => "ge",
        }
    }

    /// The comparison with its sides swapped.
    pub fn mirrored(self) -> Cmp {
        match self {
            Cmp::Lt => Cmp::Gt,
            Cmp::Le => Cmp::Ge,
            Cmp::Gt => Cmp::Lt,
            Cmp::Ge => Cmp::Le,
            same => same,
        }
    }
}

/// A rounding mode: to a float (`Rn`), or to an integral value (`Rni` nearest even, `Rzi` toward zero, `Rmi` down,
/// `Rpi` up).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Round {
    Rn,
    Rni,
    Rzi,
    Rmi,
    Rpi,
}

impl Round {
    pub fn text(self) -> &'static str {
        match self {
            Round::Rn => "rn",
            Round::Rni => "rni",
            Round::Rzi => "rzi",
            Round::Rmi => "rmi",
            Round::Rpi => "rpi",
        }
    }
}

/// The state space a load or a store addresses.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Space {
    Global,
    Shared,
    Param,
}

impl Space {
    pub fn text(self) -> &'static str {
        match self {
            Space::Global => "global",
            Space::Shared => "shared",
            Space::Param => "param",
        }
    }
}

/// What an instruction does. A variant holds what picks the instruction; its type and its float modifiers are the
/// instruction's own fields.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Opcode {
    Mov,
    Add,
    Sub,
    /// A float product, or the low half of an integer one (`mul.lo`).
    Mul,
    MulHi,
    /// A 32-bit by 32-bit product in 64 bits. The instruction's type is its operands'.
    MulWide,
    Div,
    Rem,
    Neg,
    Abs,
    Min,
    Max,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Not,
    Popc,
    Fma,
    Rcp,
    Sqrt,
    Setp(Cmp),
    Selp,
    /// A conversion to the instruction's type from this one.
    Cvt(Ty),
    Ld(Space),
    St(Space),
    Bra,
    Ret,
}

impl Opcode {
    /// Whether the instruction only computes its result from its operands: what the value table may share.
    pub fn is_pure(self) -> bool {
        !matches!(
            self,
            Opcode::Ld(_) | Opcode::St(_) | Opcode::Bra | Opcode::Ret
        )
    }
}

/// A special register: a fact of the thread and the launch, read with `mov`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Special {
    TidX,
    TidY,
    TidZ,
    CtaidX,
    CtaidY,
    CtaidZ,
    NtidX,
    NtidY,
    NtidZ,
    NctaidX,
    NctaidY,
    NctaidZ,
}

impl Special {
    pub fn text(self) -> &'static str {
        match self {
            Special::TidX => "%tid.x",
            Special::TidY => "%tid.y",
            Special::TidZ => "%tid.z",
            Special::CtaidX => "%ctaid.x",
            Special::CtaidY => "%ctaid.y",
            Special::CtaidZ => "%ctaid.z",
            Special::NtidX => "%ntid.x",
            Special::NtidY => "%ntid.y",
            Special::NtidZ => "%ntid.z",
            Special::NctaidX => "%nctaid.x",
            Special::NctaidY => "%nctaid.y",
            Special::NctaidZ => "%nctaid.z",
        }
    }
}

/// A label, by number. Only the builder makes and binds one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Label(pub u32);

/// What an address starts from: a register holding a pointer, or a symbol of the module by its position (a
/// parameter, a shared declaration). The printer names the symbols, so no Kernel IR name has to be a PTX identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Base {
    Reg(Reg),
    Param(u32),
    Shared(u32),
}

/// One operand of an instruction.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Operand {
    Reg(Reg),
    /// An integer constant.
    Int(i64),
    /// A float constant, by the bits of its `f32`.
    F32(u32),
    /// A base plus a constant byte offset.
    Addr(Base, i64),
    /// Several registers one vector access fills or stores.
    Tuple(Vec<Reg>),
    Special(Special),
    Label(Label),
}

impl Operand {
    pub fn f32(value: f32) -> Operand {
        Operand::F32(value.to_bits())
    }

    pub fn reg(&self) -> Option<Reg> {
        match self {
            Operand::Reg(reg) => Some(*reg),
            _ => None,
        }
    }

    /// The integer constant this operand is, when it is one.
    pub fn int(&self) -> Option<i64> {
        match self {
            Operand::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// How many elements the operand holds: the registers of a tuple, one for any other.
    pub fn lanes(&self) -> usize {
        match self {
            Operand::Tuple(regs) => regs.len(),
            _ => 1,
        }
    }

    /// Every register the operand reads.
    pub fn reads(&self) -> Vec<Reg> {
        match self {
            Operand::Reg(reg) | Operand::Addr(Base::Reg(reg), _) => vec![*reg],
            Operand::Tuple(regs) => regs.clone(),
            _ => Vec::new(),
        }
    }
}

impl From<Reg> for Operand {
    fn from(reg: Reg) -> Operand {
        Operand::Reg(reg)
    }
}

/// A predicate an instruction runs under: `@%p` or, negated, `@!%p`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Guard {
    pub pred: Reg,
    pub negated: bool,
}

/// One instruction: its opcode, the type it computes at, its float modifiers, the registers it defines, its
/// operands in the order PTX takes them, and the guard it runs under.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Inst {
    pub opcode: Opcode,
    pub ty: Ty,
    pub round: Option<Round>,
    pub ftz: bool,
    pub results: Vec<Reg>,
    pub operands: Vec<Operand>,
    pub guard: Option<Guard>,
}

impl Inst {
    pub fn new(opcode: Opcode, ty: Ty, results: Vec<Reg>, operands: Vec<Operand>) -> Inst {
        Inst {
            opcode,
            ty,
            round: None,
            ftz: false,
            results,
            operands,
            guard: None,
        }
    }

    pub fn rounded(mut self, round: Round) -> Inst {
        self.round = Some(round);
        self
    }

    pub fn flushing(mut self, ftz: bool) -> Inst {
        self.ftz = ftz;
        self
    }

    pub fn guarded(mut self, pred: Reg, negated: bool) -> Inst {
        self.guard = Some(Guard { pred, negated });
        self
    }

    /// Every register the instruction reads: its operands' and its guard's.
    pub fn reads(&self) -> Vec<Reg> {
        self.operands
            .iter()
            .flat_map(Operand::reads)
            .chain(self.guard.map(|guard| guard.pred))
            .collect()
    }

    /// The label a branch goes to.
    pub fn target(&self) -> Option<Label> {
        match (self.opcode, self.operands.first()) {
            (Opcode::Bra, Some(Operand::Label(label))) => Some(*label),
            _ => None,
        }
    }
}

/// One line of a kernel's code.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    Inst(Inst),
    Label(Label),
    /// A Kernel IR statement, kept above its instructions so a kernel can still be read.
    Comment(String),
}

/// How a parameter is declared: a pointer with the alignment of its elements, or a 32-bit integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamDecl {
    Pointer { align: u32 },
    U32,
}

/// A shared-memory declaration: an array of bytes of the module, or the launch-sized pool declared outside it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shared {
    pub align: u32,
    /// The array's size; `None` for the external pool.
    pub bytes: Option<u64>,
}

/// A kernel as PTX holds it: the directives, the declarations and the code.
#[derive(Clone, PartialEq, Debug)]
pub struct Module {
    pub version: (u32, u32),
    /// The target as ptxas spells it (`sm_120`, `sm_90a`).
    pub target: String,
    pub entry: String,
    pub params: Vec<ParamDecl>,
    pub shared: Vec<Shared>,
    /// The most threads a block launches with (`.maxntid`): without it ptxas reserves registers for 1,024.
    pub max_threads: u32,
    /// How many registers each bank holds.
    pub registers: Vec<(Bank, u32)>,
    pub items: Vec<Item>,
    /// The bytes each address register is proven a multiple of: what a vector access is checked against.
    pub alignment: HashMap<Reg, u32>,
}

impl Module {
    /// The instructions, without the labels and the comments.
    pub fn instructions(&self) -> impl Iterator<Item = &Inst> {
        self.items.iter().filter_map(|item| match item {
            Item::Inst(inst) => Some(inst),
            _ => None,
        })
    }
}
