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
    U16,
    S32,
    U32,
    S64,
    U64,
    F16,
    Bf16,
    F32,
    /// Two halves in one 32-bit register: what a packed conversion writes.
    F16x2,
    Bf16x2,
    /// The 8-bit and 4-bit float formats a tensor-core product reads its operands at, and the unsigned 8-bit one
    /// a block of them is scaled by. No register holds one alone: each names an instruction's operand type.
    E4m3,
    E5m2,
    E2m1,
    Ue4m3,
    /// Two 8-bit floats in 16 bits: the source of a packed conversion to two halves, or its result from two f32.
    E4m3x2,
    E5m2x2,
}

impl Ty {
    pub fn text(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B16 => "b16",
            Ty::B32 => "b32",
            Ty::B64 => "b64",
            Ty::U8 => "u8",
            Ty::U16 => "u16",
            Ty::S32 => "s32",
            Ty::U32 => "u32",
            Ty::S64 => "s64",
            Ty::U64 => "u64",
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::F32 => "f32",
            Ty::F16x2 => "f16x2",
            Ty::Bf16x2 => "bf16x2",
            Ty::E4m3 => "e4m3",
            Ty::E4m3x2 => "e4m3x2",
            Ty::E5m2x2 => "e5m2x2",
            Ty::E5m2 => "e5m2",
            Ty::E2m1 => "e2m1",
            Ty::Ue4m3 => "ue4m3",
        }
    }

    /// The bank a value of this type lives in. A byte is held zero-extended in a 32-bit register, a half in a
    /// 16-bit one, and the packed formats of a tensor-core operand in 32-bit ones.
    pub fn bank(self) -> Bank {
        match self {
            Ty::Pred => Bank::Pred,
            Ty::B16 | Ty::U16 | Ty::F16 | Ty::Bf16 | Ty::E4m3x2 | Ty::E5m2x2 => Bank::B16,
            Ty::B32 | Ty::U8 | Ty::S32 | Ty::U32 | Ty::F16x2 | Ty::Bf16x2 => Bank::B32,
            Ty::E4m3 | Ty::E5m2 | Ty::E2m1 | Ty::Ue4m3 => Bank::B32,
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
            Ty::Pred | Ty::U8 | Ty::E4m3 | Ty::E5m2 | Ty::E2m1 | Ty::Ue4m3 => 1,
            Ty::B16 | Ty::U16 | Ty::F16 | Ty::Bf16 | Ty::E4m3x2 | Ty::E5m2x2 => 2,
            Ty::B32 | Ty::S32 | Ty::U32 | Ty::F32 | Ty::F16x2 | Ty::Bf16x2 => 4,
            Ty::B64 | Ty::S64 | Ty::U64 => 8,
        }
    }
}

/// A comparison, as `setp` spells it. The ordered float forms are false on a NaN; `Neu` is true on one, and `Nan`
/// holds exactly where an operand is one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cmp {
    Eq,
    Ne,
    Neu,
    Nan,
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
            Cmp::Nan => "nan",
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
/// `Rpi` up). `Approx` stands where a rounding mode would and names the hardware's approximation of the
/// instruction, which rounds as the hardware does.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Round {
    Rn,
    Rni,
    Rzi,
    Rmi,
    Rpi,
    Approx,
}

impl Round {
    pub fn text(self) -> &'static str {
        match self {
            Round::Rn => "rn",
            Round::Rni => "rni",
            Round::Rzi => "rzi",
            Round::Rmi => "rmi",
            Round::Rpi => "rpi",
            Round::Approx => "approx",
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

/// How a warp shuffle picks the lane a thread reads: by exchanging with the lane whose number differs in the given
/// bits, from the lane that many below, or from the lane of that number in its group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Shuffle {
    Bfly,
    Up,
    Idx,
}

impl Shuffle {
    pub fn text(self) -> &'static str {
        match self {
            Shuffle::Bfly => "bfly",
            Shuffle::Up => "up",
            Shuffle::Idx => "idx",
        }
    }
}

/// Which caches an async copy from global memory goes through: every level (4 or 8 bytes), or past the first one
/// (16 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cache {
    All,
    Global,
}

impl Cache {
    pub fn text(self) -> &'static str {
        match self {
            Cache::All => "ca",
            Cache::Global => "cg",
        }
    }
}

/// The tile one tensor-core product of a warp covers: its rows, its columns and the length of the sum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaShape {
    M8n8k4,
    M16n8k16,
    M16n8k32,
    M16n8k64,
}

impl MmaShape {
    pub fn text(self) -> &'static str {
        match self {
            MmaShape::M8n8k4 => "m8n8k4",
            MmaShape::M16n8k16 => "m16n8k16",
            MmaShape::M16n8k32 => "m16n8k32",
            MmaShape::M16n8k64 => "m16n8k64",
        }
    }
}

/// One form of `mma.sync`: the tile, whether B is read row-major (Volta's second form), the type of the
/// multiplicands, the type of the accumulator, and whether every 16 elements of the sum carry a scale of their own
/// (the block-scaled 4-bit form).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Mma {
    pub shape: MmaShape,
    pub b_row: bool,
    pub ab: Ty,
    pub acc: Ty,
    pub block_scale: bool,
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
    /// The hardware's math instructions, which have an approximate form only: the reciprocal square root, two
    /// to a power, the base-two logarithm, the sine, the cosine and the hyperbolic tangent.
    Rsqrt,
    Ex2,
    Lg2,
    Sin,
    Cos,
    Tanh,
    /// The second operand with the sign of the first.
    Copysign,
    Setp(Cmp),
    Selp,
    /// A conversion to the instruction's type from this one.
    Cvt(Ty),
    Ld(Space),
    /// A load of global memory nothing writes while the kernel runs (`ld.global.nc`).
    LdConst,
    St(Space),
    /// An atomic add into memory; the result is what the cell held.
    AtomAdd(Space),
    /// A barrier: of the whole block, or of the named barrier and the thread count given as operands.
    BarSync,
    /// A barrier of the lanes of a warp its mask operand names.
    BarWarpSync,
    /// The mask of the lanes of the warp that run this instruction together.
    Activemask,
    /// A 32-bit value read from another lane of the warp: the value, the lane operand, the clamp and the mask of
    /// the lanes that take part.
    Shfl(Shuffle),
    /// An async copy from global to shared memory: the two addresses, the bytes copied and, when given, how many
    /// of them are read (the rest are written as zeros).
    CpAsync(Cache),
    CpAsyncCommit,
    /// Wait until at most the operand's count of committed copy groups are still in flight.
    CpAsyncWait,
    /// Wait for the grid launched ahead of this one (`griddepcontrol.wait`).
    GridWait,
    /// A warp's load of 8x8 matrices of 16-bit elements from shared memory, each lane giving the address of one
    /// row: two or four matrices, as many as the registers it defines, read by rows or transposed.
    Ldmatrix {
        trans: bool,
    },
    /// One tensor-core product of a warp: the accumulator's registers, then the tuples of A, B and the
    /// accumulator before it; the block-scaled form adds each side's scale register and its selector.
    Mma(Mma),
    /// One tensor-core product of four warps together (`wgmma.mma_async`), of a 64 x `n` tile at the operand type
    /// `ab`: A as a descriptor or a tuple of registers, B's descriptor, the predicate that keeps the accumulator,
    /// the two scales and the transpose bits of the descriptor operands. The accumulator's registers are read
    /// and defined.
    Wgmma {
        n: u16,
        ab: Ty,
    },
    WgmmaFence,
    WgmmaCommit,
    /// Wait until at most the operand's count of committed `wgmma` groups are still in flight.
    WgmmaWait,
    /// Start a barrier of shared memory with the count of arrivals one phase takes.
    MbarrierInit,
    /// One arrival at a barrier; the result is the barrier's state before it.
    MbarrierArrive,
    /// One arrival that also states the bytes of async copies the phase waits for.
    MbarrierArriveExpectTx,
    /// Whether the barrier's phase of the given parity is over; it may suspend the thread while it waits.
    MbarrierTryWait,
    /// Order the block's earlier shared-memory accesses before the async copies and arrivals that follow.
    FenceProxyAsync,
    /// A copy of one box of a tensor from global to shared memory, by the copy engine: the destination, the
    /// descriptor with the box's coordinates, and the barrier the copy's bytes arrive at.
    CpAsyncBulkTensor,
    /// Give back (`inc: false`) or claim the registers a warp may hold, up to the operand's count.
    SetMaxNReg {
        inc: bool,
    },
    Bra,
    Ret,
}

impl Opcode {
    /// Whether the instruction only computes its result from its operands: what the value table may share.
    pub fn is_pure(self) -> bool {
        !matches!(
            self,
            Opcode::Ld(_)
                | Opcode::LdConst
                | Opcode::St(_)
                | Opcode::AtomAdd(_)
                | Opcode::BarSync
                | Opcode::BarWarpSync
                | Opcode::Activemask
                | Opcode::Shfl(_)
                | Opcode::CpAsync(_)
                | Opcode::CpAsyncCommit
                | Opcode::CpAsyncWait
                | Opcode::GridWait
                | Opcode::Ldmatrix { .. }
                | Opcode::Mma(_)
                | Opcode::Wgmma { .. }
                | Opcode::WgmmaFence
                | Opcode::WgmmaCommit
                | Opcode::WgmmaWait
                | Opcode::MbarrierInit
                | Opcode::MbarrierArrive
                | Opcode::MbarrierArriveExpectTx
                | Opcode::MbarrierTryWait
                | Opcode::FenceProxyAsync
                | Opcode::CpAsyncBulkTensor
                | Opcode::SetMaxNReg { .. }
                | Opcode::Bra
                | Opcode::Ret
        )
    }

    pub fn is_load(self) -> bool {
        matches!(self, Opcode::Ld(_) | Opcode::LdConst)
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
    /// Several registers one vector access fills or stores, one move packs into a wider register, or a
    /// tensor-core instruction reads as one fragment.
    Tuple(Vec<Reg>),
    /// Several integer constants in braces: the selector of a block scale.
    Ints(Vec<i64>),
    /// A box of a tensor: the register holding its descriptor's address, and the box's coordinates.
    Tensor(Reg, Vec<Reg>),
    /// The address of a shared declaration of the module, by its position, in the block's shared memory.
    Shared(u32),
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
            Operand::Tensor(descriptor, coords) => std::iter::once(*descriptor)
                .chain(coords.iter().copied())
                .collect(),
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

/// The bytes a matrix load reads at each lane's address, a row of eight halves, and the alignment the copy engine
/// asks of a tensor copy's destination.
pub const ROW_ACCESS_BYTES: i64 = 16;

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

    /// Every register the instruction reads: its operands' and its guard's, and for a `wgmma` product the
    /// accumulator it adds into, which stands once in the instruction.
    pub fn reads(&self) -> Vec<Reg> {
        let accumulated: &[Reg] = match self.opcode {
            Opcode::Wgmma { .. } => &self.results,
            _ => &[],
        };
        self.operands
            .iter()
            .flat_map(Operand::reads)
            .chain(self.guard.map(|guard| guard.pred))
            .chain(accumulated.iter().copied())
            .collect()
    }

    /// The bytes one access of memory covers, with the operands that address it: a load or a store of several
    /// registers, an async copy, a row of a matrix load, the destination of a tensor copy. `None` for an
    /// instruction that reaches one element or no memory.
    pub fn wide_access(&self) -> Option<(i64, Vec<&Operand>)> {
        let addresses = |count: usize| self.operands.iter().take(count).collect();
        match self.opcode {
            Opcode::CpAsync(_) => Some((self.operands.get(2)?.int()?, addresses(2))),
            Opcode::Ldmatrix { .. } | Opcode::CpAsyncBulkTensor => {
                Some((ROW_ACCESS_BYTES, addresses(1)))
            }
            opcode if opcode.is_load() && self.results.len() > 1 => {
                Some((self.results.len() as i64 * self.ty.bytes(), addresses(1)))
            }
            Opcode::St(_) => {
                let lanes = self.operands.iter().map(Operand::lanes).max()?;
                (lanes > 1).then(|| (lanes as i64 * self.ty.bytes(), addresses(1)))
            }
            _ => None,
        }
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
    /// `.pragma "nounroll";` at the head of a loop: ptxas keeps the loop as it is.
    NoUnroll,
}

/// How a parameter is declared: a pointer with the alignment the launch gives its buffer, or a 32-bit integer.
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
