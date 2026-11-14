//! Hopper's instructions: the tensor-core product of four warps together (`wgmma`) with its descriptors and its
//! groups, the copy of a tensor's box by the copy engine with the barriers its bytes arrive at, and the limit on
//! a warp's registers. Each is one instruction with its operands, or the fixed sequence that goes with it.

use emmy_ir::expr::{Expression, SimplifyCtx};
use emmy_ir::field::Shape3;
use emmy_ir::statement::{
    MbarrierInit, MbarrierWait, SetMaxNReg, TmaDescriptor, TmaLoad, WgmmaDescriptor, WgmmaMma,
};
use emmy_ir::{Name, Result};

use super::{Lowerer, Needs, SM_LAST, VTy, Val};
use crate::inst::{Bank, Base, Cmp, Inst, Opcode, Operand, ROW_ACCESS_BYTES, Reg, Space, Ty};

/// `wgmma` assembles for `sm_90a` alone: ptxas refuses it on `sm_100a` and on `sm_120a`.
const WGMMA: Needs = Needs {
    what: "`wgmma`",
    sm: 90..=99,
    arch_specific: true,
    version: (8, 0),
};
/// A barrier of shared memory is Ampere's; what the kernels do with one is Hopper's.
const BARRIER: Needs = Needs {
    what: "`mbarrier.init`",
    sm: 80..=SM_LAST,
    arch_specific: false,
    version: (7, 0),
};
const BARRIER_WAIT: Needs = Needs {
    what: "`mbarrier.try_wait`",
    sm: 90..=SM_LAST,
    arch_specific: false,
    version: (7, 8),
};
/// Every arrival stands behind the fence, and one that states its bytes is Hopper's itself.
const BARRIER_ARRIVAL: Needs = Needs {
    what: "an arrival at an `mbarrier` behind `fence.proxy.async`",
    sm: 90..=SM_LAST,
    arch_specific: false,
    version: (8, 0),
};
/// The destination spelled `.shared::cta`, as the kernels spell it, is PTX 8.6; 8.0 has `.shared::cluster` only.
const TENSOR_COPY: Needs = Needs {
    what: "`cp.async.bulk.tensor`",
    sm: 90..=SM_LAST,
    arch_specific: false,
    version: (8, 6),
};
const REGISTER_LIMIT: Needs = Needs {
    what: "`setmaxnreg`",
    sm: 90..=SM_LAST,
    arch_specific: true,
    version: (8, 0),
};

/// The bits of a field of a matrix descriptor: an address or a byte offset in units of 16 bytes.
const DESCRIPTOR_FIELD: i64 = 0x3FFF;

/// The coordinates a tensor copy takes.
const TENSOR_RANKS: std::ops::RangeInclusive<usize> = 1..=5;

impl Lowerer {
    /// The address of a slot of a shared buffer, as the operand of an instruction that reaches memory.
    fn shared_address(
        &mut self,
        buffer: Name,
        slot: Option<&Expression>,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        if self.buffer(buffer)?.space != Space::Shared {
            return Err(self.refuse(format_args!("{buffer} is no shared buffer")));
        }
        self.address(buffer, slot.map_or(&[], std::slice::from_ref), ctx)
    }

    // --- wgmma -------------------------------------------------------------------------------------------

    /// The descriptor `wgmma` reads an operand through: the slab's address in shared memory and the two byte
    /// offsets between its 8x8 matrices, each in units of 16 bytes, and the slab's swizzle in the top two bits.
    pub(super) fn wgmma_descriptor(
        &mut self,
        node: &WgmmaDescriptor,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        self.requires(&WGMMA)?;
        let mode = node.swizzle;
        let layout: i64 = match mode.split('@').next() {
            Some("NONE") => 0,
            Some("B128") => 1,
            Some("B64") => 2,
            Some("B32") => 3,
            _ => return Err(self.no_rule(format_args!("a slab laid out as {mode}"))),
        };
        let address = self.shared_address(node.smem, node.smem_index.as_ref(), ctx)?;
        let (base, offset) = match address {
            Operand::Addr(Base::Reg(base), offset) => (base, offset),
            _ => return Err(self.refuse("a slab has an address")),
        };
        if i64::from(self.b.alignment(base)) % ROW_ACCESS_BYTES != 0
            || offset % ROW_ACCESS_BYTES != 0
        {
            let name = node.smem;
            return Err(self.refuse(format_args!(
                "a descriptor holds its address in units of {ROW_ACCESS_BYTES} bytes, and no such alignment is \
                 proven of {name}"
            )));
        }
        let mut value = |opcode, ty: Ty, operands: Vec<Operand>| {
            self.b
                .value(Inst::new(opcode, ty, Vec::new(), operands), ty.bank())
        };
        let start = match offset {
            0 => base,
            offset => value(
                Opcode::Add,
                Ty::S32,
                vec![base.into(), Operand::Int(offset)],
            ),
        };
        let units = value(Opcode::Shr, Ty::U32, vec![start.into(), Operand::Int(4)]);
        let field = value(
            Opcode::And,
            Ty::B32,
            vec![units.into(), Operand::Int(DESCRIPTOR_FIELD)],
        );
        let wide = value(Opcode::Cvt(Ty::U32), Ty::U64, vec![field.into()]);
        let in_units = |bytes: i64| (bytes >> 4) & DESCRIPTOR_FIELD;
        let fixed = in_units(node.lbo_bytes) << 16 | in_units(node.sbo_bytes) << 32 | layout << 62;
        let descriptor = value(Opcode::Or, Ty::B64, vec![wide.into(), Operand::Int(fixed)]);
        self.bind(node.name, Val::reg(descriptor, VTy::I64));
        Ok(())
    }

    /// The register a descriptor's name holds.
    fn descriptor(&mut self, name: Name) -> Result<Reg> {
        let val = self.lookup(name)?;
        match (val.op.reg(), val.ty) {
            (Some(reg), VTy::I64) => Ok(reg),
            _ => Err(self.refuse(format_args!("{name} is no matrix descriptor"))),
        }
    }

    /// One `wgmma.mma_async` of a 64 x N tile. The accumulator is the N / 8 fragments of four floats a warp holds
    /// side by side, registers in the fragments' order; A is a descriptor or the four registers of an A fragment;
    /// B is a descriptor. A predicate says whether the product adds to the accumulator or replaces it.
    pub(super) fn wgmma(&mut self, node: &WgmmaMma) -> Result<()> {
        self.requires(&WGMMA)?;
        // The statement checks its own tile, 64 by a multiple of eight by 16, with a fragment every eight
        // columns and A named once.
        let Shape3(_, n, _) = node.shape;
        let ab = match &*node.ab_dtype {
            "f16" => Ty::F16,
            "bf16" => Ty::Bf16,
            ab => return Err(self.no_rule(format_args!("a `wgmma` at {ab}"))),
        };
        let mut accumulator = Vec::with_capacity(node.c_frags.len() * 4);
        for name in node.c_frags.iter() {
            accumulator.extend(self.tensor_fragment(*name, 4, VTy::F32)?);
        }
        let a = match (node.a_desc, node.a_frag) {
            (Some(descriptor), None) => Operand::Reg(self.descriptor(descriptor)?),
            (None, Some(fragment)) => {
                Operand::Tuple(self.tensor_fragment(fragment, 4, VTy::U32)?)
            }
            _ => return Err(self.refuse("A is a descriptor or a fragment")),
        };
        let b = self.descriptor(node.b_desc)?;
        let keeps = vec![Operand::Int(node.scale_d), Operand::Int(0)];
        let keeps = Inst::new(Opcode::Setp(Cmp::Ne), Ty::B32, Vec::new(), keeps);
        let keeps = self.b.value(keeps, Bank::Pred);
        // The scales of A and B, which are one, then the transpose bit of each operand read through a descriptor.
        let mut operands = vec![a, b.into(), keeps.into(), Operand::Int(1), Operand::Int(1)];
        if node.a_desc.is_some() {
            operands.push(Operand::Int(node.trans_a));
        }
        operands.push(Operand::Int(node.trans_b));
        let product = Opcode::Wgmma { n: n as u16, ab };
        self.b
            .emit(Inst::new(product, Ty::F32, accumulator, operands));
        Ok(())
    }

    /// The fence ahead of a product, the commit of the products issued, or the wait until at most `group` committed
    /// groups are in flight.
    pub(super) fn wgmma_group(&mut self, opcode: Opcode, group: Option<i64>) -> Result<()> {
        self.requires(&WGMMA)?;
        let operands = group.map(Operand::Int).into_iter().collect();
        self.b
            .emit(Inst::new(opcode, Ty::B32, Vec::new(), operands));
        Ok(())
    }

    // --- the tensor copy and its barriers ----------------------------------------------------------------

    /// A descriptor of a tensor is built by the host and passed by the launch: the statement declares it, and
    /// writes no instruction.
    pub(super) fn tma_descriptor(&mut self, node: &TmaDescriptor) -> Result<()> {
        self.requires(&TENSOR_COPY)?;
        if !self.descriptors.contains_key(&node.name) {
            let name = node.name;
            return Err(self.refuse(format_args!("{name} is no descriptor the launch passes")));
        }
        Ok(())
    }

    /// One copy of a box of a tensor into a shared slab. The box's coordinates are given outermost first and the
    /// instruction takes them innermost first.
    pub(super) fn tma_load(&mut self, node: &TmaLoad, ctx: &SimplifyCtx) -> Result<()> {
        self.requires(&TENSOR_COPY)?;
        let Some(descriptor) = self.descriptors.get(&node.desc).copied() else {
            let name = node.desc;
            return Err(self.refuse(format_args!("{name} is no descriptor the launch passes")));
        };
        if !TENSOR_RANKS.contains(&node.coords.len()) {
            let rank = node.coords.len();
            return Err(self.no_rule(format_args!("a tensor copy of {rank} coordinates")));
        }
        let mut coordinates = Vec::with_capacity(node.coords.len());
        for coordinate in node.coords.iter().rev() {
            let val = self.int(coordinate, ctx)?;
            let val = Val {
                op: self.as_int(&val, VTy::I32),
                ty: VTy::I32,
            };
            coordinates.push(self.register(&val));
        }
        let flat = self.flat_index(node.smem, &node.smem_index)?;
        let to = self.shared_address(node.smem, Some(&flat), ctx)?;
        let barrier = self.shared_address(node.mbar, node.mbar_slot.as_ref(), ctx)?;
        let operands = vec![to, Operand::Tensor(descriptor, coordinates), barrier];
        let copy = Inst::new(Opcode::CpAsyncBulkTensor, Ty::B32, Vec::new(), operands);
        self.b.emit(copy);
        Ok(())
    }

    pub(super) fn mbarrier_init(&mut self, node: &MbarrierInit, ctx: &SimplifyCtx) -> Result<()> {
        self.requires(&BARRIER)?;
        let barrier = self.shared_address(node.mbar, node.slot.as_ref(), ctx)?;
        let operands = vec![barrier, Operand::Int(node.count)];
        self.b.emit(Inst::new(
            Opcode::MbarrierInit,
            Ty::B64,
            Vec::new(),
            operands,
        ));
        Ok(())
    }

    /// One arrival at a barrier, stating the bytes of async copies its phase waits for when `bytes` is given. The
    /// fence ahead of it orders the block's earlier shared-memory accesses before the copy engine's: the
    /// barrier's own start ahead of a fill, a consumer's reads of a slab ahead of its refill.
    pub(super) fn mbarrier_arrive(
        &mut self,
        mbar: Name,
        slot: &Option<Expression>,
        bytes: Option<i64>,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        self.requires(&BARRIER_ARRIVAL)?;
        let barrier = self.shared_address(mbar, slot.as_ref(), ctx)?;
        let fence = Inst::new(Opcode::FenceProxyAsync, Ty::B32, Vec::new(), Vec::new());
        self.b.emit(fence);
        let state = self.b.reg(Bank::B64);
        let (opcode, operands) = match bytes {
            Some(bytes) => (
                Opcode::MbarrierArriveExpectTx,
                vec![barrier, Operand::Int(bytes)],
            ),
            None => (Opcode::MbarrierArrive, vec![barrier]),
        };
        self.b
            .emit(Inst::new(opcode, Ty::B64, vec![state], operands));
        Ok(())
    }

    /// Wait for the barrier's phase of the given parity to end. The instruction may return before it has, so it
    /// is asked again until it says so; it suspends the thread while it waits.
    pub(super) fn mbarrier_wait(&mut self, node: &MbarrierWait, ctx: &SimplifyCtx) -> Result<()> {
        self.requires(&BARRIER_WAIT)?;
        let barrier = self.shared_address(node.mbar, node.slot.as_ref(), ctx)?;
        let phase = self.int(&node.phase, ctx)?;
        let phase = self.as_int(&phase, VTy::I32);
        let over = self.b.reg(Bank::Pred);
        let wait = Opcode::MbarrierTryWait;
        let wait = Inst::new(wait, Ty::B64, vec![over], vec![barrier, phase]);
        self.b.repeat_until(over, wait);
        Ok(())
    }

    /// Give back or claim registers for the warp, up to the count.
    pub(super) fn set_max_registers(&mut self, node: &SetMaxNReg) -> Result<()> {
        self.requires(&REGISTER_LIMIT)?;
        let inc = match &*node.direction {
            "inc" => true,
            "dec" => false,
            direction => {
                return Err(self.refuse(format_args!("a register limit moved {direction}")));
            }
        };
        let limit = Opcode::SetMaxNReg { inc };
        let operands = vec![Operand::Int(node.count)];
        self.b.emit(Inst::new(limit, Ty::U32, Vec::new(), operands));
        Ok(())
    }
}
