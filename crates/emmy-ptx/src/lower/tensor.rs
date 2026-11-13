//! The tensor cores of a warp: a fragment as registers, the matrix load from shared memory, `mma.sync` in each of
//! its forms, and the statements that stay in registers between two products: the promote of a half accumulator,
//! the repack of an accumulator into an operand, the load of block scales.
//!
//! A fragment read from or written to memory element by element (a fragment load from global memory, the staged
//! drains that are no `ldmatrix`, `RegStore`, `FragmentApply`, `FragmentRowReduce`) has no rule here: its
//! instructions follow from the lane-to-element table.

use emmy_ir::dtype::DType;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Builtin, Expr, Expression, SimplifyCtx};
use emmy_ir::field::Shape3;
use emmy_ir::statement::{
    BlockScaleLoad, FragmentPromote, FragmentRepack, LdmatrixLoad, Load, MmaSyncPtx, RegFragment,
    Write,
};
use emmy_ir::{Name, Result};

use super::{Lowerer, Needs, SM_LAST, VTy, Val};
use crate::inst::{Bank, Inst, Mma, MmaShape, Opcode, Operand, Reg, Round, Shuffle, Space, Ty};

/// The lanes of a warp.
const WARP: i64 = 32;

/// Which lanes supply the block scales of each side: the selector the instruction takes beside the scale register.
/// With selector `t`, A's scales come from the lanes whose index in their group of four is `2t` or `2t + 1`, B's
/// from the lanes where it is `t`. Both were found by probing the hardware; ptxas takes 0 or 1 for either.
const SCALE_SELECTOR_A: i64 = 0;
const SCALE_SELECTOR_B: i64 = 1;

const LDMATRIX: Needs = Needs {
    what: "`ldmatrix`",
    sm: 75..=SM_LAST,
    arch_specific: false,
    version: (6, 5),
};
const PACKED_CONVERSION: Needs = Needs {
    what: "a conversion of two floats into packed halves (`cvt.f16x2.f32`)",
    sm: 80..=SM_LAST,
    arch_specific: false,
    version: (7, 0),
};
const MMA_VOLTA: Needs = Needs {
    what: "`mma.sync` m8n8k4",
    sm: 70..=SM_LAST,
    arch_specific: false,
    version: (6, 4),
};
const MMA_HALVES: Needs = Needs {
    what: "`mma.sync` m16n8k16",
    sm: 80..=SM_LAST,
    arch_specific: false,
    version: (7, 0),
};
const MMA_BYTES: Needs = Needs {
    what: "`mma.sync` m16n8k32 of 8-bit floats",
    sm: 89..=SM_LAST,
    arch_specific: false,
    version: (8, 4),
};
const MMA_BLOCK_SCALED: Needs = Needs {
    what: "the block-scaled `mma.sync` m16n8k64",
    sm: 120..=SM_LAST,
    arch_specific: true,
    version: (8, 7),
};

fn thread() -> Expression {
    Expression::new(Expr::Builtin(Builtin::ThreadIdxX))
}

fn apply(op: Operation, left: Expression, right: i64) -> Expression {
    Expression::binary(op, left, Expression::int(right))
}

/// The lane of a thread in its warp.
fn lane() -> Expression {
    apply(Operation::Remainder, thread(), WARP)
}

fn sum(left: Expression, right: Expression) -> Expression {
    Expression::binary(Operation::Add, left, right)
}

fn times(left: Expression, right: i64) -> Expression {
    apply(Operation::Multiply, left, right)
}

fn over(left: Expression, right: i64) -> Expression {
    apply(Operation::FloorDivide, left, right)
}

fn modulo(left: Expression, right: i64) -> Expression {
    apply(Operation::Remainder, left, right)
}

fn masked(left: Expression, mask: i64) -> Expression {
    apply(Operation::BitwiseAnd, left, mask)
}

/// The bits of both sides; a side that is the constant zero adds none.
fn either(left: Expression, right: Expression) -> Expression {
    match (left.int_literal(), right.int_literal()) {
        (Some(0), _) => right,
        (_, Some(0)) => left,
        _ => Expression::binary(Operation::BitwiseOr, left, right),
    }
}

/// The shift and the row mask of a swizzle mode's exchange of 16-byte chunks: `e ^ (((e >> shift) & mask) << 3)`
/// over the index of a 16-bit element. `B128`, `B64` and `B32` exchange among 8, 4 and 2 chunks; `<mode>@<shift>`
/// states the shift of a slab whose row is not one swizzle row. `None` for a mode that is none of them.
fn swizzle_exchange(mode: &str) -> Option<Option<(i64, i64)>> {
    if mode == "NONE" {
        return Some(None);
    }
    let (base, shift) = match mode.split_once('@') {
        Some((base, shift)) => (base, shift.parse().ok()?),
        None => (mode, 6),
    };
    let mask = match base {
        "B128" => 7,
        "B64" => 3,
        "B32" => 1,
        _ => return None,
    };
    Some(Some((shift, mask)))
}

fn swizzled(element: Expression, (shift, mask): (i64, i64)) -> Expression {
    let row = masked(apply(Operation::RightShift, element.clone(), shift), mask);
    Expression::binary(
        Operation::BitwiseXor,
        element,
        apply(Operation::LeftShift, row, 3),
    )
}

impl Lowerer {
    // --- fragments ---------------------------------------------------------------------------------------

    pub(super) fn is_fragment(&self, name: Name) -> bool {
        matches!(self.find(name), Some((_, val)) if matches!(val.op, Operand::Tuple(_)))
    }

    /// The registers of the fragment `name`, and what each holds: a float, or 32 bits of packed elements.
    fn fragment(&self, name: Name) -> Result<(Vec<Reg>, VTy)> {
        match self.find(name) {
            Some((_, val)) => match &val.op {
                Operand::Tuple(regs) => Ok((regs.clone(), val.ty)),
                _ => Err(self.refuse(format_args!("{name} is no fragment"))),
            },
            None => Err(self.refuse(format_args!(
                "the fragment {name} is read before it is declared"
            ))),
        }
    }

    /// The fragment `name` with exactly `count` registers of `ty`.
    pub(super) fn tensor_fragment(&self, name: Name, count: usize, ty: VTy) -> Result<Vec<Reg>> {
        let (regs, held) = self.fragment(name)?;
        if regs.len() != count || held != ty {
            let (has, kind) = (regs.len(), held.text());
            let wanted = ty.text();
            return Err(self.refuse(format_args!(
                "the fragment {name} holds {has} registers of {kind}, read as {count} of {wanted}"
            )));
        }
        Ok(regs)
    }

    /// A new fragment `name` of `count` registers of `ty` in the innermost body. Every statement that names the
    /// fragment defines its registers again, so each is a state.
    fn declare_fragment(&mut self, name: Name, count: usize, ty: VTy) -> Vec<Reg> {
        let regs: Vec<Reg> = (0..count).map(|_| self.b.state(ty.bank())).collect();
        let val = Val {
            op: Operand::Tuple(regs.clone()),
            ty,
        };
        if let Some(scope) = self.names.last_mut() {
            scope.insert(name, val);
        }
        regs
    }

    /// A fragment is its registers: 32 bits of packed elements each for an operand or an accumulator of halves,
    /// a float each for an accumulator of floats. An accumulator starts at zero; an operand holds nothing until a
    /// statement loads it. A declaration of no registers is one register by the fragment's own name: a lane's
    /// block scales.
    pub(super) fn reg_fragment(&mut self, node: &RegFragment) -> Result<()> {
        let role = &*node.role;
        if node.is_scalar() {
            let reg = self.b.state(Bank::B32);
            if let Some(scope) = self.names.last_mut() {
                scope.insert(node.name, Val::reg(reg, VTy::U32));
            }
            return Ok(());
        }
        let count = node.registers()?;
        let halves = matches!(node.dtype, DType::F16 | DType::BF16);
        let ty = match role {
            "c" if !halves => VTy::F32,
            _ => VTy::U32,
        };
        let regs = self.declare_fragment(node.name, count, ty);
        if role == "c" {
            self.zero(&regs, ty);
        }
        Ok(())
    }

    fn zero(&mut self, regs: &[Reg], ty: VTy) {
        let zero = match ty {
            VTy::F32 => Operand::f32(0.0),
            _ => Operand::Int(0),
        };
        for reg in regs {
            let clear = Inst::new(Opcode::Mov, ty.held(), vec![*reg], vec![zero.clone()]);
            self.b.emit(clear);
        }
    }

    /// The registers of a fragment a `Load` or a `Write` names by a constant index, one a name or a value.
    fn fragment_registers(
        &self,
        name: Name,
        index: &[Expression],
        count: usize,
    ) -> Result<(Vec<Reg>, VTy)> {
        let (regs, ty) = self.fragment(name)?;
        let first = match index {
            [first] => first.int_literal().and_then(|at| usize::try_from(at).ok()),
            _ => None,
        };
        let named = first.and_then(|first| regs.get(first..first + count));
        match named {
            Some(named) => Ok((named.to_vec(), ty)),
            None => Err(self.refuse(format_args!(
                "a register of the fragment {name} is named by a constant below {}",
                regs.len()
            ))),
        }
    }

    /// A load from a fragment gives each name what a register of it holds now.
    pub(super) fn fragment_load(&mut self, load: &Load) -> Result<()> {
        let (regs, ty) = self.fragment_registers(load.input, &load.index, load.names.len())?;
        if load.dtype.and_then(VTy::of) != Some(ty) {
            let (input, held) = (load.input, ty.text());
            return Err(self.refuse(format_args!("a register of {input} holds {held}")));
        }
        for (name, reg) in load.names.iter().zip(regs) {
            let val = self.named(Val::reg(reg, ty));
            self.bind(*name, val);
        }
        Ok(())
    }

    /// A store into a fragment moves each value into a register of it.
    pub(super) fn fragment_store(&mut self, write: &Write) -> Result<()> {
        let output = write.output;
        if write.atomic || &*write.swizzle != "NONE" {
            return Err(self.refuse(format_args!("{output} is a fragment, stored to as it is")));
        }
        let (regs, ty) = self.fragment_registers(output, &write.index, write.values.len())?;
        for (name, reg) in write.values.iter().zip(regs) {
            let val = self.lookup(*name)?;
            let value = match (ty, val.ty) {
                (VTy::U32, VTy::I32 | VTy::U32) => self.as_int(&val, VTy::U32),
                (ty, held) if ty == held => val.op,
                (ty, held) => {
                    let (ty, held) = (ty.text(), held.text());
                    return Err(self.refuse(format_args!(
                        "a register of {output} holds {ty}, {name} is {held}"
                    )));
                }
            };
            self.b
                .emit(Inst::new(Opcode::Mov, ty.held(), vec![reg], vec![value]));
        }
        Ok(())
    }

    // --- the matrix load ---------------------------------------------------------------------------------

    /// The element a lane's row starts at in the slab: the tile's own index and the lane's offset in the tile,
    /// through the slab's swizzle when it has one. The exchange is linear over parts that share no bit, so where
    /// the tile's row and column are multiples of what a lane's offset spans, the lane's part is swizzled alone,
    /// once for every load of the kernel, and the tile's row is added after.
    fn slab_element(
        &mut self,
        node: &LdmatrixLoad,
        ldm: i64,
        offset: Expression,
        (lane_rows, lane_columns): (i64, i64),
        ctx: &SimplifyCtx,
    ) -> Result<Expression> {
        let buffer = node.src_buffer;
        let Some(exchange) = swizzle_exchange(&node.swizzle) else {
            let mode = node.swizzle;
            return Err(self.no_rule(format_args!("a slab laid out as {mode}")));
        };
        let Some(exchange) = exchange else {
            return Ok(sum(self.flat_index(buffer, &node.src_index)?, offset));
        };
        let (shift, mask) = exchange;
        let by_rows = self.extents.get(&buffer).and_then(|e| e.last()) == Some(&ldm);
        if let [row, column] = &node.src_index[..]
            && by_rows
            && ldm >= 8
            && ldm & (ldm - 1) == 0
        {
            // The rows of the slab the exchange reads its row index from: a tile that starts on a multiple of
            // them, and of the rows a lane spans, shares no bit with the lane's offset.
            let field =
                1 << (shift + i64::from((mask + 1).ilog2()) - i64::from(ldm.ilog2())).max(0);
            let (row_val, column_val) = (self.int(row, ctx)?, self.int(column, ctx)?);
            let multiple_of = |this: &Lowerer, val: &Val, of: i64| this.multiple(&val.op) % of == 0;
            if multiple_of(self, &row_val, lane_rows.max(field))
                && multiple_of(self, &column_val, lane_columns)
            {
                let lane_part = swizzled(offset, exchange);
                let within = match column.int_literal() {
                    Some(column) => match column ^ (((column >> shift) & mask) << 3) {
                        0 => lane_part,
                        moved => apply(Operation::BitwiseXor, lane_part, moved),
                    },
                    None => Expression::binary(
                        Operation::BitwiseXor,
                        lane_part,
                        swizzled(column.clone(), exchange),
                    ),
                };
                return Ok(sum(within, times(row.clone(), ldm)));
            }
        }
        let flat = self.flat_index(buffer, &node.src_index)?;
        Ok(swizzled(sum(flat, offset), exchange))
    }

    /// One `ldmatrix` of an operand staged in shared memory: each lane gives the address of one row of eight
    /// halves. A is four matrices, 16 rows by two blocks of eight columns. B from a slab with K along its rows is
    /// two transposed matrices of 16 rows; from a slab with N along its rows (a transposed B), two plain ones.
    /// Two B fragments side by side in the slab are one load of four matrices.
    pub(super) fn ldmatrix(&mut self, node: &LdmatrixLoad, ctx: &SimplifyCtx) -> Result<()> {
        self.requires(&LDMATRIX)?;
        let buffer = self.buffer(node.src_buffer)?;
        let halves = matches!(buffer.elem.dtype, DType::F16 | DType::BF16);
        if buffer.space != Space::Shared || !halves {
            let (name, dtype) = (node.src_buffer, buffer.elem.dtype);
            return Err(self.refuse(format_args!(
                "`ldmatrix` reads a shared slab of 16-bit elements, and {name} holds {dtype}"
            )));
        }
        let ldm = node.ldm;
        let lane = lane();
        let rows_of = |count: i64| times(modulo(lane.clone(), count), ldm);
        let (trans, regs, offset, spans) = match (&*node.role, node.pair_frag, node.b_trans) {
            ("b", Some(pair), b_trans) => {
                let mut regs = self.tensor_fragment(node.frag, 2, VTy::U32)?;
                regs.extend(self.tensor_fragment(pair, 2, VTy::U32)?);
                let offset = if b_trans {
                    let row = sum(modulo(lane.clone(), 8), times(over(lane.clone(), 16), 8));
                    sum(times(row, ldm), times(modulo(over(lane.clone(), 8), 2), 8))
                } else {
                    sum(rows_of(16), times(over(lane.clone(), 16), 8))
                };
                (!b_trans, regs, offset, (16, 16))
            }
            ("a", None, _) => {
                let regs = self.tensor_fragment(node.frag, 4, VTy::U32)?;
                let offset = sum(rows_of(16), times(over(lane.clone(), 16), 8));
                (false, regs, offset, (16, 16))
            }
            ("b", None, true) => {
                let regs = self.tensor_fragment(node.frag, 2, VTy::U32)?;
                let offset = sum(rows_of(8), times(modulo(over(lane.clone(), 8), 2), 8));
                (false, regs, offset, (8, 16))
            }
            ("b", None, false) => {
                let regs = self.tensor_fragment(node.frag, 2, VTy::U32)?;
                (true, regs, rows_of(16), (16, 8))
            }
            (role, ..) => return Err(self.refuse(format_args!("`ldmatrix` of role {role}"))),
        };
        let element = self.slab_element(node, ldm, offset, spans, ctx)?;
        let address = self.address(node.src_buffer, &[element], ctx)?;
        let load = Opcode::Ldmatrix { trans };
        self.b.emit(Inst::new(load, Ty::B16, regs, vec![address]));
        Ok(())
    }

    // --- mma.sync ----------------------------------------------------------------------------------------

    /// One `mma.sync`: the accumulator's registers are defined again from the operands' and their own. The form
    /// follows from the tile and the two types; each form names the registers its fragments hold and the target
    /// that has it.
    pub(super) fn mma_sync(&mut self, node: &MmaSyncPtx) -> Result<()> {
        let Shape3(m, n, k) = node.shape;
        let (ab, acc) = (&*node.ab_dtype, &*node.c_dtype);
        let scaled = node.sfa_frag.is_some();
        if scaled != node.sfb_frag.is_some() {
            return Err(self.refuse("block scales stand on both sides of a product or on neither"));
        }
        // The tile, the operand type, the accumulator type; then the registers of A and of the accumulator.
        let form = match ((m, n, k), ab, acc) {
            ((8, 8, 4), "f16", "f32") => Some((MmaShape::M8n8k4, Ty::F16, 2, 8, &MMA_VOLTA)),
            ((8, 8, 4), "f16", "f16") => Some((MmaShape::M8n8k4, Ty::F16, 2, 4, &MMA_VOLTA)),
            ((16, 8, 16), "f16", "f32") => Some((MmaShape::M16n8k16, Ty::F16, 4, 4, &MMA_HALVES)),
            ((16, 8, 16), "bf16", "f32") => Some((MmaShape::M16n8k16, Ty::Bf16, 4, 4, &MMA_HALVES)),
            ((16, 8, 16), "f16", "f16") => Some((MmaShape::M16n8k16, Ty::F16, 4, 2, &MMA_HALVES)),
            ((16, 8, 32), "e4m3", "f32") => Some((MmaShape::M16n8k32, Ty::E4m3, 4, 4, &MMA_BYTES)),
            ((16, 8, 32), "e5m2", "f32") => Some((MmaShape::M16n8k32, Ty::E5m2, 4, 4, &MMA_BYTES)),
            ((16, 8, 64), "e2m1", "f32") => {
                Some((MmaShape::M16n8k64, Ty::E2m1, 4, 4, &MMA_BLOCK_SCALED))
            }
            _ => None,
        };
        let form = form.filter(|(shape, ..)| {
            (*shape == MmaShape::M16n8k64) == scaled
                && (*shape == MmaShape::M8n8k4 || !node.b_row_major)
        });
        let Some((shape, ab_ty, a_count, c_count, needs)) = form else {
            let b = if node.b_row_major {
                " with B by rows"
            } else {
                ""
            };
            let scales = if scaled { " with block scales" } else { "" };
            return Err(self.no_rule(format_args!(
                "an `mma.sync` of m{m}n{n}k{k} at {ab} into {acc}{b}{scales}"
            )));
        };
        self.requires(needs)?;
        let floats = acc == "f32";
        let c_ty = if floats { VTy::F32 } else { VTy::U32 };
        let a = self.tensor_fragment(node.a_frag, a_count, VTy::U32)?;
        let b = self.tensor_fragment(node.b_frag, 2, VTy::U32)?;
        let c = self.tensor_fragment(node.c_frag, c_count, c_ty)?;
        let mut operands = vec![
            Operand::Tuple(a),
            Operand::Tuple(b),
            Operand::Tuple(c.clone()),
        ];
        if let (Some(sfa), Some(sfb)) = (node.sfa_frag, node.sfb_frag) {
            // Each side's scales: the register of the four of a lane's 64 elements, then the byte of it the
            // instruction starts at and the selector of the lanes that supply.
            for (scales, selector) in [(sfa, SCALE_SELECTOR_A), (sfb, SCALE_SELECTOR_B)] {
                let val = self.lookup(scales)?;
                let reg = match (val.op.reg(), val.ty) {
                    (Some(reg), VTy::U32) => reg,
                    _ => {
                        return Err(
                            self.refuse(format_args!("{scales} is no register of block scales"))
                        );
                    }
                };
                operands.push(Operand::Tuple(vec![reg]));
                operands.push(Operand::Ints(vec![0, selector]));
            }
        }
        let mma = Mma {
            shape,
            b_row: node.b_row_major,
            ab: ab_ty,
            acc: if floats { Ty::F32 } else { Ty::F16 },
            block_scale: scaled,
        };
        let ty = if floats { Ty::F32 } else { Ty::B32 };
        self.b.emit(Inst::new(Opcode::Mma(mma), ty, c, operands));
        Ok(())
    }

    // --- between two products ----------------------------------------------------------------------------

    /// The 32 bits `from` holds in the lane `lane` names, every lane of the warp taking part.
    fn taken_from(&mut self, from: Reg, lane: &Operand) -> Reg {
        let to = self.b.reg(Bank::B32);
        let operands = vec![
            from.into(),
            lane.clone(),
            Operand::Int(WARP - 1),
            Operand::Int(-1),
        ];
        let shuffle = Opcode::Shfl(Shuffle::Idx);
        self.b.emit(Inst::new(shuffle, Ty::B32, vec![to], operands));
        to
    }

    /// A 32-bit integer expression over the thread's lane, as an operand.
    fn lane_value(&mut self, expression: &Expression, ctx: &SimplifyCtx) -> Result<Operand> {
        let val = self.int(expression, ctx)?;
        Ok(self.as_int(&val, VTy::I32))
    }

    fn word(&mut self, opcode: Opcode, ty: Ty, operands: Vec<Operand>) -> Reg {
        self.b
            .value(Inst::new(opcode, ty, Vec::new(), operands), Bank::B32)
    }

    /// Two floats as the halves of one register, the first the low one, each rounded to nearest: one packed
    /// conversion.
    fn packed_pair(&mut self, low: Reg, high: Reg, to: Ty, into: Option<Reg>) -> Reg {
        let pair = into.unwrap_or_else(|| self.b.reg(Bank::B32));
        let operands = vec![high.into(), low.into()];
        let pack = Inst::new(Opcode::Cvt(Ty::F32), to, vec![pair], operands).rounded(Round::Rn);
        self.b.emit(pack);
        pair
    }

    /// The same pair on a target with no packed conversion: each float converted alone, then joined.
    fn joined_pair(&mut self, low: Reg, high: Reg) -> Reg {
        let halves: Vec<Reg> = [low, high]
            .into_iter()
            .map(|float| {
                let half = self.b.reg(Bank::B16);
                let convert = Inst::new(
                    Opcode::Cvt(Ty::F32),
                    Ty::F16,
                    vec![half],
                    vec![float.into()],
                );
                self.b.emit(convert.rounded(Round::Rn));
                half
            })
            .collect();
        let pair = self.b.reg(Bank::B32);
        let join = Inst::new(
            Opcode::Mov,
            Ty::B32,
            vec![pair],
            vec![Operand::Tuple(halves)],
        );
        self.b.emit(join);
        pair
    }

    /// Add the two halves `packed` holds, widened, into the floats `low` and `high`.
    fn add_halves(&mut self, packed: Reg, low: Reg, high: Reg) {
        let halves = vec![self.b.reg(Bank::B16), self.b.reg(Bank::B16)];
        let split = Inst::new(Opcode::Mov, Ty::B32, halves.clone(), vec![packed.into()]);
        self.b.emit(split);
        for (half, float) in halves.into_iter().zip([low, high]) {
            let wide = self.b.reg(Bank::F32);
            let widen = Inst::new(Opcode::Cvt(Ty::F16), Ty::F32, vec![wide], vec![half.into()]);
            self.b.emit(widen);
            let add = self.float_inst(Opcode::Add, VTy::F32, vec![float.into(), wide.into()]);
            self.b.emit(Inst {
                results: vec![float],
                ..add
            });
        }
    }

    /// Fold an accumulator of halves into its accumulator of floats, and zero it. On Volta a lane's halves are
    /// one row of the tile and its floats two, so each lane first takes from the lane that holds its row's pair.
    pub(super) fn fragment_promote(
        &mut self,
        node: &FragmentPromote,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        match &*node.fragment_layout {
            "m16n8k16" => {
                let floats = self.tensor_fragment(node.dst, 4, VTy::F32)?;
                let halves = self.tensor_fragment(node.src, 2, VTy::U32)?;
                for (packed, pair) in halves.iter().zip(floats.chunks(2)) {
                    self.add_halves(*packed, pair[0], pair[1]);
                }
                self.zero(&halves, VTy::U32);
            }
            "m8n8k4" => {
                let floats = self.tensor_fragment(node.dst, 8, VTy::F32)?;
                let halves = self.tensor_fragment(node.src, 4, VTy::U32)?;
                let lane = lane();
                let upper = Expression::binary(
                    Operation::NotEqual,
                    masked(lane.clone(), 2),
                    Expression::int(0),
                );
                let upper = self.pred(&upper, ctx)?;
                for (p, pair) in floats.chunks(2).enumerate() {
                    let source = either(
                        masked(lane.clone(), !2 & (WARP - 1)),
                        Expression::int((p as i64 & 1) << 1),
                    );
                    let source = self.lane_value(&source, ctx)?;
                    let low = self.taken_from(halves[p & 2], &source);
                    let high = self.taken_from(halves[(p & 2) + 1], &source);
                    let operands = vec![high.into(), low.into(), upper.into()];
                    let packed = self.word(Opcode::Selp, Ty::B32, operands);
                    self.add_halves(packed, pair[0], pair[1]);
                }
                self.zero(&halves, VTy::U32);
            }
            layout => {
                return Err(self.no_rule(format_args!("a promote in the layout {layout}")));
            }
        }
        Ok(())
    }

    /// The operand fragment a repack defines: the one of its name in scope, else a new one.
    fn repacked(&mut self, name: Name, count: usize) -> Result<Vec<Reg>> {
        if self.is_fragment(name) {
            self.tensor_fragment(name, count, VTy::U32)
        } else {
            Ok(self.declare_fragment(name, count, VTy::U32))
        }
    }

    /// An accumulator of floats as an operand of the next product, in registers.
    ///
    /// In the m16n8k16 layout a lane's four floats are two rows by one column pair, which is where A's halves
    /// stand: two accumulators side by side along K are one A fragment, a packed conversion a register. B needs
    /// the other orientation, so each lane takes the packed pairs of the two lanes that hold its column's
    /// rows and keeps the half of each that is its own.
    ///
    /// On Volta one accumulator covers 16 by 16 through four groups of lanes; `part` picks which four columns
    /// become an A fragment, or which four rows a B fragment, and the lanes exchange the same way.
    pub(super) fn fragment_repack(
        &mut self,
        node: &FragmentRepack,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let (layout, role, ab) = (&*node.fragment_layout, &*node.role, &*node.ab_dtype);
        let lane = lane();
        match (layout, role, &node.srcs[..]) {
            ("m16n8k16", "a", [first, second]) => {
                let to = match ab {
                    "f16" => Ty::F16x2,
                    "bf16" => Ty::Bf16x2,
                    _ => return Err(self.no_rule(format_args!("a repack to {ab}"))),
                };
                self.requires(&PACKED_CONVERSION)?;
                let mut floats = self.tensor_fragment(*first, 4, VTy::F32)?;
                floats.extend(self.tensor_fragment(*second, 4, VTy::F32)?);
                let packed = self.repacked(node.frag, 4)?;
                for (pair, into) in floats.chunks(2).zip(packed) {
                    self.packed_pair(pair[0], pair[1], to, Some(into));
                }
            }
            ("m16n8k16", "b", [source]) if ab == "f16" => {
                self.requires(&PACKED_CONVERSION)?;
                let floats = self.tensor_fragment(*source, 4, VTy::F32)?;
                let packed = self.repacked(node.frag, 2)?;
                let (group, within) = (over(lane.clone(), 4), modulo(lane, 4));
                let from = sum(times(within, 8), over(group.clone(), 2));
                let shift = self.lane_value(&times(modulo(group, 2), 16), ctx)?;
                let lanes = [
                    self.lane_value(&from, ctx)?,
                    self.lane_value(&sum(from, Expression::int(4)), ctx)?,
                ];
                for (pair, into) in floats.chunks(2).zip(packed) {
                    let pair = self.packed_pair(pair[0], pair[1], Ty::F16x2, None);
                    let mut halves = Vec::with_capacity(2);
                    for lane in &lanes {
                        let theirs = self.taken_from(pair, lane);
                        let mine =
                            self.word(Opcode::Shr, Ty::U32, vec![theirs.into(), shift.clone()]);
                        halves.push(self.word(
                            Opcode::And,
                            Ty::B32,
                            vec![mine.into(), Operand::Int(0xffff)],
                        ));
                    }
                    let high = self.word(
                        Opcode::Shl,
                        Ty::B32,
                        vec![halves[1].into(), Operand::Int(16)],
                    );
                    let join = Inst::new(
                        Opcode::Or,
                        Ty::B32,
                        vec![into],
                        vec![halves[0].into(), high.into()],
                    );
                    self.b.emit(join);
                }
            }
            ("m8n8k4", "a", [source]) if ab == "f16" && (0..4).contains(&node.part) => {
                let part = node.part;
                let floats = self.tensor_fragment(*source, 8, VTy::F32)?;
                let packed = self.repacked(node.frag, 2)?;
                // The group of four lanes that holds this part's columns for the lane's rows, and in it the
                // two lanes of the lane's own row.
                let group = masked(apply(Operation::RightShift, masked(lane.clone(), 15), 2), 2);
                let group = either(group, Expression::int(part >> 1));
                let from = either(
                    masked(lane.clone(), 16),
                    apply(Operation::LeftShift, group, 2),
                );
                let from = either(from, masked(lane.clone(), 1));
                let lanes = [
                    self.lane_value(&from, ctx)?,
                    self.lane_value(&sum(from, Expression::int(2)), ctx)?,
                ];
                let upper =
                    Expression::binary(Operation::NotEqual, masked(lane, 2), Expression::int(0));
                let upper = self.pred(&upper, ctx)?;
                let column = ((part & 1) << 2) as usize;
                let low = self.joined_pair(floats[column], floats[column + 1]);
                let high = self.joined_pair(floats[column + 2], floats[column + 3]);
                for (lane, into) in lanes.iter().zip(packed) {
                    let (low, high) = (self.taken_from(low, lane), self.taken_from(high, lane));
                    let operands = vec![high.into(), low.into(), upper.into()];
                    self.b
                        .emit(Inst::new(Opcode::Selp, Ty::B32, vec![into], operands));
                }
            }
            ("m8n8k4", "b", [source]) if ab == "f16" && (0..4).contains(&node.part) => {
                let part = node.part;
                let floats = self.tensor_fragment(*source, 8, VTy::F32)?;
                let packed = self.repacked(node.frag, 2)?;
                let group = masked(apply(Operation::RightShift, masked(lane.clone(), 15), 2), 1);
                let group = sum(group, Expression::int((part >> 1) * 2));
                let from = either(
                    Expression::int((part & 1) << 4),
                    apply(Operation::LeftShift, group, 2),
                );
                let from = either(from, masked(lane.clone(), 2));
                let lanes = [
                    self.lane_value(&from, ctx)?,
                    self.lane_value(&sum(from, Expression::int(1)), ctx)?,
                ];
                let upper = Expression::binary(
                    Operation::NotEqual,
                    masked(lane.clone(), 16),
                    Expression::int(0),
                );
                let upper = self.pred(&upper, ctx)?;
                let shift = self.lane_value(&times(masked(lane, 1), 16), ctx)?;
                for (p, into) in packed.into_iter().enumerate() {
                    let low = self.joined_pair(floats[2 * p], floats[2 * p + 1]);
                    let high = self.joined_pair(floats[2 * p + 4], floats[2 * p + 5]);
                    let mut columns = Vec::with_capacity(2);
                    for lane in &lanes {
                        let (low, high) = (self.taken_from(low, lane), self.taken_from(high, lane));
                        let operands = vec![high.into(), low.into(), upper.into()];
                        let theirs = self.word(Opcode::Selp, Ty::B32, operands);
                        columns.push(self.word(
                            Opcode::Shr,
                            Ty::U32,
                            vec![theirs.into(), shift.clone()],
                        ));
                    }
                    let first = self.word(
                        Opcode::And,
                        Ty::B32,
                        vec![columns[0].into(), Operand::Int(0xffff)],
                    );
                    let second = self.word(
                        Opcode::Shl,
                        Ty::B32,
                        vec![columns[1].into(), Operand::Int(16)],
                    );
                    let join = Inst::new(
                        Opcode::Or,
                        Ty::B32,
                        vec![into],
                        vec![first.into(), second.into()],
                    );
                    self.b.emit(join);
                }
            }
            _ => {
                let sources = node.srcs.len();
                return Err(self.no_rule(format_args!(
                    "a repack of {sources} accumulators into {role} at {ab} in the layout {layout}"
                )));
            }
        }
        Ok(())
    }

    /// One lane's register of block scales: the four scales of its row of A, or of its column of B, one a byte, as
    /// they stand in the slab. Only the lanes the instruction's selector names supply one; any other holds zero
    /// and reads nothing.
    pub(super) fn block_scale_load(
        &mut self,
        node: &BlockScaleLoad,
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let scales = match self.find(node.frag) {
            Some((_, val)) if val.ty == VTy::U32 => val.op.reg(),
            _ => None,
        };
        let Some(scales) = scales.filter(|reg| self.b.is_state(*reg)) else {
            let name = node.frag;
            return Err(self.refuse(format_args!("{name} is no register of block scales")));
        };
        let buffer = self.buffer(node.src_buffer)?;
        if buffer.elem.bytes() != 1 {
            let (name, dtype) = (node.src_buffer, buffer.elem.dtype);
            return Err(self.refuse(format_args!(
                "block scales are bytes, and {name} holds {dtype}"
            )));
        }
        let lane = lane();
        let (group, within) = (over(lane.clone(), 4), modulo(lane, 4));
        let (supplies, row) = match &*node.role {
            "a" => {
                let nearer = match SCALE_SELECTOR_A {
                    0 => within,
                    selector => sum(within, Expression::int(-2 * selector)),
                };
                let in_range = Expression::binary(
                    Operation::Equal,
                    over(nearer.clone(), 2),
                    Expression::int(0),
                );
                (in_range, sum(group, times(nearer, 8)))
            }
            "b" => {
                let selected =
                    Expression::binary(Operation::Equal, within, Expression::int(SCALE_SELECTOR_B));
                (selected, group)
            }
            role => return Err(self.refuse(format_args!("block scales of role {role}"))),
        };
        let empty = Inst::new(Opcode::Mov, Ty::B32, vec![scales], vec![Operand::Int(0)]);
        self.b.emit(empty);
        let supplies = self.pred(&supplies, ctx)?;
        let open = self.b.open_if(supplies);
        let flat = self.flat_index(node.src_buffer, &node.src_index)?;
        let first = self.address(node.src_buffer, &[sum(flat, times(row, node.ldm))], ctx)?;
        let Operand::Addr(base, offset) = first else {
            return Err(self.refuse("block scales have an address"));
        };
        // The four bytes, the first the lowest, as one register.
        let mut word = None;
        for byte in 0..4 {
            let scale = self.b.reg(Bank::B32);
            let address = Operand::Addr(base, offset + byte);
            let load = Inst::new(buffer.load(), Ty::U8, vec![scale], vec![address]);
            self.b.emit(load);
            let Some(below) = word else {
                word = Some(scale);
                continue;
            };
            let shift = vec![scale.into(), Operand::Int(8 * byte)];
            let placed = self.word(Opcode::Shl, Ty::B32, shift);
            let joined = vec![below.into(), placed.into()];
            if byte < 3 {
                word = Some(self.word(Opcode::Or, Ty::B32, joined));
            } else {
                let take = Inst::new(Opcode::Or, Ty::B32, vec![scales], joined);
                self.b.emit(take);
            }
        }
        self.b.close_if(open)
    }
}
