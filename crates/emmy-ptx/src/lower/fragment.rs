//! The fragment statements: a fragment gathered from memory element by element (`FragmentLoad`), stored to it
//! (`RegStore`), computed on element by element (`FragmentApply`) and folded along its rows (`FragmentRowReduce`).
//!
//! Each is expanded from the lane-to-element table of `emmy-ir` (`fragment::fragment_table`): per register and
//! element the table gives the element's row and column in the tile as expressions over the lane, and the rule adds
//! them to the statement's own index. No rule holds a lane map of its own. The lane's coordinates are integer
//! expressions over the thread index like any other, so the value table keeps each once.

use std::collections::HashMap;

use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expr, Expression, SimplifyCtx};
use emmy_ir::fragment::{
    FragmentTable, LANE, fragment_table, paired_accumulator_table, paired_operand_table,
};
use emmy_ir::statement::{FragmentApply, FragmentLoad, FragmentRowReduce, RegStore, Statement};
use emmy_ir::{Name, Result};

use super::int::{constant_difference, divided};
use super::memory::{Buffer, Elem};
use super::tensor::{lane, swizzle_exchange, swizzled};
use super::value::SM_ADA;
use super::{Lowerer, Needs, SM_LAST, VTy, Val};
use crate::inst::{Bank, Base, Inst, Opcode, Operand, Reg, Round, Shuffle, Space, Ty};

/// The names a `FragmentApply`'s coordinate and gathered operands read the element's own row and column by, and
/// the names a `RegStore`'s epilogue reads the element's offset in its fragment by.
const FRAGMENT_ROW: &str = "__frow";
const FRAGMENT_COLUMN: &str = "__fcol";
const ELEMENT_ROW: &str = "__M__";
const ELEMENT_COLUMN: &str = "__N__";

pub(super) const F8_CONVERSION: Needs = Needs {
    what: "a conversion of 8-bit floats (`cvt` with `e4m3x2`)",
    sm: SM_ADA..=SM_LAST,
    arch_specific: false,
    version: (8, 4),
};

/// What a lane holds of one fragment: per register and element, the element's row and column offset in the tile
/// as expressions over the thread index.
pub(super) struct Lanes {
    registers: Vec<Vec<(Expression, Expression)>>,
    reduce_xors: Vec<i64>,
}

/// The lane coordinates of `table` over the thread index, each by its name.
fn coordinates(table: &FragmentTable) -> HashMap<Name, Expression> {
    let mut over_thread = HashMap::from([(Name::new(LANE), lane())]);
    for (name, value) in &table.coordinates {
        let value = value.substitute(&over_thread);
        over_thread.insert(*name, value);
    }
    over_thread
}

/// `expression` with each shift right by a constant written as the floor division it is, and each mask of low bits
/// as the floor remainder: the same value, in the operations the range analysis bounds. The lane tables spell a
/// lane's group as a shift.
fn arithmetic(expression: &Expression) -> Expression {
    let Some((op, [left, right])) = expression.operation() else {
        return expression.clone();
    };
    let (left, right) = (arithmetic(left), arithmetic(right));
    match (op, right.int_literal()) {
        (Operation::RightShift, Some(shift)) if (0..31).contains(&shift) => {
            binary(Operation::FloorDivide, left, Expression::int(1 << shift))
        }
        (Operation::BitwiseAnd, Some(mask)) if mask > 0 && mask & (mask + 1) == 0 => {
            binary(Operation::Remainder, left, Expression::int(mask + 1))
        }
        _ => binary(op, left, right),
    }
}

/// Whether a store's guard holds for every thread or for none, where the ranges of its two sides tell: a bound
/// below every value, or above every one. `None` where they overlap.
fn decided(guard: &Expression, ctx: &SimplifyCtx) -> Option<bool> {
    let (op, [left, right]) = guard.operation()? else {
        return None;
    };
    match op {
        Operation::LogicalAnd => match (decided(left, ctx), decided(right, ctx)) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        Operation::Less => {
            let (left, right) = (arithmetic(left).range(ctx)?, arithmetic(right).range(ctx)?);
            if left.hi < right.lo {
                Some(true)
            } else if left.lo >= right.hi {
                Some(false)
            } else {
                None
            }
        }
        _ => None,
    }
}

impl Lanes {
    fn of(table: &FragmentTable) -> Lanes {
        let over_thread = coordinates(table);
        let place = |(row, column): &(Expression, Expression)| {
            (
                row.substitute(&over_thread),
                column.substitute(&over_thread),
            )
        };
        Lanes {
            registers: table
                .registers
                .iter()
                .map(|register| register.iter().map(place).collect())
                .collect(),
            reduce_xors: table.reduce_xors.clone(),
        }
    }

    /// Per register of an accumulator of floats, which of the lane's rows its element stands in, the rows
    /// numbered in the order the registers reach them.
    fn rows(&self) -> Vec<usize> {
        let mut rows: Vec<&Expression> = Vec::new();
        self.registers
            .iter()
            .map(|register| {
                let row = &register[0].0;
                rows.iter()
                    .position(|seen| *seen == row)
                    .unwrap_or_else(|| {
                        rows.push(row);
                        rows.len() - 1
                    })
            })
            .collect()
    }
}

fn binary(op: Operation, left: Expression, right: Expression) -> Expression {
    Expression::binary(op, left, right)
}

fn plus(left: Expression, right: Expression) -> Expression {
    binary(Operation::Add, left, right)
}

fn times(left: Expression, right: Expression) -> Expression {
    binary(Operation::Multiply, left, right)
}

fn var(name: Name) -> Expression {
    Expression::new(Expr::Var(name))
}

/// The elements left of an axis from a tile's base: `bound - base`.
fn left((base, bound): &(Expression, Expression)) -> Expression {
    binary(Operation::Subtract, bound.clone(), base.clone())
}

pub(super) fn dtype_of(ty: VTy) -> DType {
    match ty {
        VTy::F32 => DType::F32,
        VTy::F64 => DType::F64,
        VTy::F16 => DType::F16,
        VTy::Bf16 => DType::BF16,
        VTy::I32 => DType::I32,
        VTy::U32 => DType::U32,
        VTy::I64 => DType::I64,
        VTy::U64 => DType::U64,
        VTy::I16 => DType::I16,
        VTy::U16 => DType::U16,
        VTy::Bool => DType::Bool,
        VTy::F8E4M3 => DType::F8E4M3,
        VTy::F8E5M2 => DType::F8E5M2,
        VTy::F4x2 => DType::F4E2M1x2,
    }
}

impl Lowerer {
    /// Compute, where the grid's body starts, the lane coordinates every fragment statement of `body` reads: the
    /// grid's own scope then keeps each for every statement below it, so a kernel computes them once.
    pub(super) fn lane_coordinates(&mut self, body: &Body, ctx: &SimplifyCtx) -> Result<()> {
        let accumulator = |layout: &Name| fragment_table(layout, "c", DType::F32).ok();
        for stmt in body.iter_stmts() {
            let tables = match &*stmt {
                Statement::FragmentLoad(node) if node.pair_frag.is_some() => {
                    let b_row = &*node.slab_layout == "V70B";
                    [0, 1]
                        .iter()
                        .filter_map(|cell| paired_operand_table(&node.role, b_row, *cell).ok())
                        .collect()
                }
                Statement::FragmentLoad(node) => node
                    .dtype
                    .and_then(|dtype| fragment_table(&node.fragment_layout, &node.role, dtype).ok())
                    .into_iter()
                    .collect(),
                Statement::RegStore(node) => self.store_table(node).ok().into_iter().collect(),
                Statement::FragmentApply(node) => accumulator(&node.layout).into_iter().collect(),
                Statement::FragmentRowReduce(node) => {
                    accumulator(&node.layout).into_iter().collect()
                }
                _ => Vec::new(),
            };
            for table in tables {
                let over_thread = coordinates(&table);
                for (name, _) in &table.coordinates {
                    self.int(&over_thread[name], ctx)?;
                }
            }
        }
        Ok(())
    }

    /// Whether `address` is proven a multiple of `bytes`.
    pub(super) fn aligned(&self, address: &Operand, bytes: i64) -> bool {
        match address {
            Operand::Addr(Base::Reg(base), offset) => {
                i64::from(self.b.alignment(*base)) % bytes == 0 && offset % bytes == 0
            }
            _ => false,
        }
    }

    /// The low 16 bits of a 32-bit register.
    fn low_half(&mut self, word: Reg) -> Reg {
        let narrow = Inst::new(Opcode::Cvt(Ty::U32), Ty::U16, Vec::new(), vec![word.into()]);
        self.b.value(narrow, Bank::B16)
    }

    fn word_of(&mut self, opcode: Opcode, ty: Ty, left: Reg, right: i64) -> Reg {
        let operands = vec![left.into(), Operand::Int(right)];
        self.b
            .value(Inst::new(opcode, ty, Vec::new(), operands), Bank::B32)
    }

    // --- conversions of one element ----------------------------------------------------------------------

    /// One 8-bit float, a byte in a 32-bit register, as the half of the same value. An e5m2 is the top byte of
    /// that half. An e4m3 goes through the hardware's conversion where the card has it; before it, its bits
    /// shifted seven places are a half 256 times smaller, a subnormal included, and its one NaN is picked out.
    pub(super) fn f8_as_f16(&mut self, byte: Reg, from: DType) -> Result<Reg> {
        if from == DType::F8E5M2 {
            let bits = self.word_of(Opcode::Shl, Ty::B32, byte, 8);
            return Ok(self.low_half(bits));
        }
        if self.target.sm >= SM_ADA {
            self.requires(&F8_CONVERSION)?;
            let pair = self.low_half(byte);
            let convert = Inst::new(
                Opcode::Cvt(Ty::E4m3x2),
                Ty::F16x2,
                Vec::new(),
                vec![pair.into()],
            );
            let halves = self.b.value(convert.rounded(Round::Rn), Bank::B32);
            return Ok(self.low_half(halves));
        }
        let magnitude = self.word_of(Opcode::And, Ty::B32, byte, 0x7f);
        let placed = self.word_of(Opcode::Shl, Ty::B32, magnitude, 7);
        let sign = self.word_of(Opcode::And, Ty::B32, byte, 0x80);
        let sign = self.word_of(Opcode::Shl, Ty::B32, sign, 8);
        let join = Inst::new(
            Opcode::Or,
            Ty::B32,
            Vec::new(),
            vec![placed.into(), sign.into()],
        );
        let bits = self.b.value(join, Bank::B32);
        let small = self.low_half(bits);
        let constant = |this: &mut Lowerer, bits: i64| {
            let held = Inst::new(Opcode::Mov, Ty::B16, Vec::new(), vec![Operand::Int(bits)]);
            this.b.value(held, Bank::B16)
        };
        // 2^8 and the NaN, as halves.
        let (scale, nan) = (constant(self, 0x5c00), constant(self, 0x7fff));
        let scaled = Inst::new(
            Opcode::Mul,
            Ty::F16,
            Vec::new(),
            vec![small.into(), scale.into()],
        );
        let scaled = self.b.value(scaled.rounded(Round::Rn), Bank::B16);
        let is_nan = Inst::new(
            Opcode::Setp(crate::inst::Cmp::Eq),
            Ty::U32,
            Vec::new(),
            vec![magnitude.into(), Operand::Int(0x7f)],
        );
        let is_nan = self.b.value(is_nan, Bank::Pred);
        let operands = vec![nan.into(), scaled.into(), is_nan.into()];
        let pick = Inst::new(Opcode::Selp, Ty::B16, Vec::new(), operands);
        Ok(self.b.value(pick, Bank::B16))
    }

    /// One element of `from`, as a load gives it, as an f32.
    pub(super) fn widened(&mut self, value: Reg, from: DType) -> Result<Reg> {
        let (half, ty) = match from {
            DType::F32 => return Ok(value),
            DType::F16 => (value, Ty::F16),
            DType::BF16 => (value, Ty::Bf16),
            DType::F8E4M3 | DType::F8E5M2 => (self.f8_as_f16(value, from)?, Ty::F16),
            other => return Err(self.no_rule(format_args!("a conversion of {other} to f32"))),
        };
        let widen = Inst::new(Opcode::Cvt(ty), Ty::F32, Vec::new(), vec![half.into()]);
        Ok(self.b.value(widen, Bank::F32))
    }

    /// An f32 rounded to nearest into one element of `to`.
    pub(super) fn narrowed(&mut self, value: Reg, to: DType) -> Result<Reg> {
        let ty = match to {
            DType::F32 => return Ok(value),
            DType::F16 => Ty::F16,
            DType::BF16 => Ty::Bf16,
            other => return Err(self.no_rule(format_args!("a conversion of f32 to {other}"))),
        };
        let narrow = Inst::new(Opcode::Cvt(Ty::F32), ty, Vec::new(), vec![value.into()]);
        Ok(self.b.value(narrow.rounded(Round::Rn), Bank::B16))
    }

    /// The value of a 4-bit e2m1 code, the low four bits of `code`, as an f32. The code's three magnitude bits
    /// under an f32's two low exponent bits and its top mantissa bit spell the value times 2^-126, the format's
    /// subnormal 0.5 as a subnormal, and one exact multiply gives the value back. The multiply keeps subnormals
    /// whatever the kernel's flush decision says.
    pub(super) fn e2m1_value(&mut self, code: Reg) -> Reg {
        let sign = self.word_of(Opcode::And, Ty::B32, code, 8);
        let sign = self.word_of(Opcode::Shl, Ty::B32, sign, 28);
        let magnitude = self.word_of(Opcode::And, Ty::B32, code, 7);
        let magnitude = self.word_of(Opcode::Shl, Ty::B32, magnitude, 22);
        let join = Inst::new(
            Opcode::Or,
            Ty::B32,
            Vec::new(),
            vec![sign.into(), magnitude.into()],
        );
        let bits = self.b.value(join, Bank::B32);
        let small = Inst::new(Opcode::Mov, Ty::B32, Vec::new(), vec![bits.into()]);
        let small = self.b.value(small, Bank::F32);
        let operands = vec![small.into(), Operand::f32(2f32.powi(126))];
        let value = Inst::new(Opcode::Mul, Ty::F32, Vec::new(), operands);
        self.b.value(value.rounded(Round::Rn), Bank::F32)
    }

    // --- the gather --------------------------------------------------------------------------------------

    /// One element of `buffer` at `address`, as it stands in memory. Under `guard` it is read only where the
    /// guard holds, and is zero elsewhere.
    fn gathered(&mut self, buffer: &Buffer, address: Operand, guard: Option<Reg>) -> Reg {
        let ty = buffer.elem.mem;
        let load = Inst::new(buffer.load(), ty, Vec::new(), vec![address]);
        let Some(guard) = guard else {
            let reg = self.b.reg(ty.bank());
            self.b.emit(Inst {
                results: vec![reg],
                ..load
            });
            return reg;
        };
        let reg = self.b.state(ty.bank());
        let (held, zero) = match ty {
            Ty::F32 => (Ty::F32, Operand::f32(0.0)),
            Ty::B16 => (Ty::B16, Operand::Int(0)),
            _ => (Ty::B32, Operand::Int(0)),
        };
        self.b
            .emit(Inst::new(Opcode::Mov, held, vec![reg], vec![zero]));
        self.b.emit(
            Inst {
                results: vec![reg],
                ..load
            }
            .guarded(guard, false),
        );
        reg
    }

    /// The bits one element of `source` takes in a fragment of `dtype`: 16 bits in a 16-bit register, or a byte.
    fn element_bits(&mut self, value: Reg, source: DType, dtype: DType) -> Result<Reg> {
        use DType::{BF16, F8E4M3, F8E5M2, F16, F32};
        match (source, dtype) {
            _ if source == dtype => Ok(value),
            (F8E4M3 | F8E5M2, F16) => self.f8_as_f16(value, source),
            (F32 | F16 | BF16 | F8E4M3 | F8E5M2, F16 | BF16) => {
                let wide = self.widened(value, source)?;
                self.narrowed(wide, dtype)
            }
            _ => Err(self.no_rule(format_args!("a fragment of {dtype} gathered from {source}"))),
        }
    }

    /// One register of a fragment from the elements at `elements`, each an index into `name` and the guard it is
    /// read under. Elements that stand in memory as the register holds them, in a row and proven aligned, are
    /// one 32-bit load; two 8-bit floats in a row are one 16-bit load and one packed conversion where the card
    /// has it. Any other register is a load an element, each converted, then packed.
    fn gather_register(
        &mut self,
        reg: Reg,
        name: Name,
        elements: &[(Expression, Option<Expression>)],
        (source, dtype): (DType, DType),
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let buffer = self.buffer(name)?;
        let (first, shared) = &elements[0];
        let in_a_row = elements.iter().enumerate().all(|(at, (index, guard))| {
            guard == shared && constant_difference(index, first) == Some(at as i64)
        });
        if in_a_row {
            let address = self.address(name, std::slice::from_ref(first), ctx)?;
            let bytes = buffer.elem.bytes() * elements.len() as i64;
            if source == dtype && bytes == 4 && self.aligned(&address, 4) {
                let load = Inst::new(buffer.load(), Ty::B32, vec![reg], vec![address]);
                match shared {
                    Some(guard) => {
                        let guard = self.pred(guard, ctx)?;
                        self.zero(&[reg], VTy::U32);
                        self.b.emit(load.guarded(guard, false));
                    }
                    None => self.b.emit(load),
                }
                return Ok(());
            }
            let packed = source == DType::F8E4M3 && dtype == DType::F16 && bytes == 2;
            if shared.is_none() && packed && self.target.sm >= SM_ADA && self.aligned(&address, 2) {
                self.requires(&F8_CONVERSION)?;
                let pair = self.b.reg(Bank::B16);
                let load = Inst::new(buffer.load(), Ty::B16, vec![pair], vec![address]);
                self.b.emit(load);
                let convert = Inst::new(
                    Opcode::Cvt(Ty::E4m3x2),
                    Ty::F16x2,
                    vec![reg],
                    vec![pair.into()],
                );
                self.b.emit(convert.rounded(Round::Rn));
                return Ok(());
            }
        }
        let mut parts = Vec::with_capacity(elements.len());
        for (index, guard) in elements {
            let guard = match guard {
                Some(guard) => Some(self.pred(guard, ctx)?),
                None => None,
            };
            let address = self.address(name, std::slice::from_ref(index), ctx)?;
            let value = self.gathered(&buffer, address, guard);
            parts.push(self.element_bits(value, source, dtype)?);
        }
        self.pack_into(reg, &parts);
        Ok(())
    }

    /// Define `reg` as `parts`, the lowest first: two halves by one move, bytes by shifts and sums.
    fn pack_into(&mut self, reg: Reg, parts: &[Reg]) {
        let packed = match parts {
            [low, high] => Operand::Tuple(vec![*low, *high]),
            parts => self.pack(parts).into(),
        };
        self.b
            .emit(Inst::new(Opcode::Mov, Ty::B32, vec![reg], vec![packed]));
    }

    /// One operand fragment gathered element by element, from global memory or from a shared slab that is no
    /// `ldmatrix`'s. An element at the table's `(row, column)` stands that many strides and elements past the
    /// tile's base, by the role and the source's orientation. Under a clamp the lanes past the bound read the last
    /// element in range, and a fragment wholly past it reads nothing and is zero; under a zero-fill of K an
    /// element past the bound is zero and is not read.
    pub(super) fn fragment_gather(&mut self, node: &FragmentLoad, ctx: &SimplifyCtx) -> Result<()> {
        let (Some(source), Some(dtype)) = (node.src_dtype, node.dtype) else {
            return Err(self.refuse(
                "a fragment load carries the element types of its source and of its fragment",
            ));
        };
        if &*node.slab_layout != "NONE" || node.pair_frag.is_some() {
            return self.paired_gather(node, (source, dtype), ctx);
        }
        let a = match &*node.role {
            "a" => true,
            "b" => false,
            role => return Err(self.refuse(format_args!("a fragment load of role {role}"))),
        };
        let lanes = Lanes::of(&fragment_table(&node.fragment_layout, &node.role, dtype)?);
        let regs = self.tensor_fragment(node.frag, lanes.registers.len(), VTy::U32)?;
        let name = node.src_buffer;
        let buffer = self.buffer(name)?;
        if (buffer.space == Space::Shared) != node.staged || buffer.elem.dtype != source {
            let held = buffer.elem.dtype;
            return Err(self.refuse(format_args!(
                "{name} holds {held} and is not the buffer of {source} the load names"
            )));
        }
        let base = self.flat_index(name, &node.src_index)?;
        let stride = match node.ldm {
            Some(ldm) => Expression::int(ldm),
            None => self.stride(name, None)?,
        };
        if node.scale_buffer.is_some() {
            return self.scaled_gather(node, &lanes, &regs, (source, dtype), (base, stride), ctx);
        }
        let (axis_left, k_left) = (
            node.clamp.as_ref().map(left),
            node.k_zero.as_ref().map(left),
        );
        // A fragment wholly past the clamp guards each of its loads, as a zero-fill does, and is no branch: a
        // branch closes a block at each fragment's packs, which wait on its loads before the next fragment's issue.
        let some = axis_left
            .as_ref()
            .map(|left| binary(Operation::Greater, left.clone(), Expression::int(0)));
        for (reg, register) in regs.iter().zip(&lanes.registers) {
            let mut elements = Vec::with_capacity(register.len());
            for (row, column) in register {
                // The operand's own axis, which a clamp bounds, and K, which a zero-fill does.
                let (axis, k) = if a { (row, column) } else { (column, row) };
                let axis = match &axis_left {
                    Some(left) => {
                        let last = binary(Operation::Subtract, left.clone(), Expression::int(1));
                        let last = binary(Operation::Maximum, last, Expression::int(0));
                        binary(Operation::Minimum, axis.clone(), last)
                    }
                    None => axis.clone(),
                };
                let (major, minor) = if a || node.b_trans {
                    (axis, k.clone())
                } else {
                    (k.clone(), axis)
                };
                let index = plus(plus(base.clone(), times(major, stride.clone())), minor);
                let k_guard = k_left
                    .as_ref()
                    .map(|left| binary(Operation::Less, k.clone(), left.clone()));
                let guard = match (k_guard, &some) {
                    (Some(k), Some(some)) => Some(binary(Operation::LogicalAnd, k, some.clone())),
                    (k, some) => k.or_else(|| some.clone()),
                };
                elements.push((index, guard));
            }
            self.gather_register(*reg, name, &elements, (source, dtype), ctx)?;
        }
        Ok(())
    }

    /// The two fragments of one cell pair of Volta's paired tile, from a slab stored under a paired layout: A or a
    /// transposed B from the crosswise one, a B by rows from the congruous one. Each element is read at the place
    /// the layout gives its logical row and column, by the paired rows of the lane table, `frag` the first cell and
    /// `pair_frag` the second.
    fn paired_gather(
        &mut self,
        node: &FragmentLoad,
        types: (DType, DType),
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let layout = &*node.slab_layout;
        let (Some(pair), true) = (node.pair_frag, node.staged) else {
            return Err(self.refuse("a paired layout is read by a staged load of two fragments"));
        };
        let b_row = match (layout, &*node.role) {
            ("V70A", "a") => false,
            ("V70A", "b") if node.b_trans => false,
            ("V70B", "b") if !node.b_trans => true,
            (layout, role) => {
                return Err(self.refuse(format_args!(
                    "a fragment of role {role} from a slab laid out as {layout}"
                )));
            }
        };
        let [row, column] = &node.src_index[..] else {
            return Err(self.refuse("a paired slab is indexed by row and column"));
        };
        let name = node.src_buffer;
        for (frag, cell) in [(node.frag, 0), (pair, 1)] {
            let lanes = Lanes::of(&paired_operand_table(&node.role, b_row, cell)?);
            let regs = self.tensor_fragment(frag, lanes.registers.len(), VTy::U32)?;
            for (reg, register) in regs.iter().zip(&lanes.registers) {
                let mut elements = Vec::with_capacity(register.len());
                for (r, c) in register {
                    // The table's B is (k, n); a transposed B's slab holds n along its rows.
                    let (r, c) = if node.role == "b" && node.b_trans {
                        (c, r)
                    } else {
                        (r, c)
                    };
                    let at = [
                        plus(row.clone(), r.clone()),
                        plus(column.clone(), c.clone()),
                    ];
                    elements.push((self.slab_index(name, &at, layout)?, None));
                }
                self.gather_register(*reg, name, &elements, types, ctx)?;
            }
        }
        Ok(())
    }

    /// A transposed B fragment of 16-bit elements from a slab of codes beside a slab of block scales: each value
    /// is decoded, multiplied by the scale of its K block in f32 and rounded once into the fragment. A byte of
    /// packed 4-bit codes holds two K positions, the low four bits the first; an 8-bit float is one.
    fn scaled_gather(
        &mut self,
        node: &FragmentLoad,
        lanes: &Lanes,
        regs: &[Reg],
        (source, dtype): (DType, DType),
        (base, stride): (Expression, Expression),
        ctx: &SimplifyCtx,
    ) -> Result<()> {
        let fits = &*node.role == "b" && node.b_trans && node.staged;
        let packed = match source {
            DType::F4E2M1x2 => true,
            DType::F8E4M3 | DType::F8E5M2 => false,
            _ => return Err(self.no_rule(format_args!("a block-scaled gather of {source}"))),
        };
        let Some(scales) = node.scale_buffer.filter(|_| fits) else {
            return Err(self.no_rule("a block-scaled gather that is no transposed B from a slab"));
        };
        let name = node.src_buffer;
        let (buffer, scale_buffer) = (self.buffer(name)?, self.buffer(scales)?);
        let scale_base = self.flat_index(scales, &node.scale_index)?;
        for (reg, register) in regs.iter().zip(&lanes.registers) {
            let [(k, n), (next, _)] = &register[..] else {
                return Err(self.refuse("a block-scaled gather fills registers of two elements"));
            };
            if constant_difference(next, k) != Some(1) {
                return Err(
                    self.refuse("the two elements of a register are next to each other in K")
                );
            }
            let block = binary(Operation::FloorDivide, k.clone(), Expression::int(16));
            let block = block.simplify(ctx).unwrap_or(block);
            let scale_row = times(n.clone(), Expression::int(node.scale_ldm));
            let scale_index = plus(plus(scale_base.clone(), scale_row), block);
            let address = self.address(scales, &[scale_index], ctx)?;
            let scale = self.gathered(&scale_buffer, address, None);
            let scale = self.widened(scale, scale_buffer.elem.dtype)?;
            let row = plus(base.clone(), times(n.clone(), stride.clone()));
            let values = if packed {
                let address = self.address(name, &[plus(row, divided(k, 2))], ctx)?;
                let byte = self.gathered(&buffer, address, None);
                let low = self.word_of(Opcode::And, Ty::B32, byte, 0xf);
                let high = self.word_of(Opcode::Shr, Ty::U32, byte, 4);
                [self.e2m1_value(low), self.e2m1_value(high)]
            } else {
                let mut values = [scale; 2];
                for (value, k) in values.iter_mut().zip([k, next]) {
                    let address = self.address(name, &[plus(row.clone(), k.clone())], ctx)?;
                    let byte = self.gathered(&buffer, address, None);
                    *value = self.widened(byte, source)?;
                }
                values
            };
            let mut parts = Vec::with_capacity(2);
            for value in values {
                let operands = vec![value.into(), scale.into()];
                let product = Inst::new(Opcode::Mul, Ty::F32, Vec::new(), operands);
                let product = self.b.value(product.rounded(Round::Rn), Bank::F32);
                parts.push(self.narrowed(product, dtype)?);
            }
            self.pack_into(*reg, &parts);
        }
        Ok(())
    }

    // --- the store ---------------------------------------------------------------------------------------

    /// The elements between two neighbours along dimension `dim` of `name`: the product of the extents after it.
    /// With no dimension, the extent of the last one, a row's length.
    fn stride(&self, name: Name, dim: Option<i64>) -> Result<Expression> {
        let Some(extents) = self
            .extents
            .get(&name)
            .filter(|extents| !extents.is_empty())
        else {
            return Err(self.refuse(format_args!(
                "a stride of {name} is its shape's, and the interface gives none"
            )));
        };
        let after = match dim {
            None => extents.len() - 1,
            Some(dim) => usize::try_from(dim + 1)
                .unwrap_or(usize::MAX)
                .min(extents.len()),
        };
        let mut stride = Expression::int(1);
        let mut known = 1;
        for extent in &extents[after..] {
            match extent.int_literal() {
                Some(extent) => known *= extent,
                None => stride = times(stride, extent.clone()),
            }
        }
        Ok(match (known, stride.int_literal()) {
            (known, Some(_)) => Expression::int(known),
            (1, None) => stride,
            (known, None) => times(Expression::int(known), stride),
        })
    }

    fn store_table(&self, node: &RegStore) -> Result<FragmentTable> {
        if node.volta_interleaved {
            let cell = (node.fragment_index.0, node.fragment_index.1);
            return Ok(paired_accumulator_table(cell));
        }
        fragment_table(&node.fragment_layout, "c", DType::F32)
    }

    /// An accumulator of floats stored to memory, each lane its own elements. The cells of a run are each stored
    /// as their own statement says; gathering four cells into one 16-byte row a lane by shuffles, as the run
    /// allows, is not done.
    pub(super) fn reg_store(&mut self, node: &RegStore, ctx: &SimplifyCtx) -> Result<()> {
        self.store_cell(node, ctx)?;
        for cell in node.run.iter() {
            match &**cell {
                Statement::RegStore(cell) => self.store_cell(cell, ctx)?,
                _ => return Err(self.refuse("a run of stores holds stores")),
            }
        }
        Ok(())
    }

    /// One cell: per element the address from the statement's index and the table's row and column by the two
    /// strides, the value (the register, or what the epilogue makes of it at the element's own coordinates),
    /// converted to the buffer's type. An element past a guard's bound is not stored and its epilogue reads
    /// nothing; a guard the ranges decide is no branch, and an element it always turns away is not written at
    /// all. Two elements next to each other in memory, under one guard and proven aligned, are one store.
    fn store_cell(&mut self, node: &RegStore, ctx: &SimplifyCtx) -> Result<()> {
        let lanes = Lanes::of(&self.store_table(node)?);
        let count = lanes.registers.len();
        let mut frags = Vec::with_capacity(1 + node.extra_frags.len());
        for name in std::iter::once(&node.frag).chain(node.extra_frags.iter()) {
            frags.push(self.tensor_fragment(*name, count, VTy::F32)?);
        }
        let name = node.dst_buffer;
        let buffer = self.buffer(name)?;
        let elem = buffer.elem;
        if !buffer.writable {
            return Err(self.refuse(format_args!("{name} is an input")));
        }
        let half = matches!(elem.dtype, DType::F16 | DType::BF16);
        if node.atomic && elem.dtype != DType::F32 {
            let stored = elem.dtype;
            return Err(self.no_rule(format_args!("an atomic store of a fragment into {stored}")));
        }
        let Some(exchange) = swizzle_exchange(&node.swizzle) else {
            let mode = node.swizzle;
            return Err(self.no_rule(format_args!(
                "a fragment stored to a slab laid out as {mode}"
            )));
        };
        let base = self.flat_index(name, &node.dst_index)?;
        let row_stride = match node.ldm {
            0 => self.stride(name, node.row_dim)?,
            ldm => Expression::int(ldm),
        };
        let column_stride = match (node.ldn, node.col_dim) {
            (0, Some(dim)) => self.stride(name, Some(dim))?,
            (0, None) => Expression::int(1),
            (ldn, _) => Expression::int(ldn),
        };
        let within = |guard: &Option<(Expression, Expression)>, offset: &Expression| {
            guard.as_ref().map(|(base, bound)| {
                binary(
                    Operation::Less,
                    plus(base.clone(), offset.clone()),
                    bound.clone(),
                )
            })
        };
        let mut elements = Vec::with_capacity(count);
        for register in &lanes.registers {
            let (row, column) = &register[0];
            let index = plus(
                plus(base.clone(), times(row.clone(), row_stride.clone())),
                times(column.clone(), column_stride.clone()),
            );
            let index = match exchange {
                Some(exchange) => swizzled(index, exchange),
                None => index,
            };
            let guard = match (within(&node.m_guard, row), within(&node.n_guard, column)) {
                (Some(m), Some(n)) => Some(binary(Operation::LogicalAnd, m, n)),
                (m, n) => m.or(n),
            };
            // A guard the ranges decide is no branch: an element always past its bound is not stored at all.
            match guard.as_ref().and_then(|guard| decided(guard, ctx)) {
                Some(false) => elements.push((index, None, false)),
                Some(true) => elements.push((index, None, true)),
                None => elements.push((index, guard, true)),
            }
        }
        let pairs = !node.atomic && (half || elem.dtype == DType::F32);
        let mut at = 0;
        while at < count {
            let (index, guard, stored) = &elements[at];
            if !stored {
                at += 1;
                continue;
            }
            let pair = pairs
                && elements
                    .get(at + 1)
                    .is_some_and(|(next, next_guard, next_stored)| {
                        *next_stored
                            && next_guard == guard
                            && constant_difference(next, index) == Some(1)
                    });
            let group = if pair { 2 } else { 1 };
            // The addresses are computed ahead of the guard, where every store of the statement shares their parts.
            let mut addresses = Vec::with_capacity(group);
            for (index, _, _) in &elements[at..at + group] {
                addresses.push(self.address(name, std::slice::from_ref(index), ctx)?);
            }
            let open = match guard {
                Some(guard) => {
                    let pred = self.pred(guard, ctx)?;
                    Some(self.b.open_if(pred))
                }
                None => None,
            };
            let mut values = Vec::with_capacity(group);
            for element in at..at + group {
                let val =
                    self.stored_value(node, &frags, element, &lanes.registers[element][0], ctx)?;
                values.push(self.stored_element(val, elem)?);
            }
            let store = Opcode::St(buffer.space);
            if pair && self.aligned(&addresses[0], 2 * elem.bytes()) {
                let operands = vec![addresses[0].clone(), Operand::Tuple(values)];
                self.b
                    .emit(Inst::new(store, elem.mem, Vec::new(), operands));
            } else {
                for (address, value) in addresses.into_iter().zip(values) {
                    if node.atomic {
                        let held = self.b.reg(Bank::F32);
                        let add = Opcode::AtomAdd(buffer.space);
                        let operands = vec![address, value.into()];
                        self.b.emit(Inst::new(add, Ty::F32, vec![held], operands));
                    } else {
                        let operands = vec![address, value.into()];
                        self.b
                            .emit(Inst::new(store, elem.mem, Vec::new(), operands));
                    }
                }
            }
            if let Some(open) = open {
                self.b.close_if(open)?;
            }
            at += group;
        }
        Ok(())
    }

    /// What element `at` of a store's fragment is stored as: the register, or the result of the epilogue. The
    /// epilogue is a body of ordinary statements, run through their own rules in a scope of its own: its
    /// parameters are the element of each fragment, and the two reserved names the element's row and column
    /// offset in the fragment.
    fn stored_value(
        &mut self,
        node: &RegStore,
        frags: &[Vec<Reg>],
        at: usize,
        (row, column): &(Expression, Expression),
        ctx: &SimplifyCtx,
    ) -> Result<Val> {
        let (Some(epilogue), Some(result)) = (&node.epilogue, node.epilogue_result) else {
            return Ok(Val::reg(frags[0][at], VTy::F32));
        };
        if node.epilogue_params.len() != frags.len() {
            return Err(self.refuse("an epilogue has a parameter a fragment"));
        }
        let statement = self.statement.clone();
        self.names.push(HashMap::new());
        let done = (|| {
            for (param, frag) in node.epilogue_params.iter().zip(frags) {
                let val = self.named(Val::reg(frag[at], VTy::F32));
                self.bind(*param, val);
            }
            let mut ctx = ctx.clone();
            for (name, offset) in [(ELEMENT_ROW, row), (ELEMENT_COLUMN, column)] {
                let (name, val) = (Name::new(name), self.int(offset, &ctx)?);
                if let Some(range) = offset.range(&ctx) {
                    ctx.ranges.insert(name, range);
                }
                self.bind(name, val);
            }
            self.body(epilogue, &ctx)?;
            self.lookup(result)
        })();
        self.names.pop();
        self.statement = statement;
        done
    }

    /// `val` as one element of a store into a buffer of `elem`: a float rounded to nearest into a narrower one,
    /// an integer as it is.
    fn stored_element(&mut self, val: Val, elem: Elem) -> Result<Reg> {
        use DType::{BF16, F16, F32};
        if val.ty == elem.ty && val.ty.is_byte_float() {
            return Ok(self.register(&val));
        }
        if val.ty.is_int() && elem.ty.is_int() {
            let op = self.as_int(&val, VTy::I32);
            return Ok(self.register(&Val { op, ty: VTy::I32 }));
        }
        let held = dtype_of(val.ty);
        if !val.ty.is_float() || !matches!(elem.dtype, F32 | F16 | BF16) {
            let stored = elem.dtype;
            return Err(self.no_rule(format_args!(
                "a fragment element of {held} stored as {stored}"
            )));
        }
        let reg = self.register(&val);
        if held == elem.dtype {
            return Ok(reg);
        }
        let wide = self.widened(reg, held)?;
        self.narrowed(wide, elem.dtype)
    }

    // --- element by element in registers -----------------------------------------------------------------

    /// One operation over an accumulator of floats, element by element: `out[i] = op(arguments at i)`. An
    /// argument is the element of a fragment, the scalar of the element's row, one scalar for every element, a
    /// condition over the element's own row and column, or a value loaded from them.
    pub(super) fn fragment_apply(&mut self, node: &FragmentApply, ctx: &SimplifyCtx) -> Result<()> {
        let Some(op) = node.op.known() else {
            let name = node.op.name().to_string();
            return Err(self.no_rule(format_args!("the operation {name}")));
        };
        let lanes = Lanes::of(&fragment_table(&node.layout, "c", DType::F32)?);
        let (count, rows) = (lanes.registers.len(), lanes.rows());
        let out = if self.is_fragment(node.out) {
            self.tensor_fragment(node.out, count, VTy::F32)?
        } else if node.in_place {
            let name = node.out;
            return Err(self.refuse(format_args!("{name} is updated before it is declared")));
        } else {
            self.declare_fragment(node.out, count, VTy::F32)
        };
        for (at, register) in lanes.registers.iter().enumerate() {
            let (row, column) = &register[0];
            let mut place = HashMap::new();
            for (name, base, offset) in [
                (FRAGMENT_ROW, &node.row_base, row),
                (FRAGMENT_COLUMN, &node.col_base, column),
            ] {
                if let Some(base) = base {
                    place.insert(Name::new(name), plus(base.clone(), offset.clone()));
                }
            }
            self.names.push(HashMap::new());
            let done = (|| {
                let mut arguments = Vec::with_capacity(node.args.len());
                for (position, argument) in node.args.iter().enumerate() {
                    let held = Name::new(&format!("__fa{position}"));
                    let float = |this: &mut Lowerer, reg: Reg| {
                        this.bind(held, Val::reg(reg, VTy::F32));
                        var(held)
                    };
                    arguments.push(match (&*argument.kind, &argument.names[..]) {
                        ("frag", [fragment]) => {
                            let regs = self.tensor_fragment(*fragment, count, VTy::F32)?;
                            let val = self.named(Val::reg(regs[at], VTy::F32));
                            self.bind(held, val);
                            var(held)
                        }
                        ("row", names) if names.len() > rows[at] => var(names[rows[at]]),
                        ("uniform", []) if argument.expressions.len() == 1 => {
                            argument.expressions[0].clone()
                        }
                        ("uniform", [name]) => {
                            let val = self.lookup(*name)?;
                            match val.ty {
                                VTy::F32 => var(*name),
                                ty => {
                                    let reg = self.register(&val);
                                    let wide = self.widened(reg, dtype_of(ty))?;
                                    float(self, wide)
                                }
                            }
                        }
                        ("coord", []) if argument.expressions.len() == 1 => {
                            argument.expressions[0].substitute(&place)
                        }
                        ("gmem", []) if argument.buffer.is_some() => {
                            let name = argument
                                .buffer
                                .expect("a gathered operand names its buffer");
                            let index: Vec<Expression> = argument
                                .expressions
                                .iter()
                                .map(|coordinate| coordinate.substitute(&place))
                                .collect();
                            let buffer = self.buffer(name)?;
                            let address = self.address(name, &index, ctx)?;
                            let value = self.gathered(&buffer, address, None);
                            let wide = self.widened(value, buffer.elem.dtype)?;
                            float(self, wide)
                        }
                        (kind, _) => {
                            return Err(self.refuse(format_args!(
                                "a fragment operand of kind {kind} names what that kind takes"
                            )));
                        }
                    });
                }
                self.float(&Expression::apply(op, arguments), VTy::F32, ctx)
            })();
            self.names.pop();
            let value = done?;
            self.b
                .emit(Inst::new(Opcode::Mov, Ty::F32, vec![out[at]], vec![value]));
        }
        Ok(())
    }

    /// The fold of each of a lane's two rows over a chunk of accumulators side by side: first over the lane's
    /// own elements of the row, in the fragments' order, then with the lanes that hold the row's other columns, by
    /// the table's shuffles. Every lane of the warp takes part, and each then holds its row's whole fold.
    pub(super) fn fragment_row_reduce(&mut self, node: &FragmentRowReduce) -> Result<()> {
        let opcode = match node.op.known() {
            Some(Operation::Add | Operation::Sum) => Opcode::Add,
            Some(Operation::Maximum | Operation::Amax | Operation::Fmax) => Opcode::Max,
            Some(Operation::Minimum | Operation::Fmin) => Opcode::Min,
            Some(Operation::Multiply | Operation::Prod) => Opcode::Mul,
            _ => {
                let name = node.op.name().to_string();
                return Err(self.no_rule(format_args!("a row reduction of {name}")));
            }
        };
        if node.dtype != DType::F32 {
            let dtype = node.dtype;
            return Err(self.no_rule(format_args!("a row reduction at {dtype}")));
        }
        let lanes = Lanes::of(&fragment_table(&node.layout, "c", DType::F32)?);
        let (count, rows) = (lanes.registers.len(), lanes.rows());
        let mut frags = Vec::with_capacity(node.frags.len());
        for name in node.frags.iter() {
            frags.push(self.tensor_fragment(*name, count, VTy::F32)?);
        }
        let mut folds = Vec::with_capacity(2);
        for (row, name) in [node.top, node.bot].into_iter().enumerate() {
            let fold = self.b.state(Bank::F32);
            let mut parts = frags
                .iter()
                .flat_map(|frag| frag.iter().zip(&rows).filter(|(_, at)| **at == row))
                .map(|(reg, _)| *reg);
            let Some(first) = parts.next() else {
                return Err(self.refuse("a row reduction folds a fragment's elements"));
            };
            let start = Inst::new(Opcode::Mov, Ty::F32, vec![fold], vec![first.into()]);
            self.b.emit(start);
            for part in parts {
                let step = self.float_inst(opcode, VTy::F32, vec![fold.into(), part.into()]);
                self.b.emit(Inst {
                    results: vec![fold],
                    ..step
                });
            }
            if let Some(scope) = self.names.last_mut() {
                scope.insert(name, Val::reg(fold, VTy::F32));
            }
            folds.push(fold);
        }
        let every_lane = Operand::Int(-1);
        for xor in &lanes.reduce_xors {
            for fold in &folds {
                let theirs =
                    self.shuffled(Shuffle::Bfly, (*fold, VTy::F32), (*xor, 31), &every_lane)?;
                let step = self.float_inst(opcode, VTy::F32, vec![(*fold).into(), theirs.into()]);
                self.b.emit(Inst {
                    results: vec![*fold],
                    ..step
                });
            }
        }
        Ok(())
    }
}
