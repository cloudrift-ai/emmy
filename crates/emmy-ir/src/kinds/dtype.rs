//! Element types by name. The compiler's `DataType` objects live in Python; a statement holds a dtype by its canonical
//! name, which is what every rule here reads.

use crate::closed_set;
use crate::kinds::elementwise::Elementwise;

closed_set! {
    /// An element type: the compiler's `DataType` registry, by canonical name, the two packed pairs included.
    pub enum DType ("dtype") {
        F32 = "f32", F16 = "f16", F64 = "f64", BF16 = "bf16", F8E4M3 = "f8e4m3", F8E5M2 = "f8e5m2",
        I16 = "i16", I32 = "i32", I64 = "i64", U8 = "u8", U16 = "u16", U32 = "u32", U64 = "u64", Bool = "bool",
        F16x2 = "f16x2", F4E2M1x2 = "f4e2m1x2",
    }
}

impl DType {
    pub fn name(self) -> &'static str {
        self.text()
    }

    /// An integer dtype: arithmetic on them stays in the integer family.
    pub fn is_integer(self) -> bool {
        use DType::*;
        matches!(self, I16 | I32 | I64 | U8 | U16 | U32 | U64)
    }

    /// A floating dtype a literal computes at: what constant folding and the affine fold run over.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    fn width(self) -> i32 {
        use DType::*;
        match self {
            U8 => 8,
            I16 | U16 => 16,
            I32 | U32 => 32,
            I64 | U64 => 64,
            _ => 0,
        }
    }

    fn signed(self) -> bool {
        matches!(self, DType::I16 | DType::I32 | DType::I64)
    }

    fn integer(signed: bool, width: i32) -> DType {
        use DType::*;
        match (signed, width) {
            (false, 8) => U8,
            (true, 16) => I16,
            (false, 16) => U16,
            (true, 32) => I32,
            (false, 32) => U32,
            (true, _) => I64,
            (false, _) => U64,
        }
    }
}

/// Promote an elementwise op's argument dtypes to one result dtype: the common integer dtype by the C rank and sign
/// rule for integer arithmetic on integers, `f16` when every argument is `f16`, else `f32`.
pub fn promote(op: Option<Elementwise>, arg_dtypes: &[DType]) -> DType {
    if !arg_dtypes.is_empty()
        && op.is_some_and(Elementwise::integer)
        && arg_dtypes.iter().all(|d| d.is_integer())
    {
        let widest = |signed: bool| {
            arg_dtypes
                .iter()
                .filter(|d| d.signed() == signed)
                .map(|d| d.width())
                .max()
                .unwrap_or(0)
        };
        let (signed, unsigned) = (widest(true), widest(false));
        return if unsigned >= signed {
            DType::integer(false, unsigned)
        } else {
            DType::integer(true, signed)
        };
    }
    if !arg_dtypes.is_empty() && arg_dtypes.iter().all(|d| *d == DType::F16) {
        return DType::F16;
    }
    DType::F32
}
