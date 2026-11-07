//! Element types by name. The compiler's `DataType` objects live in Python; a statement holds a dtype as its
//! canonical name, which is what every rule here reads.

use crate::Name;

/// The integer dtype tokens: arithmetic on them stays in the integer family.
pub const INTEGER_NAMES: [&str; 7] = ["i16", "i32", "i64", "u8", "u16", "u32", "u64"];

/// The floating dtypes a literal computes at: what constant folding and the affine fold run over.
pub const FLOAT_NAMES: [&str; 3] = ["f32", "f16", "bf16"];

const INTEGER_ELEMENTWISE_OPS: [&str; 13] = [
    "add",
    "subtract",
    "negative",
    "multiply",
    "floor_divide",
    "remainder",
    "mod",
    "left_shift",
    "right_shift",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_count",
];

pub fn is_integer(name: &str) -> bool {
    INTEGER_NAMES.contains(&name)
}

pub fn is_float(name: &str) -> bool {
    FLOAT_NAMES.contains(&name)
}

fn width(name: &str) -> i32 {
    match name {
        "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        "i64" | "u64" => 64,
        _ => 0,
    }
}

/// Promote an elementwise op's argument dtypes to one result dtype: the common integer dtype by the C rank and sign
/// rule for integer arithmetic on integers, `f16` when every argument is `f16`, else `f32`.
pub fn promote(op_name: &str, arg_dtypes: &[&str]) -> Name {
    if !arg_dtypes.is_empty() && INTEGER_ELEMENTWISE_OPS.contains(&op_name) && arg_dtypes.iter().all(|d| is_integer(d)) {
        let signed = arg_dtypes.iter().filter(|d| d.starts_with('i')).map(|d| width(d)).max().unwrap_or(0);
        let unsigned = arg_dtypes.iter().filter(|d| d.starts_with('u')).map(|d| width(d)).max().unwrap_or(0);
        return if unsigned >= signed { format!("u{unsigned}").into() } else { format!("i{signed}").into() };
    }
    if !arg_dtypes.is_empty() && arg_dtypes.iter().all(|d| *d == "f16") {
        return "f16".into();
    }
    "f32".into()
}
