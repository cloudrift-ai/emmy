//! The typed-body check: a body a code generator reads says every type itself.
//!
//! Every value-defining statement carries its dtype; an operation takes every operand at the dtype of its definition
//! and yields it; a value read at another dtype is converted by a definition of its own, `name = (dtype)source`; a
//! write stores values of its buffer's element type. A generator that reads such a body decides no promotion and no
//! conversion. The pass that brings a body to this form lives in Python (`ir.types.explicit_conversions`); this is
//! the property as a check, and its error names the statement.

use std::collections::{HashMap, HashSet};

use crate::body::Body;
use crate::dtype::DType;
use crate::elementwise::Operation;
use crate::expr::{CastTarget, Expr, Expression, LitType};
use crate::statement::*;
use crate::text::print::statement_text;
use crate::{Error, Name, Result};

/// Check that `body` is typed in full. `buffers` gives the element dtype of the buffers the body reads and writes by
/// name; a write to a buffer it does not hold is checked against the write's own dtype alone.
pub fn check_types(body: &Body, buffers: &HashMap<Name, DType>) -> Result<()> {
    let mut scope = Scope {
        buffers,
        types: HashMap::new(),
        defined: HashSet::new(),
    };
    scope.declare(body);
    scope.body(body)
}

struct Scope<'a> {
    buffers: &'a HashMap<Name, DType>,
    /// The dtype of each name a typed statement defines. A body's names are unique, and a definition may stand in a
    /// sibling scope behind its reader, so the declared ones are collected before the walk.
    types: HashMap<Name, DType>,
    /// Every name some statement defines.
    defined: HashSet<Name>,
}

impl Scope<'_> {
    fn declare(&mut self, body: &Body) {
        for stmt in body.stmts() {
            self.defined.extend(stmt.defines());
            match &**stmt {
                Statement::Load(load) => {
                    if let Some(dtype) = load.dtype {
                        self.types
                            .extend(load.names.iter().map(|name| (*name, dtype)));
                    }
                }
                Statement::Let(index) if index.dtype.is_none() => {
                    self.types.insert(index.name, DType::I32);
                }
                Statement::Assign(_)
                | Statement::Accum(_)
                | Statement::Carry(_)
                | Statement::Init(_)
                | Statement::Let(_) => {
                    if let (Some(name), Some(dtype)) = (stmt.name(), stmt.dtype()) {
                        self.types.insert(*name, *dtype);
                    }
                }
                _ => {}
            }
            for nested in stmt.nested() {
                self.declare(&nested);
            }
        }
    }

    /// The dtype of a name. One no typed statement defines is f32 when an exempt statement defines it (a fragment
    /// statement's scalar), and an integer when the body only reads it: an axis, a symbolic size, a thread coordinate.
    fn dtype(&self, name: &Name) -> DType {
        match self.types.get(name) {
            Some(dtype) => *dtype,
            None if self.defined.contains(name) => DType::F32,
            None => DType::I32,
        }
    }

    fn body(&mut self, body: &Body) -> Result<()> {
        for stmt in body.stmts() {
            self.statement(stmt).map_err(|why| {
                let text = statement_text(stmt).unwrap_or_else(|_| format!("{stmt:?}"));
                Error::Type(format!("untyped statement `{}`: {why}", text.trim()))
            })?;
            for nested in stmt.nested() {
                self.body(&nested)?;
            }
        }
        Ok(())
    }

    fn statement(&mut self, stmt: &Statement) -> std::result::Result<(), String> {
        match stmt {
            Statement::Load(load) => {
                declared(load.dtype)?;
            }
            Statement::Assign(assign) => self.value(&assign.value, declared(assign.dtype)?)?,
            Statement::Accum(accum) => self.operand(&accum.value, declared(accum.dtype)?)?,
            Statement::Carry(carry) => self.name(&carry.value, declared(carry.dtype)?)?,
            Statement::Pre(pre) => {
                let dtype = self.dtype(&pre.carrier);
                self.types.insert(pre.name, dtype);
            }
            Statement::Let(binding) => {
                // An index with no dtype is an integer whose width the flat-index step decides (plan step A9).
                if binding.dtype.is_none() && binding.value.as_literal().is_some() {
                    return Err("a literal binding carries no dtype".into());
                }
            }
            Statement::Select(select) => {
                let Some(first) = select.branches.first() else {
                    return Ok(());
                };
                let dtype = self.dtype(&first.value);
                for branch in select.branches.iter() {
                    self.name(&branch.value, dtype)?;
                }
                self.types.insert(select.name, dtype);
            }
            Statement::Write(write) => {
                let dtype = declared(write.value_dtype)?;
                for value in write.values.iter() {
                    self.name(value, dtype)?;
                }
                if let Some(element) = self.buffers.get(&write.output)
                    && *element != dtype
                {
                    return Err(format!(
                        "stores {dtype} values into the {element} buffer {}",
                        write.output
                    ));
                }
            }
            // Exempt until the plan step named types them; the renderer still works their types out.
            //
            // Cross-thread combines: the merge program runs at the statement's dtype and converts a name read at
            // another one while printing. No step of the plan names them; the cooperative milestone of the PTX
            // compiler (PR B, milestone 2) has to type them first.
            Statement::TreeHalve(_) | Statement::WarpShuffle(_) | Statement::WarpBroadcast(_) => {}
            // A fragment gathered element by element says both element types, as a `Load` says its own.
            Statement::FragmentLoad(load) => {
                let (source, fragment) = (declared(load.src_dtype)?, declared(load.dtype)?);
                if let Some(element) = self.buffers.get(&load.src_buffer)
                    && *element != source
                {
                    return Err(format!(
                        "reads {source} elements from the {element} buffer {}",
                        load.src_buffer
                    ));
                }
                if fragment.is_integer() {
                    return Err(format!("a fragment of {fragment} elements"));
                }
            }
            // A store's epilogue is a body of its own, typed like any: its parameters are the f32 elements of the
            // accumulators, and the other names it reads are coordinates. The cells stored with it carry their own.
            Statement::RegStore(store) => {
                if let Some(epilogue) = &store.epilogue {
                    let mut scope = Scope {
                        buffers: self.buffers,
                        types: store
                            .epilogue_params
                            .iter()
                            .map(|p| (*p, DType::F32))
                            .collect(),
                        defined: store.epilogue_params.iter().copied().collect(),
                    };
                    scope.declare(epilogue);
                    scope.body(epilogue).map_err(|why| why.to_string())?;
                }
                for cell in store.run.iter() {
                    self.statement(cell)?;
                }
            }
            // What computes on a fragment's registers: a pointwise operand's conversion to the fragment algebra's
            // f32 is typed while printing, because its operands are register elements no name stands for. The
            // fragment expansion of the PTX compiler (PR B, milestone 4) types it from the fragments' declarations.
            // The other statements move whole registers and convert nothing.
            Statement::FragmentApply(_) | Statement::FragmentRowReduce(_) => {}
            // Everything else defines no scalar value: structure, declarations, copies of bytes, barriers.
            _ => {}
        }
        Ok(())
    }

    /// A definition's value at `dtype`: a name or a literal, one conversion, or one operation.
    fn value(&self, value: &Expression, dtype: DType) -> std::result::Result<(), String> {
        match &**value {
            Expr::Cast(target, _) => match cast_dtype(target) {
                Some(to) if to == dtype => Ok(()),
                Some(to) => Err(format!("a conversion to {to} defines a {dtype} value")),
                None => Err(format!("a cast to {target}, which is no dtype")),
            },
            Expr::Ternary(_, if_true, if_false) => {
                // The condition is a truth value of any dtype.
                self.operand(if_true, dtype)?;
                self.operand(if_false, dtype)
            }
            Expr::FlatIndex(..) | Expr::Builtin(_) if !dtype.is_integer() => {
                Err(format!("an index defines a {dtype} value"))
            }
            Expr::FlatIndex(..) | Expr::Builtin(_) => Ok(()),
            // A reinterpretation reads its operand's bits at whatever dtype holds them.
            Expr::Apply(..) if matches!(value.operation(), Some((Operation::Bitcast, _))) => Ok(()),
            // A comparison defines a truth value; its operands compute at one dtype, the first named one's.
            Expr::Apply(_, operands)
                if value.operation().is_some_and(|(op, _)| op.is_comparison()) =>
            {
                if dtype != DType::Bool {
                    return Err(format!("a comparison defines a {dtype} value"));
                }
                match operands.iter().find_map(|operand| operand.as_var()) {
                    Some(name) => operands
                        .iter()
                        .try_for_each(|operand| self.operand(operand, self.dtype(name))),
                    None => Ok(()),
                }
            }
            _ => self.operand(value, dtype),
        }
    }

    /// An operand of an operation at `dtype`: a name at that dtype, a literal it can spell, or a nested operation
    /// of the same region, whose operands are held to the same dtype.
    fn operand(&self, operand: &Expression, dtype: DType) -> std::result::Result<(), String> {
        match &**operand {
            Expr::Var(name) => self.name(name, dtype),
            Expr::Literal(literal) if literal.dtype == LitType::Float && dtype.is_integer() => {
                Err(format!("a float literal in {dtype} arithmetic"))
            }
            Expr::Literal(_) => Ok(()),
            Expr::Apply(_, operands) => operands
                .iter()
                .try_for_each(|operand| self.operand(operand, dtype)),
            _ => Err(format!(
                "the operand {} is neither a name, a literal nor an operation",
                operand
                    .try_text()
                    .unwrap_or_else(|_| format!("{operand:?}"))
            )),
        }
    }

    fn name(&self, name: &Name, dtype: DType) -> std::result::Result<(), String> {
        let own = self.dtype(name);
        if own == dtype {
            Ok(())
        } else {
            Err(format!(
                "reads {name} ({own}) at {dtype} with no conversion"
            ))
        }
    }
}

fn declared(dtype: Option<DType>) -> std::result::Result<DType, String> {
    dtype.ok_or_else(|| "it carries no dtype".to_string())
}

/// The dtype a cast converts to: a dtype by name, or one of the C spellings an index cast uses.
fn cast_dtype(target: &CastTarget) -> Option<DType> {
    match target {
        CastTarget::Known(LitType::Int) => Some(DType::I32),
        CastTarget::Known(LitType::Float) => Some(DType::F32),
        CastTarget::Known(LitType::Bool) => Some(DType::Bool),
        CastTarget::Other(name) if name.as_str() == "long long" => Some(DType::I64),
        CastTarget::Other(name) => name.as_str().parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse::parse_body;

    fn check(source: &str) -> Result<()> {
        let buffers = HashMap::from([(Name::new("O"), DType::F16), (Name::new("A"), DType::F16)]);
        check_types(&parse_body(source).unwrap(), &buffers)
    }

    #[test]
    fn a_body_with_every_conversion_spelled_passes() {
        check(
            "f16 a = load A[i]\n\
             f32 a__f32 = (f32)a\n\
             f32 y = a__f32 * 2 + a__f32\n\
             f16 y__f16 = (f16)y\n\
             f16 O[i] = y__f16",
        )
        .unwrap();
    }

    #[test]
    fn an_implied_conversion_names_its_statement() {
        let mixed = check("f16 a = load A[i]\nf32 y = a * 2").unwrap_err();
        assert!(
            mixed.to_string().contains("`f32 y = 2 * a`")
                || mixed.to_string().contains("`f32 y = a * 2`"),
            "{mixed}"
        );
        assert!(
            mixed.to_string().contains("reads a (f16) at f32"),
            "{mixed}"
        );
        let untyped = check("f16 a = load A[i]\ny = a").unwrap_err();
        assert!(
            untyped.to_string().contains("carries no dtype"),
            "{untyped}"
        );
        let store = check("f16 a = load A[i]\nf32 y = (f32)a\nf32 O[i] = y").unwrap_err();
        assert!(
            store.to_string().contains("into the f16 buffer O"),
            "{store}"
        );
    }

    #[test]
    fn a_comparison_defines_a_truth_value() {
        check("f16 a = load A[i]\nbool c = a < 1.0\nf32 m = (f32)c").unwrap();
        let implied = check("f32 a = (f32)i\nf32 m = a < 1.0").unwrap_err();
        assert!(
            implied
                .to_string()
                .contains("a comparison defines a f32 value"),
            "{implied}"
        );
    }

    #[test]
    fn a_name_the_body_only_reads_is_an_integer() {
        check("i32 k = i * 4 + n\nf32 x = (f32)k").unwrap();
        assert!(check("f32 x = i * 4").is_err());
    }
}
