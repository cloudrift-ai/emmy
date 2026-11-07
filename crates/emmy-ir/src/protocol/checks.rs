//! The checks a kind makes on construction, each over its own fields: what Python's `__post_init__` refused.

use crate::kinds::*;
use crate::{Error, Result};
use std::sync::Arc;

/// The checks Python's `__post_init__` made, each kind over its own fields.
impl Load {
    pub fn validate(&self) -> Result<()> {
        if self.names.is_empty() {
            return Err(Error::Value("Load.names must be non-empty".into()));
        }
        Ok(())
    }
}

impl Write {
    pub fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(Error::Value("Write.values must be non-empty".into()));
        }
        if self.atomic && self.values.len() != 1 {
            return Err(Error::Value(
                "an atomic Write is scalar (one value per atomicAdd)".into(),
            ));
        }
        Ok(())
    }
}

impl Accum {
    /// The ordinary self-fold has one spelling: a base that is the state itself is no base.
    pub fn normalized(self) -> Accum {
        let base = self.base.filter(|b| *b != self.name);
        Accum { base, ..self }
    }
}

impl Select {
    pub fn validate(&self) -> Result<()> {
        if self.branches.is_empty() {
            return Err(Error::Value("Select.branches must be non-empty".into()));
        }
        Ok(())
    }
}

impl Call {
    pub fn validate(&self) -> Result<()> {
        if self.args.len() != self.target.axes.len() {
            return Err(Error::Value(
                "subroutine arguments do not match its parameters".into(),
            ));
        }
        Ok(())
    }
}

impl Tile {
    pub fn validate(&self) -> Result<()> {
        if self.aux_threads != 0 && self.block_threads.is_none() {
            return Err(Error::Value(
                "Tile.aux_threads requires a cooperative block_threads".into(),
            ));
        }
        Ok(())
    }
}

impl WgmmaMma {
    pub fn validate(&self) -> Result<()> {
        let crate::kinds::field::Shape3(m, n, k) = self.shape;
        if (m, k) != (64, 16) || n % 8 != 0 || !(8..=256).contains(&n) {
            return Err(Error::Value(format!(
                "wgmma: the shape is m64n<8..256, a multiple of 8>k16, got ({m}, {n}, {k})"
            )));
        }
        if self.c_frags.len() as i64 != n / 8 {
            return Err(Error::Value(format!(
                "wgmma m64n{n}: the accumulator is {} m16n8 C fragments, got {}",
                n / 8,
                self.c_frags.len()
            )));
        }
        if self.a_desc.is_none() == self.a_frag.is_none() {
            return Err(Error::Value(
                "wgmma: exactly one of a_desc / a_frag names the A operand".into(),
            ));
        }
        if self.a_frag.is_some() && self.trans_a != 0 {
            return Err(Error::Value(
                "wgmma: the register-form A operand is K-major (no transpose bit)".into(),
            ));
        }
        Ok(())
    }
}

impl TreeHalve {
    pub fn validate(&self) -> Result<()> {
        carried_kind(&self.combine_states, Kind::Assign)
    }
}

impl WarpShuffle {
    pub fn validate(&self) -> Result<()> {
        carried_kind(&self.combine_states, Kind::Assign)
    }
}

impl RegStore {
    pub fn validate(&self) -> Result<()> {
        carried_kind(&self.run, Kind::RegStore)
    }
}

fn carried_kind(stmts: &[Arc<Statement>], kind: Kind) -> Result<()> {
    match stmts.iter().find(|s| s.kind() != kind) {
        Some(other) => Err(Error::Type(format!(
            "expected {kind} statements, got a {}",
            other.kind()
        ))),
        None => Ok(()),
    }
}

/// A statement with its `Accum` base normalized and its carried numbers settled: what every constructor returns.
pub fn finished(stmt: Statement) -> Result<Statement> {
    stmt.check()?;
    Ok(match stmt {
        Statement::Accum(a) => Statement::Accum(a.normalized()),
        other => other,
    })
}
