//! The checks a statement type makes on construction, each over its own fields: what Python's `__post_init__` refused.

use crate::statement::*;
use crate::{Error, Result};
use std::sync::Arc;

/// The checks Python's `__post_init__` made, each statement type over its own fields.
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
        let crate::field::Shape3(m, n, k) = self.shape;
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
        if self.lane.is_some() && (self.inner.is_some() || self.barrier_id != 0 || self.length > 32)
        {
            return Err(Error::Value(
                "TreeHalve: the register fold of a per-warp slab (lane) takes one warp's slots, no segment and the \
                 CTA barrier"
                    .into(),
            ));
        }
        carried_type(&self.combine_states, StatementType::Assign)
    }
}

impl WarpShuffle {
    pub fn validate(&self) -> Result<()> {
        carried_type(&self.combine_states, StatementType::Assign)
    }
}

impl RegStore {
    pub fn validate(&self) -> Result<()> {
        carried_type(&self.run, StatementType::RegStore)
    }
}

impl RegFragment {
    /// A fragment names a row of the lane-to-element table: its register count is that row's.
    pub fn validate(&self) -> Result<()> {
        self.registers().map(|_| ())
    }
}

/// An operand fragment is one of the two multiplicands.
fn operand_role(statement: &str, role: &str) -> Result<()> {
    match role {
        "a" | "b" => Ok(()),
        _ => Err(Error::Value(format!(
            "{statement}: the role is a or b, got {role}"
        ))),
    }
}

/// A row stride is a count of elements.
fn stride(statement: &str, ldm: i64) -> Result<()> {
    if ldm <= 0 {
        return Err(Error::Value(format!(
            "{statement}: ldm is the source's row stride in elements, got {ldm}"
        )));
    }
    Ok(())
}

impl LdmatrixLoad {
    pub fn validate(&self) -> Result<()> {
        operand_role("LdmatrixLoad", &self.role)?;
        stride("LdmatrixLoad", self.ldm)?;
        if &*self.role == "a" && (self.b_trans || self.pair_frag.is_some()) {
            return Err(Error::Value(
                "LdmatrixLoad: an A fragment is one x4 load (no b_trans, no pair_frag)".into(),
            ));
        }
        Ok(())
    }
}

impl FragmentLoad {
    pub fn validate(&self) -> Result<()> {
        operand_role("FragmentLoad", &self.role)?;
        if let Some(ldm) = self.ldm {
            stride("FragmentLoad", ldm)?;
        }
        if self.staged && (self.clamp.is_some() || self.k_zero.is_some()) {
            return Err(Error::Value(
                "FragmentLoad: a staged slab is filled in range (no clamp, no k_zero)".into(),
            ));
        }
        let slab_only =
            &*self.slab_layout != "NONE" || self.pair_frag.is_some() || self.scale_buffer.is_some();
        if !self.staged && slab_only {
            return Err(Error::Value(
                "FragmentLoad: slab_layout, pair_frag and scale_buffer describe a staged slab"
                    .into(),
            ));
        }
        if self.scale_buffer.is_some() {
            stride("FragmentLoad.scale", self.scale_ldm)?;
        }
        Ok(())
    }
}

impl BlockScaleLoad {
    pub fn validate(&self) -> Result<()> {
        operand_role("BlockScaleLoad", &self.role)?;
        stride("BlockScaleLoad", self.ldm)
    }
}

fn carried_type(stmts: &[Arc<Statement>], ty: StatementType) -> Result<()> {
    match stmts.iter().find(|s| s.ty() != ty) {
        Some(other) => Err(Error::Type(format!(
            "expected {ty} statements, got a {}",
            other.ty()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Name;
    use crate::field::{Names, Shape3};
    use crate::text::parse_statement;

    /// A wgmma cell over eight C fragments at m64n64k16, its A operand named by `a_operand`.
    fn cell(a_operand: &str) -> WgmmaMma {
        let stmt = parse_statement(&format!(
            "WgmmaMma(c_frags=(c0, c1, c2, c3, c4, c5, c6, c7), b_desc=bD, shape=(64, 64, 16), {a_operand})"
        ))
        .unwrap();
        <&WgmmaMma>::try_from(&*stmt).unwrap().clone()
    }

    /// What the constructor's checks refuse `cell` with: a value error's message.
    fn refusal(cell: WgmmaMma) -> String {
        match finished(Statement::WgmmaMma(cell)) {
            Err(Error::Value(message)) => message,
            other => panic!("not refused with a value error: {other:?}"),
        }
    }

    #[test]
    fn the_cell_rejects_malformed_operands() {
        let both = WgmmaMma {
            a_frag: Some("a".into()),
            ..cell("a_desc=aD")
        };
        assert!(refusal(both).contains("exactly one of a_desc / a_frag"));
        let sixteen = WgmmaMma {
            c_frags: (0..16).map(|j| Name::from(format!("c{j}"))).collect(),
            ..cell("a_desc=aD")
        };
        assert!(refusal(sixteen).contains("8 m16n8 C fragments"));
        let transposed = WgmmaMma {
            trans_a: 1,
            ..cell("a_frag=a")
        };
        assert!(refusal(transposed).contains("K-major"));
        let mma_sync = WgmmaMma {
            c_frags: Names::from([Name::from("c0")]),
            shape: Shape3(16, 8, 16),
            ..cell("a_desc=aD")
        };
        assert!(refusal(mma_sync).contains("m64n"));
    }

    /// An expression field admits its class and refuses the rest: an index is integer arithmetic over axes, built-ins,
    /// integer literals and names; a value, which a `let` binds and a condition compares, is any operation of the table
    /// over any operand.
    #[test]
    fn a_classed_field_refuses_an_expression_outside_its_class() {
        for text in [
            "in0 = load x[a0 * 4 + (a1 // 2) % 3, a0 < 8 ? thread_idx.x : (int)n, minimum(a0, 7) & 1, (a1 + 1) % seq_len]",
            "let v0 = 2.5 * v1 + exp(v2) - thread_idx.x",
            "if x + 1.0 < 0.0 && a0 == 1:\n    y[a0] = v0",
            "if a0 + 1 <= n && a1 == 0:\n    y[a0] = v0",
        ] {
            parse_statement(text).unwrap_or_else(|err| panic!("{err}\n{text}"));
        }
        for (text, why) in [
            ("in0 = load x[a0 + 0.5]", "a float literal"),
            ("y[exp(a0)] = v0", "the operation exp"),
        ] {
            let err = parse_statement(text)
                .err()
                .unwrap_or_else(|| panic!("accepted {text}"))
                .to_string();
            assert!(
                err.contains("expression") && err.contains(why),
                "{text}: {err}"
            );
        }
    }
}
