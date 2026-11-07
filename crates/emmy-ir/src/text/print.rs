//! The printer: a statement to its line, a block's statements indented under its header, a body to its lines.

use std::sync::Arc;

use super::INDENT;
use crate::body::{Body, Subroutine};
use crate::expr::Expression;
use crate::kinds::axis::{Axis, spell_axis, spell_window};
use crate::kinds::field::{DType, NumOr, spell_number, spell_stmt_name};
use crate::kinds::*;
use crate::{Error, Name, Result};

fn names(items: &[Name]) -> Result<String> {
    Ok(items
        .iter()
        .map(|name| spell_stmt_name(name))
        .collect::<Result<Vec<_>>>()?
        .join(", "))
}

fn typed(dtype: &Option<DType>) -> String {
    dtype
        .as_ref()
        .map_or_else(String::new, |d| format!("{} ", d.name()))
}

fn index(items: &[Expression]) -> Result<String> {
    Ok(format!(
        "[{}]",
        items
            .iter()
            .map(|e| e.try_text())
            .collect::<Result<Vec<_>>>()?
            .join(", ")
    ))
}

fn flags(unroll: bool, seed: bool) -> String {
    format!(
        "{}{}:",
        if unroll { " unroll=true" } else { "" },
        if seed { "" } else { " seed=false" }
    )
}

fn affine(stmt: &Affine) -> Result<String> {
    let mut terms: Vec<(f64, String)> = Vec::new();
    for (c, arg) in stmt.terms() {
        let name = spell_stmt_name(arg)?;
        terms.push((
            c,
            if c.abs() == 1.0 {
                name
            } else {
                format!("{}*{}", spell_number(c.abs()), name)
            },
        ));
    }
    if stmt.offset.0 != 0.0 {
        terms.push((stmt.offset.0, spell_number(stmt.offset.0.abs())));
    }
    let Some((head, rest)) = terms.split_first() else {
        return Ok(String::new());
    };
    let mut out = format!("{}{}", if head.0 < 0.0 { "-" } else { "" }, head.1);
    for (c, term) in rest {
        out.push_str(&format!(" {} {}", if *c < 0.0 { "-" } else { "+" }, term));
    }
    Ok(out)
}

/// `stmt`'s text as lines, each starting with `indent`.
pub fn lines(stmt: &Statement, indent: &str, out: &mut Vec<String>) -> Result<()> {
    let inner = format!("{indent}{INDENT}");
    let text = match stmt {
        Statement::Load(s) => {
            let carried = if s.carried.is_empty() {
                String::new()
            } else {
                format!(" carried={}", s.carried)
            };
            format!(
                "{}{} = load {}{}{}",
                typed(&s.dtype),
                names(&s.names)?,
                spell_stmt_name(&s.input)?,
                index(&s.index)?,
                carried
            )
        }
        Statement::Pre(s) => format!(
            "{} = pre {}{}",
            spell_stmt_name(&s.name)?,
            spell_stmt_name(&s.carrier)?,
            index(&s.index)?
        ),
        Statement::Assign(s) => format!(
            "{}{} = {}({})",
            typed(&s.dtype),
            spell_stmt_name(&s.name)?,
            s.op.name,
            names(&s.args)?
        ),
        Statement::Affine(s) => format!(
            "{}{} = {}",
            typed(&s.dtype),
            spell_stmt_name(&s.name)?,
            affine(s)?
        ),
        Statement::Let(s) => format!(
            "let {}{} = {}",
            typed(&s.dtype),
            spell_stmt_name(&s.name)?,
            s.value.try_text()?
        ),
        Statement::Init(s) => format!(
            "init {} {} = {}",
            s.dtype.name(),
            spell_stmt_name(&s.name)?,
            spell_number(s.identity.0)
        ),
        Statement::Accum(s) => {
            let over = if s.axes.is_empty() {
                String::new()
            } else {
                format!(" over {}", names(&s.axes)?)
            };
            let base = s.base.as_ref().unwrap_or(&s.name);
            format!(
                "{}{} <- {}({}){}",
                typed(&s.dtype),
                spell_stmt_name(&s.name)?,
                s.op.name,
                names(&[base.clone(), s.value.clone()])?,
                over
            )
        }
        Statement::Carry(s) => {
            let seed = match &s.seed {
                NumOr::Num(n) => spell_number(n.as_f64()),
                NumOr::Value(name) => spell_stmt_name(name)?,
            };
            format!(
                "{}{}{} <- {} seed {}",
                typed(&s.dtype),
                spell_stmt_name(&s.name)?,
                index(&s.index)?,
                spell_stmt_name(&s.value)?,
                seed
            )
        }
        Statement::Write(s) => {
            let value = if s.values.len() > 1 {
                format!("({})", names(&s.values)?)
            } else {
                spell_stmt_name(&s.values[0])?
            };
            let swizzle = if &*s.swizzle != "NONE" {
                format!(" swizzle={}", spell_stmt_name(&s.swizzle)?)
            } else {
                String::new()
            };
            let store = if s.atomic { "+=" } else { "=" };
            format!(
                "{}{}{} {} {}{}",
                typed(&s.value_dtype),
                spell_stmt_name(&s.output)?,
                index(&s.index)?,
                store,
                value,
                swizzle
            )
        }
        Statement::Select(s) => {
            out.push(format!("{indent}{} = select:", spell_stmt_name(&s.name)?));
            for branch in s.branches.iter() {
                out.push(format!(
                    "{inner}{} when {}",
                    spell_stmt_name(&branch.value)?,
                    branch.select.try_text()?
                ));
            }
            return Ok(());
        }
        Statement::ZeroPrologue(s) => {
            format!("zero {} words={}", spell_stmt_name(&s.dst)?, s.words)
        }
        Statement::Loop(s) => {
            out.push(format!(
                "{indent}for {}{}",
                spell_axis(&s.axis, None)?,
                flags(s.unroll, s.seed)
            ));
            return body_lines(&s.body, &inner, out);
        }
        Statement::StridedLoop(s) => {
            let until = match &s.end {
                Some(end) => format!(" until {}", end.try_text()?),
                None => String::new(),
            };
            let step = s.step.try_text()?;
            let start = s.start.try_text()?;
            // The window follows the extent, so spell the axis without it and add it after the step.
            let bare = Axis {
                name: s.axis.name.clone(),
                extent: s.axis.extent.clone(),
                window: None,
            };
            out.push(format!(
                "{indent}for {} step {}{}{}{}",
                spell_axis(&bare, Some(&start))?,
                step,
                until,
                spell_window(s.axis.window.as_ref())?,
                flags(s.unroll, s.seed)
            ));
            return body_lines(&s.body, &inner, out);
        }
        Statement::Cond(s) => {
            out.push(format!("{indent}if {}:", s.cond.try_text()?));
            body_lines(&s.body, &inner, out)?;
            if !s.else_body.is_empty() {
                out.push(format!("{indent}else:"));
                body_lines(&s.else_body, &inner, out)?;
            }
            return Ok(());
        }
        Statement::Call(s) => {
            let args = s
                .args
                .iter()
                .map(|a| a.try_text())
                .collect::<Result<Vec<_>>>()?
                .join(", ");
            format!(
                "{} = call {}[{}]({})",
                spell_stmt_name(&s.name)?,
                spell_stmt_name(&s.target.name)?,
                names(&s.target.buffers)?,
                args
            )
        }
        _ => return declared_lines(stmt, indent, out),
    };
    out.push(format!("{indent}{text}"));
    Ok(())
}

/// `Kind(field=value, …)` over the fields that are not at their default and hold no statements.
fn header(stmt: &Statement) -> Result<String> {
    let fields = stmt.header_fields()?;
    let mut out = stmt.kind().to_string();
    if !fields.is_empty() {
        out.push('(');
        out.push_str(
            &fields
                .iter()
                .map(|(name, text)| format!("{name}={text}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push(')');
    }
    Ok(out)
}

fn declared_lines(stmt: &Statement, indent: &str, out: &mut Vec<String>) -> Result<()> {
    let inner = format!("{indent}{INDENT}");
    let blocks = stmt.block_fields();
    out.push(format!(
        "{indent}{}{}",
        header(stmt)?,
        if blocks.is_empty() { "" } else { ":" }
    ));
    for (name, stmts) in blocks {
        if name == "body" {
            stmts_lines(&stmts, &inner, out)?;
        } else {
            out.push(format!("{inner}{name}:"));
            stmts_lines(&stmts, &format!("{inner}{INDENT}"), out)?;
        }
    }
    Ok(())
}

fn stmts_lines(stmts: &[Arc<Statement>], indent: &str, out: &mut Vec<String>) -> Result<()> {
    for stmt in stmts {
        lines(stmt, indent, out)?;
    }
    Ok(())
}

/// The lines of a body's text, each starting with `indent`.
pub fn body_lines(body: &Body, indent: &str, out: &mut Vec<String>) -> Result<()> {
    stmts_lines(body.stmts(), indent, out)
}

/// A body's whole text: the subroutines its calls name, each once and ahead of its callers, then its statements.
pub fn body_text(body: &Body) -> Result<String> {
    let mut out = Vec::new();
    let mut found: Vec<Arc<Subroutine>> = Vec::new();
    subroutines(body, &mut found)?;
    for sub in &found {
        definition(sub, &mut out)?;
    }
    body_lines(body, "", &mut out)?;
    Ok(out.join("\n"))
}

/// One statement's text, with the subroutines its calls name ahead of it.
pub fn statement_text(stmt: &Arc<Statement>) -> Result<String> {
    body_text(&Body::new(vec![stmt.clone()]))
}

/// A statement's text as digest material: its text, or its debug form for one that has no text (a name with a
/// backtick), which is as structural.
pub fn structural_text(stmt: &Arc<Statement>) -> String {
    statement_text(stmt).unwrap_or_else(|_| format!("{stmt:?}"))
}

/// A body's text as digest material (`structural_text` over a body).
pub fn structural_body_text(body: &Body) -> String {
    body_text(body).unwrap_or_else(|_| format!("{body:?}"))
}

fn definition(sub: &Subroutine, out: &mut Vec<String>) -> Result<()> {
    let axes = sub
        .axes
        .iter()
        .map(|axis| spell_axis(axis, None))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    out.push(format!(
        "sub {}({}) -> {}:",
        spell_stmt_name(&sub.name)?,
        axes,
        spell_stmt_name(&sub.result)?
    ));
    body_lines(&sub.body, INDENT, out)
}

fn definition_text(sub: &Subroutine) -> Result<Vec<String>> {
    let mut out = Vec::new();
    definition(sub, &mut out)?;
    Ok(out)
}

/// The subroutines `body` calls, a callee ahead of its caller, each name once; two subroutines of one name are
/// refused unless they are one definition bound to other buffers.
fn subroutines(body: &Body, found: &mut Vec<Arc<Subroutine>>) -> Result<()> {
    for here in body.of_kinds(&[Kind::Call]) {
        let Statement::Call(call) = &**here.stmt() else {
            continue;
        };
        let target = &call.target.0;
        subroutines(&target.body, found)?;
        match found.iter().find(|sub| sub.name == target.name) {
            None => found.push(target.clone()),
            Some(first) if Arc::ptr_eq(first, target) => {}
            Some(first) => {
                if definition_text(&first.with_buffers(target.buffers.clone()))?
                    != definition_text(target)?
                {
                    return Err(Error::Value(format!(
                        "two subroutines are named {}: a body's text defines a name once",
                        crate::expr::python_repr(&first.name)
                    )));
                }
            }
        }
    }
    Ok(())
}
