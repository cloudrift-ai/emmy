//! The text of a body: one printer and one reader, exact inverses. One statement is one line; a block's statements
//! are indented four spaces under its header; a field at its default is left out, nothing else is. The shared kinds
//! have clauses of their own; a kind of another layer prints and reads through its declaration (`Kind(field=value,
//! …):` over its blocks). A subroutine a call names is defined once, ahead of the statements.

use std::collections::HashMap;
use std::sync::Arc;

use crate::axis::{Axis, Dim, Window};
use crate::body::{Body, Subroutine};
use crate::elementwise::Op;
use crate::expr::{Expr, Expression, Lit, Num, Parser, TokenKind, spell_name};
use crate::field::{DType, Float, NumOr, Ty, Value, Values, spell_number};
use crate::kinds::*;
use crate::protocol::finished;
use crate::{Error, Name, Result};

pub const INDENT: &str = "    ";

/// Words the reader gives a meaning, so a name spelled like one is written between backticks.
pub const KEYWORDS: [&str; 26] = [
    "for", "in", "if", "else", "let", "init", "zero", "load", "pre", "select", "when", "over",
    "seed", "step", "until", "window", "carried", "swizzle", "words", "unroll", "sub", "call",
    "true", "false", "none", "in",
];

/// A name as a statement's text spells it: bare, or between backticks when it reads as a keyword or is no identifier.
pub fn spell_stmt_name(name: &str) -> Result<String> {
    spell_name(name, &KEYWORDS)
}

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

/// `name in start..extent`, with the hint when it is not the default, and the window.
pub fn spell_axis(axis: &Axis, start: Option<&str>) -> Result<String> {
    let mut extent = axis.extent.expression.try_text()?;
    if axis.extent.hint != axis.extent.default_hint()
        && let Some(hint) = axis.extent.hint
    {
        extent.push_str(&format!("~{hint}"));
    }
    let mut out = format!(
        "{} in {}..{}",
        spell_stmt_name(&axis.name)?,
        start.unwrap_or("0"),
        extent
    );
    out.push_str(&spell_window(axis.window.as_ref())?);
    Ok(out)
}

fn spell_window(window: Option<&Window>) -> Result<String> {
    let Some(window) = window else {
        return Ok(String::new());
    };
    let mut parts: Vec<String> = Vec::new();
    if let Some(parent) = &window.parent {
        parts.push(format!("parent={}", spell_axis(parent, None)?));
    }
    if let Some(base) = &window.base {
        parts.push(format!("base={}", base.try_text()?));
    }
    if let Some(bound) = &window.bound {
        parts.push(format!("bound={}", bound.try_text()?));
    }
    if window.partition {
        parts.push("partition=true".into());
    }
    Ok(format!(" window({})", parts.join(", ")))
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
            let step = match &s.step {
                NumOr::Num(n) => spell_number(n.as_f64()),
                NumOr::Value(e) => e.try_text()?,
            };
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
    let mut out = stmt.kind_name().to_string();
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
    for here in body.of_kinds(&["Call"]) {
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
                        "two subroutines are named {:?}: a body's text defines a name once",
                        &*first.name
                    )));
                }
            }
        }
    }
    Ok(())
}

// --- the reader -------------------------------------------------------------------------------------

type Row<'a> = (usize, usize, &'a str);

/// The body `text` spells: `body_text`'s inverse. An error names its line.
pub fn parse_body(text: &str) -> Result<Body> {
    let rows: Vec<Row> = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(number, line)| {
            (
                line.len() - line.trim_start_matches(' ').len(),
                number + 1,
                line.trim(),
            )
        })
        .collect();
    let mut subs = HashMap::new();
    block(&rows, &mut subs)
}

/// The statement `text` spells, with any subroutine its calls name defined ahead of it in `text`.
pub fn parse_statement(text: &str) -> Result<Arc<Statement>> {
    let body = parse_body(text)?;
    match body.stmts() {
        [stmt] => Ok(stmt.clone()),
        _ => Err(Error::Value(format!(
            "expected one statement, got {}",
            body.len()
        ))),
    }
}

/// `rows` split after the leading ones indented deeper than `depth`.
fn under<'a, 'b>(rows: &'b [Row<'a>], depth: usize) -> (&'b [Row<'a>], &'b [Row<'a>]) {
    let cut = rows
        .iter()
        .position(|row| row.0 <= depth)
        .unwrap_or(rows.len());
    rows.split_at(cut)
}

fn at_line(number: usize, err: Error) -> Error {
    match err {
        Error::Value(message) if message.starts_with("line ") => Error::Value(message),
        Error::Value(message) => Error::Value(format!("line {number}: {message}")),
        other => other,
    }
}

fn block(mut rows: &[Row], subs: &mut HashMap<Name, Arc<Subroutine>>) -> Result<Body> {
    let mut out = Vec::new();
    let depth = rows.first().map_or(0, |row| row.0);
    while let Some(((indent, number, text), rest)) = rows.split_first() {
        let (nested, mut rest) = under(rest, depth);
        let mut orelse: &[Row] = &[];
        if text.starts_with("if ")
            && rest
                .first()
                .is_some_and(|row| row.2.replace(' ', "") == "else:")
        {
            (orelse, rest) = under(&rest[1..], depth);
        }
        rows = rest;
        let result = (|| -> Result<()> {
            if *indent != depth {
                return Err(Error::Value("unexpected indentation".into()));
            }
            if text.starts_with("sub ") {
                let sub = Line::new(text)?.subroutine(nested, subs)?;
                subs.insert(sub.name.clone(), sub);
            } else {
                out.push(Line::new(text)?.statement(nested, orelse, subs)?);
            }
            Ok(())
        })();
        result.map_err(|err| at_line(*number, err))?;
    }
    Ok(Body::new(out))
}

/// One line of a body's text.
struct Line {
    parser: Parser,
}

const LINE_PUNCTUATION: [&str; 6] = ["<-", "->", "+=", "=", "..", "~"];

impl Line {
    fn new(text: &str) -> Result<Line> {
        Ok(Line {
            parser: Parser::new(text, &LINE_PUNCTUATION)?,
        })
    }

    fn word(&mut self, word: &str) -> bool {
        if self.parser.at_word(word, 0) {
            self.parser.index += 1;
            return true;
        }
        false
    }

    fn name(&mut self) -> Result<Name> {
        let token = self.parser.take()?;
        if !matches!(token.kind, TokenKind::Name | TokenKind::Quoted) {
            self.parser.index -= 1;
            return Err(self.parser.error("expected a name"));
        }
        Ok(token.text.into())
    }

    /// Names separated by commas, up to `close` when one is given.
    fn names(&mut self, close: Option<&str>) -> Result<Vec<Name>> {
        if let Some(close) = close
            && self.parser.accept(close)
        {
            return Ok(vec![]);
        }
        let mut out = vec![self.name()?];
        while self.parser.accept(",") {
            out.push(self.name()?);
        }
        if let Some(close) = close {
            self.parser.require(close)?;
        }
        Ok(out)
    }

    fn value(&mut self) -> Result<f64> {
        match self.parser.number()? {
            Some(lit) => Ok(lit.value.as_f64()),
            None => Err(self.parser.error("expected a number")),
        }
    }

    /// `key=true` / `key=false` when it is next, else `None`.
    fn flag(&mut self, key: &str) -> Result<Option<bool>> {
        if !(self.parser.at_word(key, 0) && self.parser.at("=", 1)) {
            return Ok(None);
        }
        self.parser.index += 2;
        if !(self.parser.at_word("true", 0) || self.parser.at_word("false", 0)) {
            return Err(self.parser.error("expected true or false"));
        }
        Ok(Some(self.parser.take()?.text == "true"))
    }

    fn end<T>(&mut self, value: T) -> Result<T> {
        if !self.parser.done() {
            return Err(self.parser.error("unexpected text"));
        }
        Ok(value)
    }

    fn dim(&mut self) -> Result<Dim> {
        let expression = self.parser.expression()?;
        if self.parser.accept("~") {
            let hint = self.value()? as i64;
            return Ok(Dim::new(expression, Some(hint)));
        }
        Ok(Dim::new(expression, None))
    }

    /// `name in start..extent`.
    fn axis(&mut self) -> Result<(Name, Expression, Dim)> {
        let name = self.name()?;
        if !self.word("in") {
            return Err(self.parser.error("expected 'in'"));
        }
        let start = self.parser.expression()?;
        self.parser.require("..")?;
        let dim = self.dim()?;
        Ok((name, start, dim))
    }

    /// An axis that is no loop's: `name in 0..extent` and its window.
    fn whole_axis(&mut self) -> Result<Arc<Axis>> {
        let (name, start, extent) = self.axis()?;
        if start != Expression::int(0) {
            return Err(self.parser.error("an axis starts at 0"));
        }
        let window = self.window()?;
        Ok(Arc::new(Axis {
            name,
            extent,
            window,
        }))
    }

    fn subroutine(
        &mut self,
        nested: &[Row],
        subs: &mut HashMap<Name, Arc<Subroutine>>,
    ) -> Result<Arc<Subroutine>> {
        self.parser.index += 1;
        let name = self.name()?;
        self.parser.require("(")?;
        let mut axes = Vec::new();
        while !self.parser.accept(")") {
            if !axes.is_empty() {
                self.parser.require(",")?;
            }
            axes.push(self.whole_axis()?);
        }
        self.parser.require("->")?;
        let result = self.name()?;
        self.parser.require(":")?;
        let body = block(nested, subs)?;
        let sub = Subroutine::new(name, axes.into(), body, result)?;
        self.end(sub)
    }

    fn window(&mut self) -> Result<Option<Window>> {
        if !(self.parser.at_word("window", 0) && self.parser.at("(", 1)) {
            return Ok(None);
        }
        self.parser.index += 2;
        let mut window = Window::default();
        let mut any = false;
        while !self.parser.accept(")") {
            if any {
                self.parser.require(",")?;
            }
            any = true;
            let key = self.name()?;
            self.parser.require("=")?;
            match &*key {
                "parent" => window.parent = Some(self.whole_axis()?),
                "base" => window.base = Some(self.parser.expression()?),
                "bound" => window.bound = Some(self.parser.expression()?),
                "partition"
                    if self.parser.at_word("true", 0) || self.parser.at_word("false", 0) =>
                {
                    window.partition = self.parser.take()?.text == "true";
                }
                other => {
                    return Err(self.parser.error(&format!(
                        "no window part {}",
                        crate::expr::python_repr(other)
                    )));
                }
            }
        }
        Ok(Some(window))
    }

    /// The statement this line spells, `nested` the rows of its block (`orelse` those of an `else`).
    fn statement(
        &mut self,
        nested: &[Row],
        orelse: &[Row],
        subs: &mut HashMap<Name, Arc<Subroutine>>,
    ) -> Result<Arc<Statement>> {
        let ends_block = self
            .parser
            .tokens
            .last()
            .is_some_and(|t| t.kind == TokenKind::Op && t.text == ":");
        if !ends_block && !nested.is_empty() {
            return Err(Error::Value(format!(
                "line {}: unexpected indentation",
                nested[0].1
            )));
        }
        if let Some(token) = self.parser.peek(0)
            && token.kind == TokenKind::Name
            && KIND_NAMES.contains(&token.text.as_str())
            && Statement::fields_of(&token.text).is_some_and(|_| true)
            && (self.parser.at("(", 1) || self.parser.at(":", 1) || self.parser.peek(1).is_none())
            && !is_shared_kind(&token.text)
        {
            let kind = token.text.clone();
            self.parser.index += 1;
            return self.declared(&kind, nested, subs);
        }
        if self.word("for") {
            let body = block(nested, subs)?;
            return self.loop_(body);
        }
        if self.word("if") {
            let cond = self.parser.expression()?;
            self.parser.require(":")?;
            let body = block(nested, subs)?;
            let else_body = block(orelse, subs)?;
            return self.end(Arc::new(Statement::Cond(Cond {
                cond,
                body,
                else_body,
            })));
        }
        if self.word("let") {
            let (dtype, name) = self.typed_name()?;
            self.parser.require("=")?;
            let value = self.parser.expression()?;
            return self.end(Arc::new(Statement::Let(Let { name, value, dtype })));
        }
        if self.word("init") {
            let (dtype, name) = self.typed_name()?;
            self.parser.require("=")?;
            let Some(dtype) = dtype else {
                return Err(self.parser.error("an init names its dtype"));
            };
            let identity = Float(self.value()?);
            return self.end(Arc::new(Statement::Init(Init {
                name,
                identity,
                dtype,
            })));
        }
        if self.word("zero") {
            let dst = self.name()?;
            if !(self.word("words") && self.parser.accept("=")) {
                return Err(self.parser.error("expected words=N"));
            }
            let words = self.value()? as i64;
            return self.end(Arc::new(Statement::ZeroPrologue(ZeroPrologue {
                dst,
                words,
            })));
        }
        let (dtype, name) = self.typed_name()?;
        if self.parser.accept("[") {
            let index = self.parser.expressions("]")?;
            let stmt = self.store(dtype, name, index.into())?;
            return self.end(Arc::new(stmt));
        }
        if self.parser.accept("<-") {
            let op = Op::new(&self.name()?);
            self.parser.require("(")?;
            let operands = self.names(Some(")"))?;
            let [base, value] = &operands[..] else {
                return Err(self.parser.error("an accumulation has two operands"));
            };
            let axes: Vec<Name> = if self.word("over") {
                self.names(None)?
            } else {
                vec![]
            };
            let base = if *base == name {
                None
            } else {
                Some(base.clone())
            };
            let accum = Accum::checked(name, value.clone(), op, dtype, axes.into(), base);
            return self.end(Arc::new(Statement::Accum(accum)));
        }
        let mut defined = vec![name];
        if self.parser.accept(",") {
            defined.extend(self.names(None)?);
        }
        self.parser.require("=")?;
        if self.parser.at_word("load", 0) && !self.parser.at("(", 1) {
            self.parser.index += 1;
            let buffer = self.name()?;
            self.parser.require("[")?;
            let index = self.parser.expressions("]")?;
            let carried = if self.word("carried") && self.parser.accept("=") {
                self.name()?
            } else {
                Name::from("")
            };
            let load = Load::checked(defined.into(), buffer, index.into(), dtype, carried)?;
            return self.end(Arc::new(Statement::Load(load)));
        }
        let [name] = &defined[..] else {
            return Err(self.parser.error("only a load binds several names"));
        };
        let name = name.clone();
        if self.parser.at_word("call", 0) && !self.parser.at("(", 1) {
            self.parser.index += 1;
            let label = self.name()?;
            let Some(target) = subs.get(&label).cloned() else {
                return Err(self.parser.error(&format!(
                    "no subroutine {} is defined above",
                    crate::expr::python_repr(&label)
                )));
            };
            self.parser.require("[")?;
            let buffers = self.names(Some("]"))?;
            self.parser.require("(")?;
            let target = if buffers[..] == target.buffers[..] {
                target
            } else {
                target.with_buffers(buffers.into())
            };
            let args = self.parser.expressions(")")?;
            let call = Call::checked(name, crate::field::SubroutineRef(target), args.into())?;
            return self.end(Arc::new(Statement::Call(call)));
        }
        if self.parser.at_word("pre", 0) && !self.parser.at("(", 1) {
            self.parser.index += 1;
            let carrier = self.name()?;
            self.parser.require("[")?;
            let index = self.parser.expressions("]")?;
            return self.end(Arc::new(Statement::Pre(Pre {
                name,
                carrier,
                index: index.into(),
            })));
        }
        if self.parser.at_word("select", 0) && self.parser.at(":", 1) {
            self.parser.index += 2;
            let mut branches = Vec::new();
            for (_, number, text) in nested {
                branches.push(
                    Line::new(text)
                        .and_then(|mut line| line.branch())
                        .map_err(|err| at_line(*number, err))?,
                );
            }
            let select = Select::checked(name, branches.into())?;
            return self.end(Arc::new(Statement::Select(select)));
        }
        if let Some(token) = self.parser.peek(0)
            && token.kind == TokenKind::Name
            && self.parser.at("(", 1)
        {
            let op = Op::new(&token.text);
            self.parser.index += 2;
            let args = self.names(Some(")"))?;
            return self.end(Arc::new(Statement::Assign(Assign {
                name,
                op,
                args: args.into(),
                dtype,
            })));
        }
        let affine = self.affine(name, dtype)?;
        self.end(Arc::new(Statement::Affine(affine)))
    }

    /// A kind of another layer: its header's fields, then its blocks among the rows under it.
    fn declared(
        &mut self,
        kind: &str,
        nested: &[Row],
        subs: &mut HashMap<Name, Arc<Subroutine>>,
    ) -> Result<Arc<Statement>> {
        let Some(fields) = Statement::fields_of(kind) else {
            return Err(self.parser.error(&format!("no statement kind {kind}")));
        };
        let mut values = self.record(kind, fields)?;
        self.parser.accept(":");
        self.end(())?;
        let mut body: Vec<Row> = Vec::new();
        let mut rest = nested;
        while let Some((row, after)) = rest.split_first() {
            let (rows, after) = under(after, row.0);
            rest = after;
            let label = row
                .2
                .strip_suffix(':')
                .filter(|label| fields.iter().any(|f| f.name == *label && f.is_block()));
            match label {
                None => {
                    body.push(*row);
                    body.extend(rows.iter().copied());
                }
                Some(label) => {
                    let stmts = block(rows, subs)?;
                    values
                        .set(label, Value::Stmts(stmts.stmts().to_vec()))
                        .map_err(|err| at_line(row.1, err))?;
                }
            }
        }
        if !body.is_empty() {
            if !fields.iter().any(|f| f.name == "body") {
                return Err(Error::Value(format!(
                    "line {}: a {kind} holds no body",
                    body[0].1
                )));
            }
            values.set("body", Value::Body(block(&body, subs)?))?;
        }
        let stmt = Statement::from_values(kind, values)?;
        Ok(Arc::new(finished(stmt)?))
    }

    /// The `(field=value, …)` of a declared kind or record, each value read by its field's type.
    fn record(&mut self, kind: &str, fields: &[FieldInfo]) -> Result<Values> {
        let mut values = Values::default();
        if self.parser.accept("(") {
            let mut any = false;
            while !self.parser.accept(")") {
                if any {
                    self.parser.require(",")?;
                }
                any = true;
                let key = self.name()?;
                let field = fields
                    .iter()
                    .find(|f| f.name == &*key)
                    .filter(|f| !f.is_block());
                let Some(field) = field else {
                    return Err(self.parser.error(&format!(
                        "a {kind} has no field {} on its header",
                        crate::expr::python_repr(&key)
                    )));
                };
                self.parser.require("=")?;
                let Some(ty) = field_type(kind, field.name) else {
                    return Err(self
                        .parser
                        .error(&format!("{kind} has no field {}", field.name)));
                };
                let value = self.typed(&ty)?;
                values.set(field.name, value)?;
            }
        }
        Ok(values)
    }

    /// A value of the declared type `ty`.
    fn typed(&mut self, ty: &Ty) -> Result<Value> {
        match ty {
            Ty::Opt(inner) => {
                if self.word("none") {
                    return Ok(Value::None);
                }
                self.typed(inner)
            }
            Ty::NumOr(inner) => match self.parser.number()? {
                Some(lit) => Ok(Value::Number(lit.value)),
                None => self.typed(inner),
            },
            Ty::Seq(inner) => {
                self.parser.require("(")?;
                let mut out = Vec::new();
                while !self.parser.accept(")") {
                    if !out.is_empty() {
                        self.parser.require(",")?;
                    }
                    out.push(self.typed(inner)?);
                }
                Ok(Value::Tuple(out))
            }
            Ty::Tuple(items) => {
                self.parser.require("(")?;
                let mut out = Vec::new();
                for (position, item) in items.iter().enumerate() {
                    if position > 0 {
                        self.parser.require(",")?;
                    }
                    out.push(self.typed(item)?);
                }
                self.parser.require(")")?;
                Ok(Value::Tuple(out))
            }
            Ty::Bool => {
                if !(self.parser.at_word("true", 0) || self.parser.at_word("false", 0)) {
                    return Err(self.parser.error("expected true or false"));
                }
                Ok(Value::Bool(self.parser.take()?.text == "true"))
            }
            Ty::Int => Ok(Value::Number(Num::Int(self.value()? as i64))),
            Ty::Float => Ok(Value::Number(Num::Float(self.value()?))),
            Ty::Name | Ty::DType | Ty::Op => Ok(Value::Name(self.name()?)),
            Ty::Expr => Ok(Value::Expr(self.parser.expression()?)),
            Ty::Axis => Ok(Value::Axis(self.whole_axis()?)),
            Ty::Record(record) => {
                if &*self.name()? != *record {
                    return Err(self.parser.error(&format!("expected a {record}")));
                }
                let Some(fields) = record_fields(record) else {
                    return Err(self.parser.error(&format!("no record {record}")));
                };
                let values = self.record(record, fields)?;
                Ok(Value::Record(record.to_string(), values))
            }
            Ty::Block | Ty::Subroutine => Err(self
                .parser
                .error(&format!("a field of type {ty:?} is not read on a header"))),
        }
    }

    /// `value when predicate`, one branch of a select.
    fn branch(&mut self) -> Result<SelectBranch> {
        let value = self.name()?;
        if !self.word("when") {
            return Err(self.parser.error("expected 'when'"));
        }
        let select = self.parser.expression()?;
        self.end(SelectBranch { value, select })
    }

    /// `[dtype] name`: two names in a row are a dtype and the name it types.
    fn typed_name(&mut self) -> Result<(Option<DType>, Name)> {
        let first = self.parser.peek(0).cloned();
        let following = self.parser.peek(1).cloned();
        if let (Some(first), Some(following)) = (first, following)
            && first.kind == TokenKind::Name
            && matches!(following.kind, TokenKind::Name | TokenKind::Quoted)
        {
            self.parser.index += 1;
            let name = self.name()?;
            return Ok((Some(DType(first.text.into())), name));
        }
        Ok((None, self.name()?))
    }

    /// What follows `name[index]`: a carried state's next value, or a write.
    fn store(
        &mut self,
        dtype: Option<DType>,
        name: Name,
        index: crate::field::Exprs,
    ) -> Result<Statement> {
        if self.parser.accept("<-") {
            let value = self.name()?;
            if !self.word("seed") {
                return Err(self.parser.error("expected 'seed'"));
            }
            let seed = match self.parser.number()? {
                Some(lit) => NumOr::Num(Num::Float(lit.value.as_f64())),
                None => NumOr::Value(self.name()?),
            };
            return Ok(Statement::Carry(Carry {
                name,
                value,
                index,
                seed,
                dtype,
            }));
        }
        let atomic = self.parser.accept("+=");
        if !atomic {
            self.parser.require("=")?;
        }
        let values = if self.parser.accept("(") {
            self.names(Some(")"))?
        } else {
            vec![self.name()?]
        };
        let swizzle = if self.word("swizzle") && self.parser.accept("=") {
            self.name()?
        } else {
            Name::from("NONE")
        };
        Ok(Statement::Write(Write::checked(
            name,
            index,
            values.into(),
            dtype,
            atomic,
            swizzle,
        )?))
    }

    fn affine(&mut self, name: Name, dtype: Option<DType>) -> Result<Affine> {
        let mut args: Vec<Name> = Vec::new();
        let mut coefficients: Vec<Float> = Vec::new();
        let mut offset = 0.0;
        while !self.parser.done() {
            let negative = self.parser.accept("-");
            if !negative && (!args.is_empty() || offset != 0.0) && !self.parser.accept("+") {
                return Err(self.parser.error("expected '+' or '-'"));
            }
            let sign = if negative { -1.0 } else { 1.0 };
            match self.parser.number()? {
                Some(literal) if !self.parser.accept("*") => {
                    offset = sign * literal.value.as_f64();
                    break;
                }
                literal => {
                    args.push(self.name()?);
                    coefficients.push(Float(
                        sign * literal.map_or(1.0, |lit: Lit| lit.value.as_f64()),
                    ));
                }
            }
        }
        Affine::checked(name, args.into(), coefficients.into(), Float(offset), dtype)
    }

    fn loop_(&mut self, body: Body) -> Result<Arc<Statement>> {
        let (name, start, extent) = self.axis()?;
        let step = if self.word("step") {
            Some(self.parser.expression()?)
        } else {
            None
        };
        let end = if self.word("until") {
            Some(self.parser.expression()?)
        } else {
            None
        };
        let window = self.window()?;
        let axis = Arc::new(Axis {
            name,
            extent,
            window,
        });
        let unroll = self.flag("unroll")?.unwrap_or(false);
        let seed = self.flag("seed")?.unwrap_or(true);
        self.parser.require(":")?;
        if let Some(step) = step {
            let loop_ = StridedLoop {
                axis,
                start,
                step: NumOr::Value(step),
                body,
                unroll,
                end,
                seed,
            };
            return self.end(Arc::new(Statement::StridedLoop(loop_)));
        }
        if start != Expression::int(0) || end.is_some() {
            return Err(self
                .parser
                .error("a loop that does not start at 0, or stops early, names its step"));
        }
        self.end(Arc::new(Statement::Loop(Loop {
            axis,
            body,
            unroll,
            seed,
        })))
    }
}

fn is_shared_kind(name: &str) -> bool {
    matches!(
        name,
        "Load"
            | "Assign"
            | "Affine"
            | "Accum"
            | "Carry"
            | "Pre"
            | "Init"
            | "Let"
            | "Write"
            | "Select"
            | "ZeroPrologue"
            | "Loop"
            | "StridedLoop"
            | "Cond"
            | "Call"
    )
}

fn field_type(kind: &str, field: &str) -> Option<Ty> {
    Statement::field_ty(kind, field).or_else(|| match kind {
        SelectBranch::NAME => SelectBranch::field_ty(field),
        FragmentArg::NAME => FragmentArg::field_ty(field),
        _ => None,
    })
}

fn record_fields(record: &str) -> Option<&'static [FieldInfo]> {
    match record {
        SelectBranch::NAME => Some(SelectBranch::FIELDS),
        FragmentArg::NAME => Some(FragmentArg::FIELDS),
        _ => None,
    }
}

/// A literal number's expression text as a statement spells it (for the Python side's `spell_number`).
pub fn number_text(value: f64) -> String {
    spell_number(value)
}

/// A literal expression, for a `Let` built from a bare number.
pub fn float_literal(value: f64) -> Expression {
    Expression::new(Expr::Literal(Lit {
        value: Num::Float(value),
        dtype: "float".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATEMENTS: &[&str] = &[
        "in0 = load x[a0, a1 + 1]",
        "f8e4m3 in0, in1 = load x[a0]",
        "in0 = load x[a0] carried=unpack",
        "in0 = load `model.layers.0.weight`[a0]",
        "p = pre state[a0, 0]",
        "v1 = multiply(in0, in1)",
        "f16 v1 = exp(in0)",
        "`load` = copy(`for`)",
        "v2 = x",
        "v2 = -x + 0.5*y",
        "f32 v2 = 2*x - y + 1e-06",
        "v2 = 3*x - 4",
        "let v0 = 0.125",
        "let f32 i = a0 * 4 + a1",
        "init f32 acc = -inf",
        "acc0 <- add(acc0, v1)",
        "f32 acc0 <- maximum(acc0, v1) over a1, a2",
        "acc0 <- add(scaled, v1)",
        "state[a0, a1] <- v seed 0",
        "f32 state[a0] <- v seed state_in",
        "y[a0, a1] = v",
        "f16 y[a0] = (v0, v1)",
        "y[a0] += v",
        "slab[a0] = v swizzle=B32",
        "zero acc words=1024",
        "v3 = select:\n    x when a0 < 1\n    y when a0 >= 1",
        "for a0 in 0..4:\n    y[a0] = v",
        "for a0 in 0..seq~128 unroll=true seed=false:",
        "for a0 in 0..seq * 2~64:",
        "for kb in 0..16 window(parent=ko in 0..32 window(parent=k in 0..64), partition=true):",
        "for kb in 0..16 window(base=a0, bound=a0 + 1):",
        "for kb in 0..16 window():",
        "for k in t..64 step 32:\n    y[k] = v",
        "for k in 0..64 step 8 until k_end unroll=true:",
        "if a0 < 1:\n    y[a0] = v\nelse:\n    y[a0] = w",
        "if a0 < 1:",
        "SetMaxNReg(count=232, direction=inc)",
        "Smem(name=s_x, extents=(64, 64), dtype=f16)",
        "Tile(axes=(a0 in 0..64, a1 in 0..64), block_threads=128):\n    RegStore(dst_buffer=y, dst_index=(a0, a1), frag=c0, shape=(16, 8, 8)):\n        epilogue:\n            v1 = multiply(v0, s)\n        run:\n            RegStore(dst_buffer=y, dst_index=(a0, a1 + 8), frag=c1, shape=(16, 8, 8))",
        "FragmentApply(out=o, op=where, args=(FragmentArg(kind=coord, expressions=(_vc > _vr)), FragmentArg(kind=gmem, expressions=(i, j), buffer=mask)), row_base=q)",
        "TreeHalve(bufs=(b), state=(acc), state_b=(acc__o), length=8, tid_var=t):\n    combine_states:\n        t = multiply(acc, acc__o)\n        acc = add(acc, t)",
    ];

    const CALLS: &str = "sub inner(r in 0..4) -> s:\n    for k in 0..8:\n        x = load a[r, k]\n        s <- add(s, x) over k\nsub outer(r in 0..4, c in 0..2 window(parent=cc in 0..4)) -> t:\n    u = call inner[a](r)\n    w = load b[c]\n    t = multiply(u, w)\nfor i in 0..4:\n    p = call outer[a, b](i, 0)\n    q = call outer[a, b](i, 1)\n    z = call outer[m, b](i, 1)\n    v = p + q + z\n    y[i] = v";

    #[test]
    fn every_statement_reads_back() {
        for text in STATEMENTS {
            let body = parse_body(text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(body_text(&body).unwrap(), *text);
            assert_eq!(parse_body(&body_text(&body).unwrap()).unwrap(), body);
        }
    }

    #[test]
    fn calls_read_back() {
        let body = parse_body(CALLS).unwrap();
        assert_eq!(body_text(&body).unwrap(), CALLS);
        let calls = body.of_kinds(&["Call"]);
        let targets: Vec<_> = calls
            .iter()
            .map(|c| match &**c.stmt() {
                Statement::Call(call) => call.target.0.clone(),
                _ => unreachable!(),
            })
            .collect();
        assert!(Arc::ptr_eq(&targets[0], &targets[1]) && !Arc::ptr_eq(&targets[0], &targets[2]));
    }

    #[test]
    fn errors_name_their_line() {
        for (text, message) in [
            ("y[a0] = v\n    z[a0] = v", "line 2: unexpected indentation"),
            ("v = 1", "line 1: "),
            ("for a0 in 2..4:", "line 1: "),
            (
                "a, b = multiply(x, y)",
                "line 1: only a load binds several names",
            ),
            ("v = x + ", "line 1: unexpected end"),
            ("y[a0] = v extra", "line 1: unexpected text"),
            (
                "p = call missing[a](i)",
                "line 1: no subroutine 'missing' is defined above",
            ),
        ] {
            let err = parse_body(text)
                .err()
                .unwrap_or_else(|| panic!("{text} parsed"));
            assert!(err.to_string().starts_with(message), "{text}: {err}");
        }
    }
}
