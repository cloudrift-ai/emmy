//! The reader: the lines of a body back to the body, an error naming its line.

use std::collections::HashMap;
use std::sync::Arc;

use crate::body::{Body, Subroutine};
use crate::expr::{Expression, Lit, Num, Parser, TokenKind};
use crate::kinds::axis::{Axis, Dim, Window};
use crate::kinds::elementwise::Op;
use crate::kinds::field::{DType, Float, NumOr, Ty, Value, Values};
use crate::kinds::*;
use crate::protocol::finished;
use crate::{Error, Name, Result};

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
            && let Ok(kind) = token.text.parse::<Kind>()
            && kind.layer() != Layer::Shared
            && (self.parser.at("(", 1) || self.parser.at(":", 1) || self.parser.peek(1).is_none())
        {
            self.parser.index += 1;
            return self.declared(kind, nested, subs);
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
            let accum = Accum {
                name,
                value: value.clone(),
                op,
                dtype,
                axes: axes.into(),
                base,
            }
            .normalized();
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
            let load = Load {
                names: defined.into(),
                input: buffer,
                index: index.into(),
                dtype,
                carried,
            };
            load.validate()?;
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
            let call = Call {
                name,
                target: crate::kinds::field::SubroutineRef(target),
                args: args.into(),
            };
            call.validate()?;
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
            let select = Select {
                name,
                branches: branches.into(),
            };
            select.validate()?;
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
        kind: Kind,
        nested: &[Row],
        subs: &mut HashMap<Name, Arc<Subroutine>>,
    ) -> Result<Arc<Statement>> {
        let fields = kind.fields();
        let mut values = self.record(kind.name(), fields, &|field| kind.field_ty(field))?;
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
        let stmt = kind.from_values(values)?;
        Ok(Arc::new(finished(stmt)?))
    }

    /// The `(field=value, …)` of a declared kind or record, each value read by its field's type.
    fn record(
        &mut self,
        kind: &str,
        fields: &[FieldInfo],
        ty_of: &dyn Fn(&str) -> Option<Ty>,
    ) -> Result<Values> {
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
                let Some(ty) = ty_of(field.name) else {
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
                let values =
                    self.record(record, fields, &|field| record_field_ty(record, field))?;
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
        index: crate::kinds::field::Exprs,
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
        let write = Write {
            output: name,
            index,
            values: values.into(),
            value_dtype: dtype,
            atomic,
            swizzle,
        };
        write.validate()?;
        Ok(Statement::Write(write))
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
        let affine = Affine {
            name,
            args: args.into(),
            coefficients: coefficients.into(),
            offset: Float(offset),
            dtype,
        };
        affine.validate()?;
        Ok(affine)
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
                step,
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

/// The type a record's field is read by.
fn record_field_ty(record: &str, field: &str) -> Option<Ty> {
    match record {
        SelectBranch::NAME => SelectBranch::field_ty(field),
        FragmentArg::NAME => FragmentArg::field_ty(field),
        _ => None,
    }
}

fn record_fields(record: &str) -> Option<&'static [FieldInfo]> {
    match record {
        SelectBranch::NAME => Some(SelectBranch::FIELDS),
        FragmentArg::NAME => Some(FragmentArg::FIELDS),
        _ => None,
    }
}
