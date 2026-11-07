//! The reader: the tokenizer and the parser that take an expression's text back, and that the body text's reader
//! extends with the statement punctuation.

use super::{BinOp, Expr, Expression, Lit, LitType, Num, is_builtin, python_repr};
use crate::{Error, Name, Result};

// --- the reader -----------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenKind {
    Number,
    Name,
    Quoted,
    Op,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub position: usize,
}

/// Operators and punctuation an expression's text uses.
pub const PUNCTUATION: [&str; 26] = [
    "||", "&&", "==", "!=", "<=", ">=", "<<", ">>", "//", "-", "+", "*", "/", "%", "<", ">", "^",
    "&", "|", "?", ":", "(", ")", ",", "[", "]",
];

/// The reader of an expression's text. Text that holds expressions among other things (a statement) is read by a
/// parser built with more punctuation, which takes each expression with [`Parser::expression`].
pub struct Parser {
    pub text: String,
    pub tokens: Vec<Token>,
    pub index: usize,
    depth: usize,
}

/// How deep an expression may nest before the reader refuses it, so a pathological text cannot exhaust the stack.
pub const MAX_NESTING: usize = 256;

impl Parser {
    /// Tokenize `text` with the expression punctuation plus `extra` (longest first where one starts another).
    pub fn new(text: &str, extra: &[&str]) -> Result<Parser> {
        let mut punctuation: Vec<&str> = PUNCTUATION.to_vec();
        punctuation.extend(extra.iter().copied());
        punctuation.sort_by_key(|p| std::cmp::Reverse(p.len()));
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut position = 0;
        while position < bytes.len() {
            let Some(c) = text[position..].chars().next() else {
                break;
            };
            if c.is_whitespace() {
                position += c.len_utf8();
                continue;
            }
            if c.is_ascii_digit() {
                let start = position;
                let mut end = position;
                let rest = &text[position..];
                let digits = |s: &str| s.chars().take_while(|c| c.is_ascii_digit()).count();
                end += digits(rest);
                // A number needs a digit after its point, so ``0..4`` is a number, two dots and a number.
                if text[end..].starts_with('.')
                    && text[end + 1..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit())
                {
                    end += 1 + digits(&text[end + 1..]);
                }
                if let Some(e) = text[end..]
                    .chars()
                    .next()
                    .filter(|c| *c == 'e' || *c == 'E')
                {
                    let mut probe = end + e.len_utf8();
                    if text[probe..].starts_with('+') || text[probe..].starts_with('-') {
                        probe += 1;
                    }
                    let count = digits(&text[probe..]);
                    if count > 0 {
                        end = probe + count;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Number,
                    text: text[start..end].to_string(),
                    position: start,
                });
                position = end;
                continue;
            }
            if c.is_alphabetic() || c == '_' {
                let start = position;
                let word = |s: &str| {
                    s.chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .map(char::len_utf8)
                        .sum::<usize>()
                };
                let mut end = position + word(&text[position..]);
                if text[end..].starts_with('.')
                    && text[end + 1..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                {
                    end += 1 + word(&text[end + 1..]);
                }
                tokens.push(Token {
                    kind: TokenKind::Name,
                    text: text[start..end].to_string(),
                    position: start,
                });
                position = end;
                continue;
            }
            if c == '`' {
                let start = position;
                let inner = &text[position + 1..];
                let Some(close) = inner.find('`') else {
                    return Err(Parser::error_at("unexpected character", start, text));
                };
                let quoted = &inner[..close];
                if quoted.is_empty() || quoted.contains('\n') {
                    return Err(Parser::error_at("unexpected character", start, text));
                }
                tokens.push(Token {
                    kind: TokenKind::Quoted,
                    text: quoted.to_string(),
                    position: start + 1,
                });
                position += close + 2;
                continue;
            }
            match punctuation
                .iter()
                .find(|p| text[position..].starts_with(**p))
            {
                Some(p) => {
                    tokens.push(Token {
                        kind: TokenKind::Op,
                        text: p.to_string(),
                        position,
                    });
                    position += p.len();
                }
                None => return Err(Parser::error_at("unexpected character", position, text)),
            }
        }
        Ok(Parser {
            text: text.to_string(),
            tokens,
            index: 0,
            depth: 0,
        })
    }

    fn error_at(message: &str, position: usize, text: &str) -> Error {
        Error::Value(format!(
            "{message} at position {position} of {}",
            python_repr(text)
        ))
    }

    pub fn error(&self, message: &str) -> Error {
        let position = self
            .tokens
            .get(self.index)
            .map_or(self.text.len(), |t| t.position);
        Parser::error_at(message, position, &self.text)
    }

    pub fn peek(&self, ahead: usize) -> Option<&Token> {
        self.tokens.get(self.index + ahead)
    }

    pub fn take(&mut self) -> Result<Token> {
        let token = self
            .peek(0)
            .cloned()
            .ok_or_else(|| self.error("unexpected end"))?;
        self.index += 1;
        Ok(token)
    }

    pub fn at(&self, op: &str, ahead: usize) -> bool {
        self.peek(ahead)
            .is_some_and(|t| t.kind == TokenKind::Op && t.text == op)
    }

    pub fn at_word(&self, word: &str, ahead: usize) -> bool {
        self.peek(ahead)
            .is_some_and(|t| t.kind == TokenKind::Name && t.text == word)
    }

    pub fn accept(&mut self, op: &str) -> bool {
        if self.at(op, 0) {
            self.index += 1;
            return true;
        }
        false
    }

    pub fn require(&mut self, op: &str) -> Result<()> {
        if !self.accept(op) {
            return Err(self.error(&format!("expected {}", python_repr(op))));
        }
        Ok(())
    }

    pub fn done(&self) -> bool {
        self.index >= self.tokens.len()
    }

    pub fn expression(&mut self) -> Result<Expression> {
        if self.depth >= MAX_NESTING {
            return Err(self.error("the expression nests too deep"));
        }
        self.depth += 1;
        let result = self.nested_expression();
        self.depth -= 1;
        result
    }

    fn nested_expression(&mut self) -> Result<Expression> {
        let cond = self.binary(1)?;
        if !self.accept("?") {
            return Ok(cond);
        }
        let if_true = self.expression()?;
        self.require(":")?;
        let if_false = self.expression()?;
        Ok(Expression::new(Expr::Ternary(cond, if_true, if_false)))
    }

    /// A comma-separated list of expressions up to `close`, which is consumed.
    pub fn expressions(&mut self, close: &str) -> Result<Vec<Expression>> {
        let mut items = Vec::new();
        while !self.accept(close) {
            if !items.is_empty() {
                self.require(",")?;
            }
            items.push(self.expression()?);
        }
        Ok(items)
    }

    fn binary(&mut self, least: u8) -> Result<Expression> {
        let mut left = self.operand()?;
        while let Some(token) = self.peek(0) {
            if token.kind != TokenKind::Op {
                break;
            }
            let Some(op) = BinOp::parse(&token.text) else {
                break;
            };
            if op.precedence() < least {
                break;
            }
            self.index += 1;
            let right = self.binary(op.precedence() + 1)?;
            left = Expression::binary(op, left, right);
        }
        Ok(left)
    }

    /// The number next in the text, signed or not, or `None` when what is next is no number.
    pub fn number(&mut self) -> Result<Option<Lit>> {
        let sign = self.at("-", 0);
        let ahead = sign as usize;
        let Some(token) = self.peek(ahead) else {
            return Ok(None);
        };
        let literal = match token.kind {
            TokenKind::Number => {
                let text = format!("{}{}", if sign { "-" } else { "" }, token.text);
                if token.text.contains(['.', 'e', 'E']) {
                    let value: f64 = text.parse().map_err(|_| self.error("not a number"))?;
                    Lit {
                        value: Num::Float(value),
                        dtype: LitType::Float,
                    }
                } else {
                    let value: i64 = text
                        .parse()
                        .map_err(|_| self.error("integer out of range"))?;
                    Lit {
                        value: Num::Int(value),
                        dtype: LitType::Int,
                    }
                }
            }
            TokenKind::Name if token.text == "inf" || token.text == "nan" => {
                let value = if token.text == "inf" {
                    f64::INFINITY
                } else {
                    f64::NAN
                };
                Lit {
                    value: Num::Float(if sign { -value } else { value }),
                    dtype: LitType::Float,
                }
            }
            _ => return Ok(None),
        };
        self.index += ahead + 1;
        Ok(Some(literal))
    }

    fn operand(&mut self) -> Result<Expression> {
        if let Some(literal) = self.number()? {
            return Ok(Expression::new(Expr::Literal(literal)));
        }
        let token = self.take()?;
        match token.kind {
            TokenKind::Quoted => {
                if self.accept("[") {
                    let index = self.expressions("]")?;
                    return Ok(Expression::new(Expr::FlatIndex(
                        token.text.into(),
                        index.into(),
                    )));
                }
                Ok(Expression::new(Expr::Var(token.text.into())))
            }
            TokenKind::Name => {
                if token.text == "true" || token.text == "false" {
                    return Ok(Expression::literal(
                        Num::Bool(token.text == "true"),
                        LitType::Bool,
                    ));
                }
                if self.accept("(") {
                    let args = self.expressions(")")?;
                    return Ok(Expression::new(Expr::Call(token.text.into(), args.into())));
                }
                if self.accept("[") {
                    let index = self.expressions("]")?;
                    return Ok(Expression::new(Expr::FlatIndex(
                        token.text.into(),
                        index.into(),
                    )));
                }
                if is_builtin(&token.text) {
                    return Ok(Expression::new(Expr::Builtin(token.text.into())));
                }
                Ok(Expression::new(Expr::Var(token.text.into())))
            }
            TokenKind::Op if token.text == "(" => {
                // ``(dtype)`` followed by an operand is a cast; the writer never parenthesizes a lone name otherwise.
                if let Some(next) = self.peek(0)
                    && next.kind == TokenKind::Name
                    && self.at(")", 1)
                    && self.starts_operand(2)
                {
                    let dtype: Name = next.text.clone().into();
                    self.index += 2;
                    let inner = self.operand()?;
                    return Ok(Expression::new(Expr::Cast(dtype, inner)));
                }
                let inner = self.expression()?;
                self.require(")")?;
                Ok(inner)
            }
            _ => {
                self.index -= 1;
                Err(self.error("expected an operand"))
            }
        }
    }

    fn starts_operand(&self, ahead: usize) -> bool {
        self.peek(ahead).is_some_and(|t| {
            matches!(
                t.kind,
                TokenKind::Number | TokenKind::Name | TokenKind::Quoted
            ) || t.text == "("
        })
    }
}

/// The expression `text` spells: the inverse of [`Expression::text`].
pub fn parse(text: &str) -> Result<Expression> {
    let mut parser = Parser::new(text, &[])?;
    let expression = parser.expression()?;
    if !parser.done() {
        return Err(parser.error("unexpected text"));
    }
    Ok(expression)
}
