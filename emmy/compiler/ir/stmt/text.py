"""The text of a body: one printer and one reader, exact inverses.

``str(body)`` is the body's text and :func:`parse_body` reads it back to an equal body, so the text says everything
a statement holds: a field at its default is left out, nothing else is. One statement is one line; a block's
statements are indented four spaces under its header::

    let f32 v0 = 0.125
    for a0 in 0..4:
        for a1 in 0..seq~128:
            in0 = load x[a0, a1]
            v1 = multiply(in0, v0)
            v2 = 2*v1 - in0 + 1
            acc0 <- add(acc0, v2) over a1
        v3 = select:
            acc0 when a0 < 2
            v0 when a0 >= 2
        y[a0] = v3

The statements, with ``[dtype]`` an optional element type written first (``f32 v1 = …``):

==================  ================================================================================
``Load``            ``[dtype] n0, n1 = load buf[i, j] [carried=load|unpack]``
``Pre``             ``n = pre state[i, j]``
``Assign``          ``[dtype] n = op(a, b)``
``Affine``          ``[dtype] n = 2*a - b + 1`` — coefficient ``*`` name per term, the offset last
``Let``             ``let [dtype] n = <expression>``
``Init``            ``init dtype n = <number>``
``Accum``           ``[dtype] n <- op(n, value) [over a, b]`` — the first operand is the base when it is not ``n``
``Carry``           ``[dtype] state[i, j] <- value seed <number or buffer>``
``Write``           ``[dtype] buf[i, j] = v`` / ``= (v0, v1)`` / ``+= v`` (atomic) ``[swizzle=MODE]``
``Select``          ``n = select:`` over one ``value when <predicate>`` line per branch
``ZeroPrologue``    ``zero buf words=N``
``Loop``            ``for <axis> [unroll=true] [seed=false]:``
``StridedLoop``     ``for a in <start>..<extent> step <step> [until <end>] [window(…)] [unroll=true] [seed=false]:``
``Cond``            ``if <predicate>:`` and an optional ``else:``
``Call``            ``n = call sub[buf0, buf1](<expression>, …)`` — the buffers the call binds the subroutine's to
==================  ================================================================================

A subroutine a call names is defined once, ahead of the statements: ``sub name(<axis>, …) -> result:`` over its body.
Subroutines compare by object, so what reading a text back keeps is which calls share one, not the object.

An axis is ``name in 0..extent``; a symbolic extent's hint follows it as ``seq~128`` when it is not the default, and a
window as ``window(parent=<axis>, base=<expression>, bound=<expression>, partition=true)``, each part only when set. Expressions
are in their own text (:mod:`emmy.compiler.ir.expression`), and a name that is no identifier or that reads as a keyword is
written between backticks.

A kind of another layer (the Kernel IR's) prints and reads through its declaration: its class name, then in
parentheses every field that is not at its default as ``field=value`` in declaration order, then a colon and one block
per field that holds statements — the field named ``body`` as the block itself, any other labelled by its name::

    Smem(name=s_x, extents=(64, 64), dtype=f16)
    Tile(axes=(a0 in 0..64, a1 in 0..64), block_threads=128):
        RegStore(dst_buffer=y, dst_index=[a0, a1], frag=c0, shape=(16, 8, 8)):
            epilogue:
                v1 = multiply(v0, s)
            run:
                RegStore(dst_buffer=y, dst_index=[a0, a1 + 8], frag=c1, shape=(16, 8, 8))

A value is spelled by its type: a name, a number, ``true`` / ``false`` / ``none``, a dtype or an elementwise op by its
name, an expression in its own text, an axis as a loop spells it, a tuple in parentheses, a record (a ``FragmentArg``)
as a kind is. The reader takes each value by the field's declared type.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Iterable
from dataclasses import MISSING, fields, is_dataclass
from functools import cache

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType
from emmy.compiler.dtype import get as dtype_of
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Expression, Literal, Parser, spell_name
from emmy.compiler.ir.stmt.base import Statement
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.declared import KINDS
from emmy.compiler.ir.stmt.leaves import Accum, Affine, Assign, Carry, Init, Let, Load, Pre, Select, SelectBranch, Write, ZeroPrologue
from emmy.compiler.ir.stmt.subroutine import Call, Subroutine

INDENT = "    "
#: Words the reader gives a meaning, so a name spelled like one is written between backticks.
_KEYWORDS = frozenset(
    "for in if else let init zero load pre select when over seed step until window carried swizzle words unroll sub call "
    "true false none".split()
)
#: The roles of a field that holds statements, written as a block under a declared kind's header.
_BLOCKS = frozenset({"body", "term", "stmts"})


def _name(name: str) -> str:
    return spell_name(name, _KEYWORDS)


def _names(names: Iterable[str]) -> str:
    return ", ".join(_name(name) for name in names)


def spell_number(value: float) -> str:
    """The one spelling of a number a statement holds: an integer without a point, anything else as Python writes it."""
    value = float(value)
    return str(int(value)) if value.is_integer() and abs(value) < 2**53 else repr(value)


def _typed(dtype: DataType | None) -> str:
    return f"{dtype.name} " if dtype is not None else ""


def _index(index: tuple[Expression, ...]) -> str:
    return f"[{', '.join(str(e) for e in index)}]"


def _axis(axis: Axis, start: str = "0") -> str:
    extent = str(axis.extent.expression)
    if axis.extent.hint != Dim(axis.extent.expression).hint:
        extent += f"~{axis.extent.hint}"
    return f"{_name(axis.name)} in {start}..{extent}"


def _window(window: Window | None) -> str:
    if window is None:
        return ""
    parts = {
        "parent": None if window.parent is None else _axis(window.parent) + _window(window.parent.window),
        "base": window.base,
        "bound": window.bound,
        "partition": "true" if window.partition else None,
    }
    return f" window({', '.join(f'{key}={value}' for key, value in parts.items() if value is not None)})"


def _flags(stmt: Loop | StridedLoop) -> str:
    return (" unroll=true" if stmt.unroll else "") + ("" if stmt.seed else " seed=false") + ":"


def _affine(stmt: Affine) -> str:
    terms = [(c, _name(arg) if abs(c) == 1 else f"{spell_number(abs(c))}*{_name(arg)}") for c, arg in stmt.terms]
    if stmt.offset != 0:
        terms.append((stmt.offset, spell_number(abs(stmt.offset))))
    head, *rest = terms
    return ("-" if head[0] < 0 else "") + head[1] + "".join(f" {'-' if c < 0 else '+'} {term}" for c, term in rest)


def lines(stmt: Statement, indent: str = "") -> list[str]:
    """``stmt``'s text as lines, each starting with ``indent``."""
    inner = indent + INDENT
    match stmt:
        case Load():
            carried = f" carried={stmt.carried}" if stmt.carried else ""
            text = f"{_typed(stmt.dtype)}{_names(stmt.names)} = load {_name(stmt.input)}{_index(stmt.index)}{carried}"
        case Pre():
            text = f"{_name(stmt.name)} = pre {_name(stmt.carrier)}{_index(stmt.index)}"
        case Assign():
            text = f"{_typed(stmt.dtype)}{_name(stmt.name)} = {stmt.op.name}({_names(stmt.args)})"
        case Affine():
            text = f"{_typed(stmt.dtype)}{_name(stmt.name)} = {_affine(stmt)}"
        case Let():
            text = f"let {_typed(stmt.dtype)}{_name(stmt.name)} = {stmt.value}"
        case Init():
            text = f"init {stmt.dtype.name} {_name(stmt.name)} = {spell_number(stmt.identity)}"
        case Accum():
            over = f" over {_names(stmt.axes)}" if stmt.axes else ""
            base = stmt.name if stmt.base is None else stmt.base
            text = f"{_typed(stmt.dtype)}{_name(stmt.name)} <- {stmt.op.name}({_names((base, stmt.value))}){over}"
        case Carry():
            seed = _name(stmt.seed) if isinstance(stmt.seed, str) else spell_number(stmt.seed)
            text = f"{_typed(stmt.dtype)}{_name(stmt.name)}{_index(stmt.index)} <- {_name(stmt.value)} seed {seed}"
        case Write():
            value = f"({_names(stmt.values)})" if stmt.is_vector else _name(stmt.values[0])
            swizzle = f" swizzle={_name(stmt.swizzle)}" if stmt.swizzle != "NONE" else ""
            store = "+=" if stmt.atomic else "="
            text = f"{_typed(stmt.value_dtype)}{_name(stmt.output)}{_index(stmt.index)} {store} {value}{swizzle}"
        case Select():
            return [f"{indent}{_name(stmt.name)} = select:", *(f"{inner}{_name(b.value)} when {b.select}" for b in stmt.branches)]
        case ZeroPrologue():
            text = f"zero {_name(stmt.dst)} words={stmt.words}"
        case Loop():
            return [f"{indent}for {_axis(stmt.axis)}{_window(stmt.axis.window)}{_flags(stmt)}", *pretty_body(stmt.body, inner)]
        case StridedLoop():
            until = f" until {stmt.end}" if stmt.end is not None else ""
            head = f"for {_axis(stmt.axis, str(stmt.start))} step {stmt.step}{until}{_window(stmt.axis.window)}{_flags(stmt)}"
            return [indent + head, *pretty_body(stmt.body, inner)]
        case Cond():
            orelse = [f"{indent}else:", *pretty_body(stmt.else_body, inner)] if stmt.else_body else []
            return [f"{indent}if {stmt.cond}:", *pretty_body(stmt.body, inner), *orelse]
        case Call():
            args = ", ".join(str(arg) for arg in stmt.args)
            text = f"{_name(stmt.name)} = call {_name(stmt.target.name)}[{_names(stmt.target.buffers)}]({args})"
        case _:
            return _declared_lines(stmt, indent)
    return [indent + text]


@cache
def _shape(cls: type) -> tuple[tuple[str, object, object, bool], ...]:
    """A declared class's fields: name, type, default (``MISSING`` when none) and whether the field is a block."""
    hints = typing.get_type_hints(cls)
    out = []
    for f in fields(cls):
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        out.append((f.name, hints[f.name], default, bool(f.metadata.get("roles", frozenset()) & _BLOCKS)))
    return tuple(out)


def _spell(value: object) -> str:
    """A field value's text, by what it is."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return spell_number(value)
    if isinstance(value, str):
        return _name(value)
    if isinstance(value, (DataType, ElementwiseImpl)):
        return _name(value.name)
    if isinstance(value, Expression):
        return str(value)
    if isinstance(value, Axis):
        return _axis(value) + _window(value.window)
    if isinstance(value, tuple):
        return f"({', '.join(_spell(item) for item in value)})"
    if is_dataclass(value):
        return _header(value)
    raise TypeError(f"a {type(value).__name__} has no spelling in a statement's text")


def _header(value: object) -> str:
    """``Kind(field=value, …)`` over the fields of ``value`` that are not at their default and hold no statements."""
    parts = [
        f"{name}={_spell(got)}" for name, _, default, block in _shape(type(value)) if not block and (got := getattr(value, name)) != default
    ]
    return type(value).__name__ + (f"({', '.join(parts)})" if parts else "")


def _declared_lines(stmt: Statement, indent: str) -> list[str]:
    """The lines of a kind that prints through its declaration."""
    inner = indent + INDENT
    blocks = [(name, got) for name, _, default, block in _shape(type(stmt)) if block and (got := getattr(stmt, name)) != default]
    out = [indent + _header(stmt) + (":" if blocks else "")]
    for name, value in blocks:
        if name == "body":
            out.extend(pretty_body(value, inner))
        else:
            out.append(f"{inner}{name}:")
            out.extend(pretty_body(value, inner + INDENT))
    return out


def pretty_body(body: Iterable[Statement], indent: str = "") -> list[str]:
    """The lines of a body's text, each starting with ``indent``."""
    return [line for stmt in body for line in lines(stmt, indent)]


def body_text(body: Body) -> str:
    """A body's whole text: the subroutines its calls name, each once and ahead of its callers, then its statements."""
    return "\n".join([*(line for sub in _subroutines(body, {}).values() for line in _definition(sub)), *pretty_body(body)])


def _definition(sub: Subroutine) -> list[str]:
    axes = ", ".join(_axis(axis) + _window(axis.window) for axis in sub.axes)
    return [f"sub {_name(sub.name)}({axes}) -> {_name(sub.result)}:", *pretty_body(sub.body, INDENT)]


def _subroutines(body: Body, found: dict[str, Subroutine]) -> dict[str, Subroutine]:
    """The subroutines ``body`` calls by name, a callee ahead of its caller. Two calls may name one subroutine bound
    to other buffers; two subroutines of one name are refused."""
    for call in body.iter_of_type(Call):
        _subroutines(call.target.body, found)
        first = found.setdefault(call.target.name, call.target)
        if first is not call.target and _definition(first.with_buffers(call.target.buffers)) != _definition(call.target):
            raise ValueError(f"two subroutines are named {first.name!r}: a body's text defines a name once")
    return found


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def parse_body(text: str) -> Body:
    """The body ``text`` spells — ``str(body)``'s inverse. An error names its line."""
    rows = [(len(line) - len(line.lstrip(" ")), number, line.strip()) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    return _block(rows, {})


def _under(rows: list, depth: int) -> tuple[list, list]:
    """``rows`` split after the leading ones indented deeper than ``depth``."""
    cut = next((index for index, row in enumerate(rows) if row[0] <= depth), len(rows))
    return rows[:cut], rows[cut:]


def _block(rows: list, subs: dict[str, Subroutine]) -> Body:
    """The statements ``rows`` spell: each row at the first one's depth, with the deeper rows after it under it. A
    subroutine a row defines joins ``subs``, which the calls after it name."""
    out: list[Statement] = []
    depth = rows[0][0] if rows else 0
    while rows:
        (indent, number, text), rows = rows[0], rows[1:]
        under, rows = _under(rows, depth)
        orelse: list = []
        if text.startswith("if ") and rows and rows[0][2].replace(" ", "") == "else:":
            orelse, rows = _under(rows[1:], depth)
        try:
            if indent != depth:
                raise ValueError("unexpected indentation")
            if text.startswith("sub "):
                sub = _Line(text).subroutine(under, subs)
                subs[sub.name] = sub
            else:
                out.append(_Line(text).statement(under, orelse, subs))
        except ValueError as exc:
            raise ValueError(str(exc) if str(exc).startswith("line ") else f"line {number}: {exc}") from exc
    return Body(out)


class _Line(Parser):
    """One line of a body's text."""

    PUNCTUATION = (*Parser.PUNCTUATION, "<-", "->", "+=", "=", "..", "~")

    def at_word(self, word: str, ahead: int = 0) -> bool:
        token = self.peek(ahead)
        return token is not None and token[:2] == ("name", word)

    def word(self, word: str) -> bool:
        """Step over the keyword ``word`` when it is next."""
        if not self.at_word(word):
            return False
        self.index += 1
        return True

    def name(self) -> str:
        kind, value, _position = self.take()
        if kind not in ("name", "quoted"):
            self.index -= 1
            raise self.error("expected a name")
        return value

    def names(self, close: str | None = None) -> tuple[str, ...]:
        """Names separated by commas, up to ``close`` when one is given."""
        if close is not None and self.accept(close):
            return ()
        out = [self.name()]
        while self.accept(","):
            out.append(self.name())
        if close is not None:
            self.expect(close)
        return tuple(out)

    def value(self) -> float:
        literal = self.number()
        if literal is None:
            raise self.error("expected a number")
        return float(literal.value)

    def flag(self, key: str) -> bool | None:
        """``key=true`` / ``key=false`` when it is next, else ``None``."""
        if not (self.at_word(key) and self.at("=", 1)):
            return None
        self.index += 2
        if not (self.at_word("true") or self.at_word("false")):
            raise self.error("expected true or false")
        return self.take()[1] == "true"

    def end(self, stmt):  # noqa: ANN001, ANN201 — what the line spelled, once nothing is left on it
        if self.peek() is not None:
            raise self.error("unexpected text")
        return stmt

    def dim(self) -> Dim:
        expression = self.expression()
        return Dim(expression, hint=int(self.value())) if self.accept("~") else Dim(expression)

    def axis(self) -> tuple[str, Expression, Dim]:
        """``name in start..extent``."""
        name = self.name()
        if not self.word("in"):
            raise self.error("expected 'in'")
        start = self.expression()
        self.expect("..")
        return name, start, self.dim()

    def whole_axis(self) -> Axis:
        """An axis that is no loop's: ``name in 0..extent`` and its window."""
        name, start, extent = self.axis()
        if start != Literal(0, "int"):
            raise self.error("an axis starts at 0")
        return Axis(name, extent, window=self.window())

    def subroutine(self, under: list, subs: dict[str, Subroutine]) -> Subroutine:
        """``sub name(<axis>, …) -> result:`` over the rows of its body."""
        self.index += 1
        name = self.name()
        self.expect("(")
        axes: list[Axis] = []
        while not self.accept(")"):
            if axes:
                self.expect(",")
            axes.append(self.whole_axis())
        self.expect("->")
        result = self.name()
        self.expect(":")
        return self.end(Subroutine(name, tuple(axes), _block(under, subs), result))

    def window(self) -> Window | None:
        if not (self.at_word("window") and self.at("(", 1)):
            return None
        self.index += 2
        parts: dict = {}
        while not self.accept(")"):
            if parts:
                self.expect(",")
            key = self.name()
            self.expect("=")
            if key == "parent":
                parts[key] = self.whole_axis()
            elif key in ("base", "bound"):
                parts[key] = self.expression()
            elif key == "partition" and (self.at_word("true") or self.at_word("false")):
                parts[key] = self.take()[1] == "true"
            else:
                raise self.error(f"no window part {key!r}")
        return Window(**parts)

    def statement(self, under: list, orelse: list, subs: dict[str, Subroutine]) -> Statement:
        """The statement this line spells, ``under`` the rows of its block (``orelse`` those of an ``else``); a call
        names one of ``subs``."""
        if self.tokens[-1][:2] != ("op", ":") and under:
            raise ValueError(f"line {under[0][1]}: unexpected indentation")
        if (
            (token := self.peek()) is not None
            and token[0] == "name"
            and token[1] in KINDS
            and (self.at("(", 1) or self.at(":", 1) or self.peek(1) is None)
        ):
            self.index += 1
            return self.declared(KINDS[token[1]], under, subs)
        if self.word("for"):
            return self.loop(_block(under, subs))
        if self.word("if"):
            cond = self.expression()
            self.expect(":")
            return self.end(Cond(cond, _block(under, subs), _block(orelse, subs)))
        if self.word("let"):
            dtype, name = self.typed_name()
            self.expect("=")
            return self.end(Let(name, self.expression(), dtype))
        if self.word("init"):
            dtype, name = self.typed_name()
            self.expect("=")
            if dtype is None:
                raise self.error("an init names its dtype")
            return self.end(Init(name, self.value(), dtype=dtype))
        if self.word("zero"):
            dst = self.name()
            if not (self.word("words") and self.accept("=")):
                raise self.error("expected words=N")
            return self.end(ZeroPrologue(dst, int(self.value())))
        dtype, name = self.typed_name()
        if self.accept("["):
            return self.end(self.store(dtype, name, self.expressions("]")))
        if self.accept("<-"):
            op = self.name()
            self.expect("(")
            base, value = self.names(")")
            axes = self.names() if self.word("over") else ()
            return self.end(Accum(name, value, op, dtype, axes, None if base == name else base))
        names = (name, *(self.names() if self.accept(",") else ()))
        self.expect("=")
        if self.at_word("load") and not self.at("(", 1):
            self.index += 1
            buffer = self.name()
            self.expect("[")
            index = self.expressions("]")
            carried = self.name() if self.word("carried") and self.accept("=") else ""
            return self.end(Load(names=names, input=buffer, index=index, dtype=dtype, carried=carried))
        if len(names) > 1:
            raise self.error("only a load binds several names")
        if self.at_word("call") and not self.at("(", 1):
            self.index += 1
            target = subs.get(label := self.name())
            if target is None:
                raise self.error(f"no subroutine {label!r} is defined above")
            self.expect("[")
            buffers = self.names("]")
            self.expect("(")
            target = target if buffers == target.buffers else target.with_buffers(buffers)
            return self.end(Call(name, target, self.expressions(")")))
        if self.at_word("pre") and not self.at("(", 1):
            self.index += 1
            carrier = self.name()
            self.expect("[")
            return self.end(Pre(name, carrier, self.expressions("]")))
        if self.at_word("select") and self.at(":", 1):
            self.index += 2
            return self.end(Select(name, tuple(_Line(text).branch(number) for _indent, number, text in under)))
        if (token := self.peek()) is not None and token[0] == "name" and self.at("(", 1):
            self.index += 2
            return self.end(Assign(name, token[1], self.names(")"), dtype))
        return self.end(self.affine(name, dtype))

    def declared(self, cls: type, under: list, subs: dict[str, Subroutine]) -> Statement:
        """A kind of another layer: its header's fields, then its blocks among the rows ``under`` it."""
        shape = {name: (hint, block) for name, hint, _, block in _shape(cls)}
        values = self.record(cls, shape)
        self.accept(":")
        self.end(None)
        body: list = []
        while under:
            (indent, number, text), under = under[0], under[1:]
            rows, under = _under(under, indent)
            label = text[:-1] if text.endswith(":") and text[:-1] in shape and shape[text[:-1]][1] else None
            if label is None:
                body.extend(((indent, number, text), *rows))
            elif label in values:
                raise ValueError(f"line {number}: {label} is given twice")
            else:
                stmts = _block(rows, subs)
                values[label] = stmts if typing.get_origin(shape[label][0]) is not tuple else tuple(stmts)
        if body:
            if "body" not in shape:
                raise ValueError(f"line {body[0][1]}: a {cls.__name__} holds no body")
            values["body"] = _block(body, subs)
        return cls(**values)

    def record(self, cls: type, shape: dict | None = None) -> dict:
        """The ``(field=value, …)`` of a declared kind or record, each value read by its field's type."""
        shape = {name: (hint, block) for name, hint, _, block in _shape(cls)} if shape is None else shape
        values: dict = {}
        if self.accept("("):
            while not self.accept(")"):
                if values:
                    self.expect(",")
                key = self.name()
                if key not in shape or shape[key][1]:
                    raise self.error(f"a {cls.__name__} has no field {key!r} on its header")
                self.expect("=")
                values[key] = self.typed(shape[key][0])
        return values

    def typed(self, hint: object):  # noqa: ANN201 — the value the field's type ``hint`` reads
        """A value of the declared type ``hint``."""
        origin = typing.get_origin(hint)
        if origin in (types.UnionType, typing.Union):
            options = [option for option in typing.get_args(hint) if option is not type(None)]
            if self.word("none"):
                return None
            return self.typed(options[0])
        if origin is tuple:
            args = typing.get_args(hint)
            self.expect("(")
            out: list = []
            if len(args) == 2 and args[1] is Ellipsis:
                while not self.accept(")"):
                    if out:
                        self.expect(",")
                    out.append(self.typed(args[0]))
            else:
                for index, arg in enumerate(args):
                    if index:
                        self.expect(",")
                    out.append(self.typed(arg))
                self.expect(")")
            return tuple(out)
        if hint is bool:
            if not (self.at_word("true") or self.at_word("false")):
                raise self.error("expected true or false")
            return self.take()[1] == "true"
        if hint is int:
            return int(self.value())
        if hint is float:
            return self.value()
        if hint is str:
            return self.name()
        if hint is DataType:
            return dtype_of(self.name())
        if hint is ElementwiseImpl:
            return ElementwiseImpl(self.name())
        if hint is Expression:
            return self.expression()
        if hint is Axis:
            return self.whole_axis()
        if isinstance(hint, type) and is_dataclass(hint) and not issubclass(hint, (Statement, Body)):
            if self.name() != hint.__name__:
                raise self.error(f"expected a {hint.__name__}")
            return hint(**self.record(hint))
        raise self.error(f"a field of type {hint} is not read on a header")

    def branch(self, number: int) -> SelectBranch:
        """``value when predicate``, one branch of a select, on line ``number``."""
        try:
            value = self.name()
            if not self.word("when"):
                raise self.error("expected 'when'")
            return self.end(SelectBranch(value, self.expression()))
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc

    def typed_name(self) -> tuple[DataType | None, str]:
        """``[dtype] name``: two names in a row are a dtype and the name it types."""
        first = self.peek()
        following = self.peek(1)
        if first is not None and first[0] == "name" and following is not None and following[0] in ("name", "quoted"):
            self.index += 1
            return dtype_of(first[1]), self.name()
        return None, self.name()

    def store(self, dtype: DataType | None, name: str, index: tuple[Expression, ...]) -> Statement:
        """What follows ``name[index]``: a carried state's next value, or a write."""
        if self.accept("<-"):
            value = self.name()
            if not self.word("seed"):
                raise self.error("expected 'seed'")
            literal = self.number()
            return Carry(name, value, index, float(literal.value) if literal is not None else self.name(), dtype)
        atomic = self.accept("+=")
        if not atomic:
            self.expect("=")
        values = self.names(")") if self.accept("(") else (self.name(),)
        swizzle = self.name() if self.word("swizzle") and self.accept("=") else "NONE"
        return Write(name, index, values=values, value_dtype=dtype, atomic=atomic, swizzle=swizzle)

    def affine(self, name: str, dtype: DataType | None) -> Affine:
        args: list[str] = []
        coefficients: list[float] = []
        offset = 0.0
        while self.peek() is not None:
            negative = self.accept("-")
            if not negative and (args or offset) and not self.accept("+"):
                raise self.error("expected '+' or '-'")
            sign = -1.0 if negative else 1.0
            literal = self.number()
            if literal is None or self.accept("*"):
                args.append(self.name())
                coefficients.append(sign * (1.0 if literal is None else float(literal.value)))
            else:
                offset = sign * float(literal.value)
                break
        return Affine(name, tuple(args), tuple(coefficients), offset, dtype)

    def loop(self, body: Body) -> Statement:
        name, start, extent = self.axis()
        step = self.expression() if self.word("step") else None
        end = self.expression() if self.word("until") else None
        axis = Axis(name, extent, window=self.window())
        unroll, seed = self.flag("unroll") or False, self.flag("seed")
        self.expect(":")
        flags = {"unroll": unroll, "seed": True if seed is None else seed}
        if step is not None:
            return self.end(StridedLoop(axis, start, step, body, end=end, **flags))
        if start != Literal(0, "int") or end is not None:
            raise self.error("a loop that does not start at 0, or stops early, names its step")
        return self.end(Loop(axis, body, **flags))
