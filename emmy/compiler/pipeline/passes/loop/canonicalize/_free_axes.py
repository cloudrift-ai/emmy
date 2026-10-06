"""The free-coordinate canonicalization ``loop/canonicalize`` applies, as a body → body function: a flattened
coordinate read through its quotient and remainder split into its factors, then perfectly nested free pairs that
fold clean fused into one axis. The rule documents the design; the cut pass forms each piece through it too, so a
piece is the kernel its own program canonicalizes to."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.address import split_pair
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import simplify as _simplify_stmt


def _no_divmod_on(e: Expr, name: str) -> bool:
    """No ``/`` / ``%`` subterm of ``e`` has ``name`` in its dividend — the residue detector."""
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "//", "%") and name in e.left.free_vars():
            return False
        return _no_divmod_on(e.left, name) and _no_divmod_on(e.right, name)
    if isinstance(e, TernaryExpr):
        return all(_no_divmod_on(x, name) for x in (e.cond, e.if_true, e.if_false))
    if isinstance(e, CastExpr):
        return _no_divmod_on(e.expr, name)
    if isinstance(e, FuncCallExpr):
        return all(_no_divmod_on(a, name) for a in e.args)
    return True


def _access_ok(index: tuple, shape, fname: str, store: bool = False) -> bool:
    """A rewritten access folds clean when its index exprs carry no div/mod residue on the fused
    axis, or — the split-store spelling ``[…, f/Q, f%Q]`` — when the buffer's row-major flatten
    recomposes the residue to an affine address (needs the full static shape). A ``store`` may
    also keep the bare pair at permuted strides (``[…, f/Q, …, f%Q]``): every tier addresses an
    output element at its own coordinate, so the spelling is exact by construction."""
    if all(_no_divmod_on(e, fname) for e in index if fname in e.free_vars()):
        return True
    if store and split_pair(index, fname) is not None:
        return True
    if shape is None or len(shape) != len(index) or not all(getattr(d, "is_static", False) for d in shape):
        return False
    flat: Expr = Literal(0, "int")
    stride = 1
    for e, d in zip(reversed(index), reversed(list(shape)), strict=True):
        flat = BinaryExpr("+", flat, BinaryExpr("*", e, Literal(stride, "int")))
        stride *= d.as_static()
    return _no_divmod_on(flat.simplify(SimplifyCtx.empty()), fname)


def _folds_clean(fused: Loop, fname: str, shapes: dict) -> bool:
    for s in Body((fused,)).iter():
        if isinstance(s, Load):
            if not _access_ok(s.index, shapes.get(s.input), fname):
                return False
        elif isinstance(s, Write):
            if not _access_ok(s.index, shapes.get(s.output), fname, store=True):
                return False
        elif any(fname in e.free_vars() and not _no_divmod_on(e, fname) for e in s.exprs()):
            return False
    return True


def _fuse_pair(outer: Loop, inner: Loop, shapes: dict, between: tuple[Loop, ...] = ()) -> Loop | None:
    """The fused nest for a perfectly-nested free pair, or ``None`` when the pair declines. The
    free loops ``between`` them (outermost first) interchange outward: the fused axis sits where
    ``inner`` was, under them."""
    p, q = outer.axis, inner.axis
    if p.name == q.name or p.window is not None or q.window is not None:
        return None
    if not (p.extent.is_static and q.extent.is_static):
        return None
    big, small = p.extent.as_static(), q.extent.as_static()
    if big <= 1 or small <= 1:
        return None  # a size-1 side is drop_size_one_free_axes' job
    for s in Body(tuple(inner.body)).iter():
        ax = getattr(s, "axis", None)
        if ax is not None and ax.name in (p.name, q.name):
            return None  # an inner loop shadows a pair name — substitution would capture
    f = Var(q.name)
    lit = Literal(small, "int")
    sigma = Sigma({p.name: BinaryExpr("//", f, lit), q.name: BinaryExpr("%", f, lit)})
    body = Body(tuple(s.substitute(sigma) for s in inner.body))
    fused = Loop(axis=Axis(q.name, big * small), body=body, unroll=outer.unroll or inner.unroll, seed=inner.seed)
    fused = _simplify_stmt(fused, SimplifyCtx.empty())
    if not _folds_clean(fused, q.name, shapes):
        return None
    for mid in reversed(between):
        fused = replace(mid, body=Body((fused,)))
    return fused


def _free_chain(loop: Loop) -> list[Loop]:
    """``loop`` and the perfectly-nested free loops under it, outermost first."""
    out = [loop]
    while len(out[-1].body) == 1 and isinstance(out[-1].body[0], Loop) and not out[-1].body[0].is_reduce:
        out.append(out[-1].body[0])
    return out


def _fuse_once(body: Body, shapes: dict) -> Body | None:
    """The body with ONE pair fused (outermost-first, depth-first, the nearest partner first), or
    ``None`` when no pair fuses. The caller iterates to fixpoint, so an outer pair exposed by an
    inner fusion is picked up on the next round."""
    for i, s in enumerate(body):
        if not isinstance(s, Loop) or s.is_reduce:
            continue
        chain = _free_chain(s)
        for j in range(1, len(chain)):
            fused = _fuse_pair(s, chain[j], shapes, tuple(chain[1:j]))
            if fused is not None:
                return Body((*body[:i], fused, *body[i + 1 :]))
        inner = _fuse_once(s.body, shapes)
        if inner is not None:
            return Body((*body[:i], replace(s, body=inner), *body[i + 1 :]))
    return None


def _reads_whole(e: Expr, name: str) -> bool:
    """Whether ``e`` reads ``name`` outside every ``/`` / ``%`` dividend."""
    if isinstance(e, Var):
        return e.name == name
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "//", "%"):
            return _reads_whole(e.right, name)
        return _reads_whole(e.left, name) or _reads_whole(e.right, name)
    if isinstance(e, TernaryExpr):
        return any(_reads_whole(x, name) for x in (e.cond, e.if_true, e.if_false))
    if isinstance(e, CastExpr):
        return _reads_whole(e.expr, name)
    if isinstance(e, FuncCallExpr):
        return any(_reads_whole(a, name) for a in e.args)
    return False


def _divisors(loads, name: str) -> dict[int, set[str]]:
    """The literal divisors ``name`` is read through, and whether as quotient, remainder or both."""
    divisors: dict[int, set[str]] = {}
    for load in loads:
        for index in load.index:
            for expr in index.subterms():
                if (
                    isinstance(expr, BinaryExpr)
                    and expr.op in ("/", "//", "%")
                    and expr.left == Var(name)
                    and isinstance(expr.right, Literal)
                    and isinstance(expr.right.value, int)
                    and expr.right.value > 1
                ):
                    divisors.setdefault(expr.right.value, set()).add("%" if expr.op == "%" else "/")
    return divisors


def _owned_whole(stmt: Loop, name: str, factor: int) -> bool:
    """Whether every reduction that reads ``name`` through ``factor`` also reads it whole — one
    operand owning the coordinate, not two operands sharing its factors. A packed int4 weight reads
    its channel whole beside the zero-point's ``n / 8`` and the shift's ``n % 8``; splitting that
    channel leaves the contraction two own axes on its channel side, of which the tile can take
    only the 8-wide one. A reduction reading only the factors (a head/dim pair) still needs the split."""
    sharing = [
        loads
        for loop in stmt.body.iter_of_type(Loop)
        if loop.is_reduce
        for loads in [list(loop.body.iter_of_type(Load))]
        if factor in _divisors(loads, name)
    ]
    return bool(sharing) and all(any(_reads_whole(i, name) for load in loads for i in load.index) for loads in sharing)


def _split_once(body: Body, names: frozenset[str]) -> Body | None:
    """Expose a free coordinate's quotient and remainder as separate operand axes.

    A flattened row/head coordinate reads A through ``i / H`` and B through ``i % H``.
    Neither operand owns that mixed axis. The exact inverse of the fusion above restores both
    coordinates; fusion cannot undo it because those separate operand reads do not fold clean.
    A reduction that also reads the coordinate whole owns it (:func:`_owned_whole`) and keeps it.
    """
    for i, stmt in enumerate(body):
        if not isinstance(stmt, Loop) or stmt.is_reduce:
            continue
        axis = stmt.axis
        if axis.extent.is_static and axis.window is None and not stmt.body.carries:
            extent = axis.extent.as_static()
            for factor, uses in sorted(_divisors(stmt.body.iter_of_type(Load), axis.name).items()):
                if uses != {"/", "%"} or factor >= extent or extent % factor or _owned_whole(stmt, axis.name, factor):
                    continue
                outer = axis.name + "_quotient"
                while outer in names:
                    outer += "_"
                sigma = Sigma({axis.name: Var(outer) * Literal(factor, "int") + Var(axis.name)})
                inner = replace(stmt, axis=Axis(axis.name, factor), body=Body(s.substitute(sigma) for s in stmt.body))
                split = Loop(Axis(outer, extent // factor), Body((inner,)), unroll=stmt.unroll, seed=stmt.seed)
                split = _simplify_stmt(split, SimplifyCtx.empty())
                return Body((*body[:i], split, *body[i + 1 :]))
        inner = _split_once(stmt.body, names)
        if inner is not None:
            return Body((*body[:i], replace(stmt, body=inner), *body[i + 1 :]))
    return None


def canonical_free_axes(body: Body, shapes: dict) -> Body | None:
    """``body`` with its free coordinates canonical, or ``None`` when they already are. ``shapes`` holds the buffer
    shapes a flattened access needs to fold clean; a buffer missing from it declines that fold."""
    changed = False
    while (step := _split_once(body, body.axis_names)) is not None:
        body, changed = step, True
    while (step := _fuse_once(body, shapes)) is not None:
        body, changed = step, True
    return body if changed else None
