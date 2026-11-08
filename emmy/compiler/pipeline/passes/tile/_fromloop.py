"""Mechanical Loop IR reduction lifting.

A reduce ``Loop`` already states its fold algebra in its ``Accum`` members. Lift it directly:
recursively replace nested reductions in place, remove the ``Accum`` statements from the step,
and store their operations as the fold's componentwise monoid. The one shape formation reads is
the SEMIRING step (:func:`_factor_products`): its product arguments become operand edges — a slab,
a cone over the step, or the raw slab of a storage decode with the invariant factors hoisted to an
epilogue — so the bilinear reading is canonical by construction. There is no round-trip gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, TernaryExpression, Var, apply
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import EXTENTS, Fold, binding_extents
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Carry,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    SelectBranch,
    Statement,
    Write,
)
from emmy.compiler.ir.tile import Placement, TileOp, extract_output_specs
from emmy.compiler.ir.tile.ir import guarded_store

if TYPE_CHECKING:
    from collections.abc import Mapping

    from emmy.compiler.graph import Tensor


def _stamp_axes(loop: Loop) -> Loop:
    """Attach each accumulator to the reduce axis that contains it."""
    body = tuple(
        _stamp_axes(stmt)
        if isinstance(stmt, Loop)
        else stmt.replace(axes=(loop.axis.name,))
        if isinstance(stmt, Accum) and not stmt.axes
        else stmt
        for stmt in loop.body
    )
    return loop.replace(body=Body(body))


@dataclass
class _Level:
    """One statement level under construction: the axes in scope and what the level has produced
    so far — its statements, and by name the terms it exposes. A term formed at or below this level
    closes over these (:func:`_supply`); what a reader took is remembered so the level can drop a
    statement or a sibling whose one remaining position is under that reader."""

    axes: tuple
    stmts: list = field(default_factory=list)
    exposed: dict = field(default_factory=dict)
    consumed: set = field(default_factory=set)
    drained: set = field(default_factory=set)
    #: The carried states in scope, each with the axis of the loop carrying it — what a ``Pre``
    #: read below this level names (:meth:`Fold.carrier_read`).
    carriers: dict = field(default_factory=dict)


def _supply(names: set[str], levels: tuple[_Level, ...]) -> tuple[Fold, ...]:
    """The operands that close a term over the VALUES it reads from its levels, innermost first.

    A statement a term reads arrives as an operand — a bare load as a slab, a scalar chain as a
    zero-axis cone over that level's statements, itself closed the same way — and a sibling term's
    state as that sibling, the same object, so sharing keeps it one value.
    """
    extra: list[Fold] = []
    pending = set(names) - EXTENTS.get()
    for depth in range(len(levels) - 1, -1, -1):
        level = levels[depth]
        defined = {name for stmt in level.stmts for name in stmt.defines()}
        chain = sorted(name for name in pending if name in defined)
        siblings = sorted(name for name in pending if name in level.exposed)
        pending -= set(chain) | set(siblings)
        if chain:
            cone = Body(tuple(level.stmts)).backward_cone(tuple(chain))
            level.consumed.update(id(stmt) for stmt in cone.members)
            if len(cone.members) == 1 and isinstance(cone.members[0], Load):
                extra.append(Fold.slab(cone.members[0]))
            else:
                values = set(cone.external_reads) - {axis.name for axis in level.axes}
                operands, lift = _close(
                    (), _supply(values, levels[: depth + 1]), Body(cone.members), tuple(chain), level.axes, levels[: depth + 1]
                )
                extra.append(Fold(operands=operands, lift=lift))
        for name in siblings:
            term = level.exposed[name]
            level.drained.add(id(term))
            if all(term is not edge for edge in extra):
                extra.append(term)
        if not pending:
            break
    if pending:
        raise ValueError(f"a term reads {sorted(pending)}, which no enclosing level defines")
    return tuple(extra)


def _close(lead: tuple, operands: tuple, body, results: tuple, scope: tuple, levels: tuple) -> tuple[tuple, Lambda]:
    """Form a lift CLOSED over its levels — ``(operands, lift)``.

    ``Lambda.closing`` binds the operand results positionally and leaves whatever else the body
    reads as trailing params. A VALUE among those arrives as one more operand (:func:`_supply`);
    what remains free are coordinates, which a term reads without binding — only the binder can
    tell the two apart, by ``scope``.
    """
    bound = tuple(name for edge in operands for name in edge.exposes)
    lift = Lambda.closing((*lead, *bound), body, results)
    values = set(lift.params[len(lead) + len(bound) :]) - {axis.name for axis in scope} - EXTENTS.get()
    if values:
        operands = (*operands, *_supply(values, levels))
        bound = tuple(name for edge in operands for name in edge.exposes)
        lift = Lambda.closing((*lead, *bound), body, results)
    return operands, lift


def product_spine(
    value: Expression, defs: dict, *, divide: bool = False
) -> tuple[tuple[Expression, ...], tuple[Expression, ...], tuple[Statement, ...]]:
    """Flatten the ``⊗`` spine of ``value`` into ``(leaves, divisors, followed)`` — the spine recognized by the
    ``semiring_product`` TRAIT, never an op-name list, through the definitions in ``defs`` a name on the spine
    stands for (``followed``: the statements the spine runs through, which a caller checks nothing else reads). A
    leaf is an operand that is no product: a name, a literal or another operation. ``divide`` additionally admits a
    division on the numerator side: ``(Σ x)/c`` equals ``Σ (x/c)`` for a fold-invariant ``c``, but nothing licenses
    moving a fold into a denominator, so the divisor is a leaf and only the numerator continues the spine. A value
    with no product above it is the one-leaf product."""
    leaves: list[Expression] = []
    divisors: list[Expression] = []
    followed: list[Statement] = []

    def walk(current: Expression) -> None:
        op = current.operation
        if isinstance(current, Var) and isinstance(stmt := defs.get(current.name), Assign) and stmt.dtype is None:
            inner = stmt.value.operation
            if inner is not None and (inner.semiring_product or (divide and inner.name == "divide")) and len(stmt.value.operands) == 2:
                followed.append(stmt)
                walk(stmt.value)
                return
        if op is not None and op.semiring_product and len(current.operands) == 2:
            for operand in current.operands:
                walk(operand)
            return
        if divide and op is not None and op.name == "divide" and len(current.operands) == 2:
            divisors.append(current.operands[1])
            leaves.append(current.operands[1])
            walk(current.operands[0])
            return
        leaves.append(current)

    walk(value)
    return tuple(leaves), tuple(divisors), tuple(followed)


@dataclass
class _Hoist:
    """Fold-invariant factors commuted off one product argument — ``Σ_k a·(s·w) = s·Σ_k a·w`` —
    the terms and statements that define them, and the factors in spine order, each with whether it
    divides."""

    terms: tuple = ()
    stmts: tuple = ()
    factors: tuple = ()


def _decode_split(arg: str, plain: Body, defs: dict, exposed: dict, k_name: str) -> tuple[Fold, _Hoist] | None:
    """A product argument that is a STORAGE DECODE of one raw slab times fold-invariant factors,
    split into ``(the raw slab, its hoist)`` — or ``None`` for any other cone, which stays whole.

    The decode is absorbed by the slab's storage dtype (every consumer converts a bits-carrier
    element by dtype, the mma fragment loaders included), and the invariant factors commute out
    onto the accumulator: the same reassociation category as split-K. Only a decode licenses it,
    so an ordinary floating-point factor chain is never reassociated here."""
    stmt = defs.get(arg)
    if not isinstance(stmt, Assign):
        return None
    leaves, divisors, followed = product_spine(stmt.value, defs, divide=True)

    def varies(leaf: Expression) -> bool:
        for name in leaf.free_vars():
            if name in exposed:
                if k_name in exposed[name].free_axes:
                    return True
            elif name in defs:
                reads = plain.backward_cone((name,)).external_reads
                if k_name in reads or any(k_name in exposed[read].free_axes for read in reads if read in exposed):
                    return True
        return False

    varying = [leaf for leaf in leaves if varies(leaf)]
    if len(varying) != 1:
        return None
    # The decode is the one varying leaf: a name the step defines (the lift names every operation node).
    named = defs.get(varying[0].name) if isinstance(varying[0], Var) else None
    decode = named.value if isinstance(named, Assign) else varying[0]
    code = decode.operands[0] if decode.operation is not None and len(decode.operands) == 1 else None
    if decode.operation is None or decode.operation.decodes is None or not isinstance(code, Var) or code.name not in exposed:
        return None
    slab = exposed[code.name].as_slab()
    if slab is None or (slab.load.dtype is not None and slab.load.dtype.name != decode.operation.decodes):
        return None
    seen = {id(stmt), *(id(member) for member in followed), *([id(named)] if named is not None else [])}
    stmts: list[Statement] = []
    terms: list[Fold] = []
    for leaf in leaves:
        if leaf is varying[0]:
            continue
        for name in leaf.free_vars():
            if name in exposed:
                terms.append(exposed[name])
            elif name in defs:
                stmts.extend(member for member in plain.backward_cone((name,)).members if id(member) not in seen)
                seen.update(id(member) for member in stmts)
    if any(id(member) not in seen for member in plain.backward_cone((arg,)).members):
        return None
    terms.extend(
        exposed[name] for member in stmts for name in member.deps() if name in exposed and all(exposed[name] is not t for t in terms)
    )
    factors = tuple((leaf, leaf in divisors) for leaf in leaves if leaf is not varying[0])
    return exposed[code.name], _Hoist(tuple(dict.fromkeys(terms)), tuple(stmts), factors)


def _operand_names(product: Assign) -> tuple[str, ...]:
    """The names a product's operands read: every operand of a lifted step's product is a name (:func:`_one_per_node`)."""
    return tuple(operand.name for operand in product.value.operands if isinstance(operand, Var))


def _one_per_node(body: Body) -> Body:
    """``body`` with one definition per operation node: every nested operation node of an ``Assign``'s value is a
    definition of its own ahead of its reader, named after it (``v__e0``, ``v__e1``, …), and an ``Accum`` sums a name.
    An affine region (sums, differences, negations and literal scales, the normal form's affine form) is one node, as a
    sum is one operation. The Loop IR's normal form folds a value read once into its reader; the tile IR's nodes are
    one operation each, so the lift translates between the two granularities here, once, and every cone, edge and seam
    it forms downstream is the one the chain spelling gave. Nested bodies are lifted by their own call."""
    taken = set(body.ssa_uses) | {name for stmt in body.iter() for name in stmt.defines()}
    out: list[Statement] = []

    def fresh(base: str) -> str:
        index = 0
        while f"{base}__e{index}" in taken:
            index += 1
        taken.add(f"{base}__e{index}")
        return f"{base}__e{index}"

    def part(node: Expression, base: str, in_affine: bool) -> Expression:
        """``node`` with the operation nodes below it defined, itself defined too unless it is the root or a node of
        the enclosing affine region."""
        affine = _affine_node(node)
        rebuilt = node.with_children([part(child, base, affine) for child in node.children()])
        if rebuilt.operation is None or (in_affine and affine):
            return rebuilt
        name = fresh(base)
        out.append(Assign(name, rebuilt))
        return Var(name)

    for stmt in body:
        if isinstance(stmt, Assign):
            affine = _affine_node(stmt.value)
            value = stmt.value.with_children([part(child, stmt.name, affine) for child in stmt.value.children()])
            out.append(stmt if value == stmt.value else stmt.replace(value=value))
        elif isinstance(stmt, Accum) and not isinstance(stmt.value, Var):
            value = part(stmt.value, stmt.name, False)
            if not isinstance(value, Var):
                name = fresh(stmt.name)
                out.append(Assign(name, value))
                value = Var(name)
            out.append(stmt.replace(value=value))
        else:
            out.append(stmt)
    return Body(tuple(out))


def _affine_node(node: Expression) -> bool:
    """Whether ``node`` is a node of an affine region: a sum, a difference, a negation, or a product or quotient with a
    literal factor."""
    op = node.operation
    if op is None:
        return False
    if op.name in ("add", "subtract", "negative"):
        return True
    operands = node.operands
    if op.name == "multiply":
        return any(isinstance(operand, Literal) for operand in operands)
    return op.name == "divide" and len(operands) == 2 and isinstance(operands[1], Literal)


def _factor_products(
    plain: Body, values: tuple, ops: tuple, terms: tuple, scope: tuple, levels: tuple, axes: tuple, *, hoist: bool
) -> tuple:
    """Factor a SEMIRING step into ``(operand edges, products)`` — the bilinear form by construction.

    The step is a semiring step when every accumulated value is one product ``⊗`` of two distinct
    names, all products share the ⊗, and ⊗ distributes over the one commutative-monoid ⊕ the
    accumulators fold through. Each product argument the step computes (the storage decode ``w_bits ×
    scale``, the normalized ``x × rsqrt``) is then a cone over the step, hoisted into a zero-axis
    term closed like any other (:func:`_close`); arguments whose cones overlap share one term
    exposing both; a slab or a nested reduce an argument names outright rides as that edge. The
    products are all that remains of the step. Any other step — a non-semiring ⊕, a square
    ``x × x``, a member no product reads — is returned as it came. With ``hoist``, an argument that
    is a storage decode times invariant factors is split (:func:`_decode_split`) and the factors
    are returned per accumulator, for the epilogue :func:`_hoisted` wraps around the fold.

    The pair is ORIENTED here, where the enclosing axis order is known: a product argument shared
    by every channel leads (the fused sibling edge's one A), else the argument carrying the earlier
    output axis does, so ``operands[0]`` is A by construction and placement reads M/N off it."""
    unfactored = (terms, plain, {})
    plus = ops[0]
    if len(set(ops)) != 1 or not (plus.associative and plus.commutative and plus.has_identity):
        return unfactored
    defs = plain.definitions
    products = [defs.get(value) for value in values]
    if any(not isinstance(product, Assign) or len(set(_operand_names(product))) != 2 for product in products):
        return unfactored
    operation = products[0].value.operation
    if any(product.value.operation != operation or not product.value.operation.distributes_over(plus) for product in products):
        return unfactored
    product_ids = {id(product) for product in products}
    exposed = {name: term for term in terms for name in term.exposes}
    computed = tuple(dict.fromkeys(arg for product in products for arg in _operand_names(product) if arg in defs))
    if any(id(defs[arg]) in product_ids for arg in computed):
        return unfactored
    cones = {arg: plain.backward_cone((arg,)) for arg in computed}
    covered = product_ids | {id(member) for cone in cones.values() for member in cone.members}
    if any(id(stmt) not in covered for stmt in plain):
        return unfactored
    edge_of: dict[str, Fold] = {}
    hoists: dict[str, _Hoist] = {}
    k_name = scope[-1].name
    for arg in computed if hoist else ():
        split = _decode_split(arg, plain, defs, exposed, k_name)
        if split is not None:
            edge_of[arg], hoists[arg] = split
    # A split argument's product reads the raw slab in its place.
    raw_names = {arg: edge.exposes[0] for arg, edge in edge_of.items()}
    products = [product.replace(value=product.value.rename(raw_names)) for product in products]
    # Overlapping cones are one term exposing every root they share members with.
    groups: list[list[str]] = []
    for arg in (arg for arg in computed if arg not in edge_of):
        members = {id(member) for member in cones[arg].members}
        touching = [group for group in groups if any(members & {id(member) for member in cones[other].members} for other in group)]
        if touching:
            touching[0].append(arg)
            for group in touching[1:]:
                touching[0].extend(group)
                groups.remove(group)
        else:
            groups.append([arg])
    consumed: set[int] = set()
    arguments = tuple(dict.fromkeys(arg for product in products for arg in _operand_names(product)))
    for group in groups:
        cone = plain.backward_cone(tuple(group))
        reads = tuple(term for term in terms if any(name in cone.external_reads for name in term.exposes))
        consumed.update(id(term) for term in reads)
        # A product argument this cone READS (``x`` beside ``x · scale``) rides the cone as a
        # pass-through result: the two arguments are ONE multi-result edge, not a term and a cone
        # over it competing for the shared A slot.
        passed = tuple(arg for arg in arguments if arg in exposed and arg not in edge_of and any(arg in term.exposes for term in reads))
        results = tuple(arg for arg in arguments if arg in group or arg in passed)  # as the products read them
        operands, lift = _close((), reads, Body(cone.members), results, scope, levels)
        edge_of.update((arg, Fold(operands=operands, lift=lift)) for arg in results)
    pairs = [
        tuple(edge_of.get(arg) or exposed.get(arg) for arg in _operand_names(product)) for product in products
    ]  # raw names are exposed
    # Alpha-equal argument edges (two loads of one slab, two spellings of one cone) are ONE edge:
    # the first survives and the later products read its names, so the channels share A outright.
    survivors: dict[tuple, Fold] = {}
    renames: dict[str, str] = {}

    def unified(edge):
        if edge is None:
            return None
        keep = survivors.setdefault((edge.canonical(), frozenset(edge.free_axes)), edge)
        if keep is not edge:
            renames.update(zip(edge.exposes, keep.exposes, strict=True))
            consumed.add(id(edge))
        return keep

    pairs = [tuple(unified(edge) for edge in pair) for pair in pairs]
    if renames:
        products = [product.replace(value=product.value.rename(renames)) for product in products]
    ordered = _orient(pairs, tuple(axis.name for axis in axes))
    split_names = {raw: arg for arg, raw in raw_names.items()}
    consumed.update(id(term) for hoist in hoists.values() for term in hoist.terms)  # a factor's slab rides the epilogue
    consumed -= {id(edge) for edge in ordered}
    leftover = tuple(term for term in terms if id(term) not in consumed and all(term is not edge for edge in ordered))
    per_state = {
        index: tuple(hoists[split_names[name]] for name in _operand_names(product) if name in split_names)
        for index, product in enumerate(products)
    }
    hoisted = {index: hoist for index, hoist in per_state.items() if any(h.factors for h in hoist)}  # a bare decode has no epilogue
    return (*ordered, *leftover), Body(tuple(products)), hoisted


def _hoisted(fold: Fold, names: tuple[str, ...], hoists: dict, axes: tuple, levels: tuple) -> Fold:
    """The epilogue projection that applies the hoisted factors to the fold's states, under the
    original state names: the fold reduces into ``<state>__sum``, the projection multiplies (or
    divides) that by each factor in spine order and exposes the result as ``<state>``."""
    inner = {name: f"{name}__sum" for index, name in enumerate(names) if index in hoists}
    fold = replace(fold, base=fold.base.rename(inner))
    terms: list[Fold] = [fold]
    epilogue: list[Statement] = []
    for index, name in enumerate(names):
        current = inner.get(name, name)
        factors = [factor for hoist in hoists.get(index, ()) for factor in hoist.factors]
        for hoist in hoists.get(index, ()):
            terms.extend(term for term in hoist.terms if all(term is not held for held in terms))
            epilogue.extend(stmt for stmt in hoist.stmts if all(stmt is not held for held in epilogue))
        for position, (leaf, divide) in enumerate(factors):
            result = name if position == len(factors) - 1 else f"{name}__c{position}"
            epilogue.append(Assign(result, apply("divide" if divide else "multiply", current, leaf)))
            current = result
    operands, lift = _close((), tuple(terms), Body(tuple(epilogue)), names, axes, levels)
    return Fold(operands=operands, lift=lift)


def _orient(pairs: list[tuple], axes: tuple[str, ...]) -> tuple[Fold, ...]:
    """The unique edges the products read, A first: the edge every channel shares, else — for one
    channel — the edge whose earliest own output axis comes earlier in ``axes`` (the normal form orders a
    product's operands by name, so the order as read decides nothing). An argument no edge here supplies (an
    enclosing value ``_close`` binds later) leaves the order as read."""
    edges: list[Fold] = []
    for pair in pairs:
        edges.extend(edge for edge in pair if edge is not None and all(edge is not seen for seen in edges))
    if any(edge is None for pair in pairs for edge in pair):
        return tuple(edges)
    if len(pairs) > 1:
        shared = [edge for edge in edges if all(any(edge is member for member in pair) for pair in pairs)]
        lead = shared[0] if len(shared) == 1 else None
    else:
        left, right = pairs[0]
        left_only, right_only = left.free_axes - right.free_axes, right.free_axes - left.free_axes
        lead = None
        if left_only and right_only:
            position = {name: index for index, name in enumerate(axes)}
            earliest = [min(position.get(name, len(axes)) for name in own) for own in (left_only, right_only)]
            lead = left if earliest[0] <= earliest[1] else right
    return tuple(edges) if lead is None else (lead, *(edge for edge in edges if edge is not lead))


def _renamed_sweep(loop: Loop, taken: set[str]) -> Loop:
    """``loop`` under a FRESH axis name — a sibling sweep that reuses an enclosing or sibling loop's
    name (the fused encode kernel's byte sweep and scale sweep, both ``a1`` at different
    extents). Loop IR binds per loop, so the reuse is legal there; a term tree names one coordinate
    per name, so the second sweep is alpha-renamed, binder and references together."""
    taken = taken | set(Body(loop.body).definitions)
    fresh = next(name for index in range(1, len(taken) + 2) if (name := f"{loop.axis.name}_{index}") not in taken)
    coords = Sigma({loop.axis.name: Var(fresh)})
    return loop.replace(axis=replace(loop.axis, name=fresh), body=Body(tuple(stmt.substitute(coords) for stmt in loop.body)))


def lift_body(body, axes: tuple = (), levels: tuple = (), carriers: dict | None = None) -> tuple[tuple, Body]:
    """Lift one statement tree into ``(operand terms, statements)`` — SEPARATED, bottom up.

    A reduction becomes an operand EDGE of the level it sat in; it is never substituted into the
    statement sequence where its ``Loop`` stood. So no mixed stmt/term stream exists at any point
    and none crosses a function boundary: a ``Body`` holds statements because that is all it was
    ever given, rather than because a later boundary sorted a sequence that should not have been
    built. Separation IS the construction, and so is closure: a term takes what it reads from its
    levels as operands when it is FORMED (:func:`_close`), so the level lowers its operands first
    and its statements after, with nothing left to order by dependency.

    An output SWEEP lifts to terms evaluated over the sweep coordinate: a reduce under it —
    attention's ``Σ_k P·V`` per output column — joins the enclosing level's operands, its slabs
    reading the sweep axis, and the sweep's own per-cell projection joins beside it as a
    zero-axis term evaluated over that axis. The sweep keeps its stores alone; the boundary extracts
    them as sweep specs, and ``Fold.lower`` opens the sweep loop around exactly the terms evaluated
    over it.

    ``axes`` names the iteration variables the ENCLOSING loops bind, threaded down from
    :func:`_peel`; ``levels`` are the enclosing levels under construction, the providers a term
    formed here may close over. A term cannot tell an axis from a value — both are a bare ``Var``
    — but the binder can, because it bound them; so the classification arrives from above rather
    than being inferred by walking a lowered body for names that look axis-shaped. ``carriers``
    are the carried states in scope with their carrying axes, inherited from the enclosing level
    unless a carrying loop adds its own.
    """
    level = _Level(axes, carriers=carriers if carriers is not None else (levels[-1].carriers if levels else {}))
    inner_levels = (*levels, level)
    edges: list = []
    bound = {axis.name for axis in axes}  # one coordinate per name: a sibling sweep reusing one is renamed apart
    for stmt in _one_per_node(Body.coerce(body)):
        if isinstance(stmt, Loop) and not stmt.is_reduce:
            if stmt.axis.name in bound:
                stmt = _renamed_sweep(stmt, bound)
            bound.add(stmt.axis.name)
        if isinstance(stmt, Loop) and stmt.is_reduce:
            fold, trailing = scan_from_loop(stmt, axes, inner_levels)
            seeds = set(fold.exposes)  # the accumulators, or the epilogue's names for them once hoisted factors wrap the fold
            level.stmts = [m for m in level.stmts if not (isinstance(m, Init) and m.name in seeds)]
            edges.append(fold)
            level.exposed.update((name, fold) for name in fold.exposes)
            level.stmts.extend(trailing)
            continue
        if isinstance(stmt, Loop):
            # An OUTPUT SWEEP. Its reductions are terms of this level (hoisted just above); its
            # per-cell projection — the pure statements its stores read — is one more, a zero-axis
            # term evaluated over the sweep coordinate it declares, closed over the level like any
            # other. The loop keeps only its stores, which the boundary extracts as sweep specs.
            inner, cell = lift_body(stmt.body, (*axes, stmt.axis), inner_levels)
            edges.extend(inner)
            level.exposed.update((name, fold) for fold in inner for name in fold.exposes)
            writes = tuple(cell.writes)
            # A NESTED output sweep is a statement of this cell, not projection material: its own
            # recursion already reduced it to its stores, so it stays in the retained cell — at
            # its source position among the writes — while the projection term is formed from the
            # pure members alone.
            stores = {id(member) for member in cell if isinstance(member, (Write, Loop)) or guarded_store(member) is not None}
            pure = Body(tuple(member for member in cell if id(member) not in stores))
            defined = {name for member in pure for name in member.defines()}
            retained = Body(tuple(member for member in cell if id(member) in stores))
            results = tuple(dict.fromkeys(value for write in writes for value in write.values if value in defined))
            # A store's coordinates and guard read values of the cell too (a scatter's row): the term exposes them.
            guards = tuple(member.cond for member in cell if guarded_store(member) is not None)
            read = {
                name for expression in (*(index for write in writes for index in write.index), *guards) for name in expression.free_vars()
            }
            results += tuple(sorted((defined & (retained.ssa_uses | read)) - set(results)))
            if results:
                operands, lift = _close((), (), pure, results, (*axes, stmt.axis), inner_levels)
                term = Fold(operands=operands, lift=lift)
                edges.append(term)
                level.exposed.update((name, term) for name in term.exposes)
            level.stmts.append(stmt.replace(body=retained))
            continue
        nested = stmt.nested()
        if nested:
            lifted = tuple(lift_body(child, axes, inner_levels) for child in nested)
            if any(inner for inner, _ in lifted):
                # A reduce under a conditional is not a value the level can evaluate outright.
                # Leave it intact and let the raw-loop check report it.
                level.stmts.append(stmt)
                continue
            stmt = stmt.with_bodies(tuple(cell for _, cell in lifted))
        level.stmts.append(stmt)
    stmts = Body(tuple(level.stmts))
    rest = tuple(stmt for stmt in stmts if id(stmt) not in level.consumed)
    reads = {name for stmt in rest for name in Body((stmt,)).ssa_uses}
    keep = {id(stmt) for stmt in rest} | {id(stmt) for stmt in stmts.backward_cone(tuple(reads)).members}
    kept = tuple(stmt for stmt in stmts if id(stmt) in keep)
    # A statement or a sibling that moved under a reader stays at the level only while the level's
    # own KEPT statements still read it — a consumed statement the level keeps for another reader
    # reads its providers here too; otherwise its one position is under that reader.
    reads = {name for stmt in kept for name in Body((stmt,)).ssa_uses}
    remaining = tuple(edge for edge in edges if id(edge) not in level.drained or reads & set(edge.exposes))
    return remaining, Body(kept)


def fold_from_loop(loop: Loop) -> Fold:
    """Lift one PURE reduction from its explicit ``Accum`` statements."""
    fold, trailing = scan_from_loop(loop)
    if trailing:
        raise ValueError(f"reduce loop {loop.axis.name!r} carries per-step stores — a scan, not a pure reduction")
    return fold


def scan_from_loop(loop: Loop, axes: tuple = (), levels: tuple = ()) -> tuple[Fold, tuple[Write, ...]]:
    """Lift one reduction from its explicit ``Accum`` statements. A per-step ``Write`` makes it a
    SCAN: the store observes the carried state, so the fold gains an observer — a pure per-step
    tap binding fresh ``<state>__obs`` names — and each store returns rewritten to read the
    observed name. The rewritten stores ride the stream position after the node (the observed
    names are the fold's extra ``defines``), where boundary extraction claims them as ordinary
    ``OutputSpec``\\ s and reconstitution splices them back into the loop."""
    if loop.carries:
        return _carried_from_loop(loop, axes, levels)
    loop = _stamp_axes(loop)
    scope = (*axes, loop.axis)
    edges, body = lift_body(loop.body, scope, levels)
    accums = tuple(stmt for stmt in body if isinstance(stmt, Accum))
    if not accums:
        raise ValueError(f"reduce loop {loop.axis.name!r} has no Accum")
    if any(stmt.base is not None or stmt.dtype is not None for stmt in accums):
        raise ValueError(f"reduce loop {loop.axis.name!r} is not in canonical Loop IR")
    writes = tuple(stmt for stmt in body if isinstance(stmt, Write))
    write_ids = {id(stmt) for stmt in writes}
    names = tuple(stmt.name for stmt in accums)
    # Already separated by :func:`lift_body` — ``edges`` are the step's nested reductions, ``plain``
    # its statements. ``Fold.lower`` places each edge ahead of its reader, so the split preserves
    # evaluation order without the step ever having been a mixed sequence. A statement computing a
    # STORED value from the carried state — the rounding a narrowing store spells — observes the state
    # as the store does; it belongs to the observer below, not to the step.
    step = tuple(stmt for stmt in body if not isinstance(stmt, Accum) and id(stmt) not in write_ids)
    values = tuple(stmt.value.name for stmt in accums)  # one definition per node: every Accum sums a name
    observed, step = _observers(step, tuple(dict.fromkeys(value for stmt in writes for value in stmt.values)), names, loop.axis.name)
    # Every ``Load`` in the step becomes a SLAB — a term declaring the coordinates it indexes —
    # and a semiring step's product ARGUMENTS become operand edges too (:func:`_factor_products`):
    # a chain the step computes ahead of a product is a zero-axis cone. That is what makes a
    # semiring fold canonical BY CONSTRUCTION: the lift body is the products alone, so the bilinear
    # reading is a reading of the stored term and no later pass rewrites the tree into that form.
    # A load over COORDINATES is a slab; a data-dependent GATHER — an index reading a value the step
    # computes (the packed-pair table read by a decoded code) — is a statement of its cone: the value
    # it reads is not an axis, and a slab would declare it as one. A LOAD defines such a value too:
    # a code read from gmem indexes the next table read, and reading only the arithmetic
    # here left that index declared as a coordinate the kernel could hand no extent.
    slabs, plain = _slabs(step, levels[-1].carriers if levels else {}, _values_in_scope(levels))
    ops = tuple(stmt.op for stmt in accums)
    edges, plain, hoists = _factor_products(plain, values, ops, (*edges, *slabs), scope, levels, axes, hoist=not writes)
    # FORM the lift closed: a value the step reads from an enclosing level arrives as an operand;
    # the coordinates it reads outright (a mask's ``Select``) stay free — at the construction
    # site, which is the one that knows it is turning a Loop into a term.
    edges, lift = _close((loop.axis.name,), edges, plain, values, axes, levels)
    if not all(op.has_identity for op in ops):
        raise ValueError(f"reduce loop {loop.axis.name!r}: an Accum op without an identity is not a monoid ⊕")
    init, combine = tuple(op.identity for op in ops), Lambda.componentwise(ops, names)
    if not writes:
        fold = Fold(operands=edges, lift=lift, init=init, base=combine)
        return (_hoisted(fold, names, hoists, axes, levels) if hoists else fold), ()
    stored = tuple(dict.fromkeys(value for stmt in writes for value in stmt.values))
    observe = Lambda(
        params=(loop.axis.name, *names),
        body=Body((*observed, *(Assign(f"{value}__obs", Var(value)) for value in stored))),
        results=tuple(f"{value}__obs" for value in stored),
    )
    fold = Fold(operands=edges, lift=lift, init=init, base=combine, observe=observe)
    renamed = tuple(stmt.replace(values=tuple(f"{value}__obs" for value in stmt.values)) for stmt in writes)
    return fold, renamed


def _observers(step: tuple, stored: tuple[str, ...], names: tuple[str, ...], axis: str) -> tuple[tuple[Statement, ...], tuple]:
    """Split the step of a scan into the statements that compute its STORED values from the carried
    state — a per-step tap reading only the state, the axis and its own definitions, in step order —
    and the rest. A stored value that is the state itself needs no statement; one that reads anything
    the step computes is no observation of the state and is refused."""
    defined = {name: stmt for stmt in step for name in stmt.defines()}
    taken: dict[int, Statement] = {}
    pending = [value for value in stored if value not in names]
    while pending:
        value = pending.pop()
        stmt = defined.get(value)
        if stmt is None or isinstance(stmt, (Load, Pre, Loop)):
            raise ValueError(f"a per-step store may only observe the carried state {names}, not {value!r}")
        if id(stmt) in taken:
            continue
        taken[id(stmt)] = stmt
        pending.extend(dep for dep in stmt.deps() if dep not in names and dep != axis)
    observed = tuple(stmt for stmt in step if id(stmt) in taken)
    return observed, tuple(stmt for stmt in step if id(stmt) not in taken)


def _values_in_scope(levels: tuple) -> frozenset[str]:
    """The names the enclosing levels define — their statements' results and the terms they expose — which a
    step below may read as VALUES. Any other free name of a step is a coordinate."""
    return frozenset(name for level in levels for stmt in level.stmts for name in stmt.defines()) | frozenset(
        name for level in levels for name in level.exposed
    )


def _slabs(step: tuple, carriers: dict, known: frozenset[str]) -> tuple[tuple[Fold, ...], Body]:
    """The step's reads as terms and what remains of it: every ``Load`` over COORDINATES becomes a
    SLAB — a term declaring the coordinates it indexes — and every ``Pre`` a CARRIER READ, a slab
    over the carried state naming the axis of the loop carrying it. A data-dependent GATHER — an
    index reading a VALUE, one the step computes (the packed-pair table read by a decoded code) or
    one an enclosing level holds (``known``: an expert's weights picked by a routed index) — is a
    statement of its cone: the value it reads is not an axis, and a slab would declare it as one.
    A LOAD defines such a value too: a code read from gmem indexes the next table read, and reading
    only the arithmetic here left that index declared as a coordinate the kernel could hand no
    extent."""
    defined = {name for stmt in step for name in stmt.defines()} | known
    gathers = {id(stmt) for stmt in step if isinstance(stmt, Load) and any(expression.free_vars() & defined for expression in stmt.index)}
    slabs = tuple(
        Fold.carrier_read(stmt, carriers[stmt.carrier]) if isinstance(stmt, Pre) else Fold.slab(stmt)
        for stmt in step
        if isinstance(stmt, (Load, Pre)) and id(stmt) not in gathers
    )
    return slabs, Body(stmt for stmt in step if not isinstance(stmt, (Load, Pre)) or id(stmt) in gathers)


def _carried_from_loop(loop: Loop, axes: tuple, levels: tuple) -> tuple[Fold, tuple[Write, ...]]:
    """Lift a loop that CARRIES A STATE into a fold whose ⊕ is the action ``next``: the cell loops
    are the fold's ``cells``, each ``Carry`` a state whose lift result is its value and whose seed
    is the fold's ``init``, a ``Pre`` a carrier-read slab the way a ``Load`` is a slab. A per-step
    ``Write`` streams a value of the step — one the step defines, or one an operand exposes (the
    state a step read) — and that value enters the lift as a fresh ``<value>__obs`` copy, so every
    streamed value is a result of the fold's own and the store, rewritten to read it, rides the
    stream position after the node like a scan's."""
    loop = _stamp_axes(loop)
    cells: list[Axis] = []

    def flatten(body: Body) -> list[Statement]:
        # The chain of loops over the cells, from the carrying loop down to the level that defines
        # the state; a free loop beside that chain is an output SWEEP of the step (the delta rule's
        # corrected values, over the chunk's rows beside the state's), left as one.
        if any(isinstance(stmt, Carry) for stmt in body):
            return list(body)
        chain = [index for index, stmt in enumerate(body) if isinstance(stmt, Loop) and not stmt.is_reduce and stmt.body.carries]
        if len(chain) != 1:
            return list(body)
        (index,) = chain
        cells.append(body[index].axis)
        return [*body[:index], *flatten(body[index].body), *body[index + 1 :]]

    flat = Body(flatten(loop.body))
    defined = tuple(stmt for stmt in flat if isinstance(stmt, Carry))
    if any(
        carry.cells != defined[0].cells or [isinstance(e, Var) for e in carry.index] != [isinstance(e, Var) for e in defined[0].index]
        for carry in defined
    ):
        raise ValueError(f"loop {loop.axis.name!r}: every carried state is one value per cell over the same cell axes")
    # A position where normalization dropped a size-one axis is ``0`` in every index over the
    # state; the term's cells are the axes alone, so the reads and the definition drop it too.
    unit = tuple(position for position, e in enumerate(defined[0].index) if not isinstance(e, Var))

    def strip(stmt: Statement) -> Statement:
        if isinstance(stmt, Carry) or (isinstance(stmt, Pre) and stmt.carrier in loop.carries):
            return stmt.replace(index=tuple(e for position, e in enumerate(stmt.index) if position not in unit))
        return stmt

    if unit:
        flat = flat.map(strip)
    carriers = {**(levels[-1].carriers if levels else {}), **dict.fromkeys(loop.carries, loop.axis.name)}
    scope = (*axes, loop.axis, *cells)
    edges, body = lift_body(flat, scope, levels, carriers=carriers)
    carried = tuple(stmt for stmt in body if isinstance(stmt, Carry))
    writes = tuple(stmt for stmt in body if isinstance(stmt, Write))
    # An output sweep of the step keeps its stores alone (:func:`lift_body`); the values it
    # writes are results of terms evaluated over the sweep, which the fold passes through as
    # results of its own so the boundary can bind them.
    sweeps = tuple(stmt for stmt in body if isinstance(stmt, Loop))
    swept = tuple(dict.fromkeys(value for sweep in sweeps for stmt in sweep.body.iter_of_type(Write) for value in stmt.values))
    skip = {id(stmt) for stmt in (*carried, *writes, *sweeps)}
    slabs, plain = _slabs(tuple(stmt for stmt in body if id(stmt) not in skip), carriers, _values_in_scope(levels))
    observed = {value: f"{value}__obs" for stmt in writes for value in stmt.values}
    plain = Body((*plain, *(Assign(fresh, Var(value)) for value, fresh in observed.items())))
    results = (*(carry.value for carry in carried), *observed.values(), *swept)
    edges, lift = _close((loop.axis.name,), (*edges, *slabs), plain, results, scope, levels)
    names = tuple(carry.name for carry in carried)
    fold = Fold(
        operands=edges,
        lift=lift,
        init=tuple(carry.seed for carry in carried),
        base=Lambda.componentwise(("next",) * len(names), names),
        cells=carried[0].cells,
    )
    return fold, (*(stmt.replace(values=tuple(observed[value] for value in stmt.values)) for stmt in writes), *sweeps)


def carrying_loops(body: Body, outer: tuple[Axis, ...] = ()) -> list[tuple[Loop, tuple[Axis, ...]]]:
    """Every loop of ``body`` that carries a state, with the free loops outside it."""
    out = []
    for stmt in Body.coerce(body):
        if isinstance(stmt, Loop) and stmt.carries:
            out.append((stmt, outer))
        elif isinstance(stmt, Loop):
            out.extend(carrying_loops(stmt.body, (*outer, stmt.axis)))
    return out


def stored_states(loop: Loop, outer: tuple[Axis, ...]) -> dict[str, tuple[Carry, Write]]:
    """The states ``loop`` carries whose next value the term stores itself, every step: a plain store of the
    ``Carry``'s value (or an exact copy of it a normalization step named) at the step, the free coordinates
    outside the loop and the cell, with ``0`` where a size-one dim pins the coordinate — the per-step store the
    roll gives a rolled recurrence's state. Such a state needs no buffer of its own: that output holds every
    step, so the lagged read is a read of it one step back (:func:`states_as_buffers`)."""
    fixed = {loop.axis.name, *(axis.name for axis in outer)}
    stmts = tuple(loop.body.iter())
    carries = {s.value: s for s in stmts if isinstance(s, Carry) and s.name in loop.carries}
    copies = {s.name: s.value.name for s in stmts if isinstance(s, Assign) and isinstance(s.value, Var) and s.dtype in (None, F32)}
    out: dict[str, tuple[Carry, Write]] = {}
    for stmt in stmts:
        if not (isinstance(stmt, Write) and stmt.is_scalar and not stmt.atomic):
            continue
        value = stmt.value
        while value not in carries and value in copies:
            value = copies[value]
        if value not in carries:
            continue
        carry = carries[value]
        cells = {cell.name for cell in carry.index if isinstance(cell, Var)}
        names = [expression.name for expression in stmt.index if isinstance(expression, Var)]
        plain = all(isinstance(expression, Var) or expression == Literal(0, "int") for expression in stmt.index)
        if plain and len(set(names)) == len(names) and set(names) == fixed | cells:
            out.setdefault(carry.name, (carry, stmt))
    return out


def states_as_buffers(body: Body, prefix: str, inputs: Mapping[str, Tensor]) -> tuple[Body, tuple[Axis, ...], dict[str, tuple]]:
    """``body`` with every carried state spelled as a STATE BUFFER — ``(body, serial axes,
    buffer shapes)`` — the form a serial launch axis realizes (:attr:`Placement.serial`): the
    classic schedule's realization of a carried state, taken at its fork from the carrying
    kernel's own Loop IR (:func:`lift_serial`).

    The loop that carries a state becomes the kernel's time: one launch per step, in order. The
    buffer keeps every step, indexed by the step, the free axes outside the loop and the cell. A
    ``Pre`` read is a load one step back, the seed where there is no step before the first; the
    ``Carry`` is a store of its value at this step. A state the term already stores that way
    (:func:`stored_states`) keeps that output as its buffer: the read is a read of it one step
    back, the ``Carry`` stores nothing twice, and ``shapes`` names no buffer for it. Nothing of the
    step's own algebra changes, so the rest of the nest lifts as it always did. ``inputs`` are the
    kernel's input tensors, the seeds among them (:func:`seed_index`).
    """
    serial: list[Axis] = []
    shapes: dict[str, tuple] = {}

    def read(stmt: Pre, time: Axis, outer: tuple[Axis, ...], seeds: dict[str, float | str], stored: dict) -> tuple[Statement, ...]:
        step = Var(time.name)
        after_first = BinaryExpression(">", step, Literal(0, "int"))
        previous = TernaryExpression(after_first, step - Literal(1, "int"), Literal(0, "int"))
        if stmt.carrier in stored:
            carry, write = stored[stmt.carrier]
            cells = {cell.name: at for cell, at in zip(carry.index, stmt.index, strict=True) if isinstance(cell, Var)}
            buffer, index = write.output, tuple(expression.substitute({time.name: previous, **cells}) for expression in write.index)
        else:
            buffer, index = f"{prefix}__{stmt.carrier}", (previous, *(Var(axis.name) for axis in outer), *stmt.index)
        held, seed = f"{stmt.name}__prev", f"{stmt.name}__seed"
        start = seeds[stmt.carrier]
        # The seed is a buffer of the state's shape (the tensor an unrolled loop started from) or a constant.
        if isinstance(start, str):
            cell = (*(Var(axis.name) for axis in outer), *stmt.index)
            seeded: Statement = Load(names=(seed,), input=start, index=seed_index(cell, inputs[start].shape))
        else:
            seeded = Let(name=seed, value=Literal(start))
        return (
            Load(names=(held,), input=buffer, index=index),
            seeded,
            Select(name=stmt.name, branches=(SelectBranch(held, after_first), SelectBranch(seed, Literal(True, "bool")))),
        )

    def define(stmt: Carry, time: Axis, outer: tuple[Axis, ...], stored: dict) -> tuple[Statement, ...]:
        if stmt.name in stored:
            return ()
        index = (Var(time.name), *(Var(axis.name) for axis in outer), *stmt.index)
        return (Write(output=f"{prefix}__{stmt.name}", index=index, values=(stmt.value,)),)

    def walk(stmts: Body, outer: tuple[Axis, ...], time: Axis | None, seeds: dict[str, float | str], stored: dict) -> Body:
        out: list[Statement] = []
        for stmt in stmts:
            if isinstance(stmt, Pre):
                if stmt.carrier not in seeds:
                    raise ValueError(f"a read of {stmt.carrier!r} outside the loop that carries it has no launch to run in")
                out.extend(read(stmt, time, outer, seeds, stored))
            elif isinstance(stmt, Carry):
                out.extend(define(stmt, time, outer, stored))
            elif isinstance(stmt, Loop) and stmt.carries:
                if time is not None:
                    raise ValueError(f"loop {stmt.axis.name!r} carries a state inside loop {time.name!r}, which carries one too")
                serial.append(stmt.axis)
                carried = {carry.name: carry.seed for carry in stmt.body.carries if carry.name in stmt.carries}
                stored = stored_states(stmt, outer)
                buffers = {name: cells for name, cells in stmt.carries.items() if name not in stored}
                shapes.update((f"{prefix}__{name}", (stmt.axis, *outer, *cells)) for name, cells in buffers.items())
                out.append(stmt.replace(body=walk(stmt.body, outer, stmt.axis, carried, stored)))
            elif isinstance(stmt, Loop) and time is None:
                out.append(stmt.replace(body=walk(stmt.body, (*outer, stmt.axis), time, seeds, stored)))
            elif stmt.nested():
                out.append(stmt.with_bodies(tuple(walk(child, outer, time, seeds, stored) for child in stmt.nested())))
            else:
                out.append(stmt)
        return Body(out)

    return walk(Body.coerce(body), (), None, {}, {}), tuple(serial), shapes


def _peel(body: Body) -> tuple[list, list[Statement]]:
    """Peel the outer parallel loop chain into placement axes."""
    axes = []
    prefix: list[Statement] = []
    current = list(body)
    while True:
        index = 0
        while index < len(current) and isinstance(current[index], (Load, Assign, Init, Let, Select)):
            index += 1
        head, rest = current[:index], current[index:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + current
        prefix.extend(head)
        axes.append(rest[0].axis)
        current = list(rest[0].body)


def _raw_loops(body: Body) -> list[Loop]:
    """Return every Loop that survived total reduction lifting.

    Takes a ``Body``: the lift hands back statements and terms already separated, so this walks a
    statement sequence with statement vocabulary and never meets a term."""
    out = []
    for stmt in body:
        if isinstance(stmt, Loop):
            out.append(stmt)
        for nested in stmt.nested():
            out.extend(_raw_loops(nested))
    return out


def _root_results(body: Body) -> tuple[str, ...]:
    """The names a root projection passes to its consumer — its body's last definition.

    Spelled here, at the one construction site that needs it, rather than as a former's default.
    """
    for stmt in reversed(tuple(body)):
        names = stmt.defines()
        if names:
            return (names[-1],)
    return ()


def lift_loop_op(op: LoopOp, *, name: str = "", body: Body | None = None, serial: tuple[Axis, ...] = ()) -> TileOp:
    """Peel free axes and lift the complete remaining nest as one Fold tree.

    ``body`` lifts a rewritten spelling of ``op``'s program instead of the stored one. Constructing
    a ``LoopOp`` around such a spelling would not preserve it — Loop-IR normalization is what drops
    a size-one free axis, so the one caller that restores one (``_row``) hands its body here.
    ``serial`` names the peeled axes that are the kernel's time (:func:`states_as_buffers`).
    """
    with binding_extents(name for tensor in op.inputs.values() for dim in tensor.shape for name in dim.expression.free_vars()):
        return _lift_loop_op(op, name, body, serial)


def _lift_loop_op(op: LoopOp, name: str, body, serial: tuple[Axis, ...]) -> TileOp:
    free, cell = _peel(op.body if body is None else Body.coerce(body))
    if any(axis not in free for axis in serial):
        raise ValueError(f"a loop that carries a state must sit outside every reduction: {[axis.name for axis in serial]}")
    edges, stmts = lift_body(cell, tuple(free))
    split = extract_output_specs(stmts)
    if split is None:
        raise ValueError("Loop IR effects cannot be represented as output specifications")
    body, output_specs = split
    raw = _raw_loops(Body(body))
    if raw:
        axes = ", ".join(inner.axis.name for inner in raw)
        raise ValueError(f"total lift left raw inner loops: {axes}")
    # The root term, constructed DIRECTLY: ``lift_body`` already handed back its operands and its
    # statements apart, so there is nothing for a former to separate, dedup or name. The lift binds
    # one param per operand result component, positionally, and reads the grid axes free.
    # It exposes what the kernel stores — its body's last definition, or with no body of its own
    # the operand values the boundary writes — so a wrapper over one operand is the identity
    # projection normalization dissolves, rather than a permanent layer over every bare kernel.
    results = _root_results(Body(body)) or tuple(dict.fromkeys(value for spec in output_specs for value in spec.write.values))
    edges, lift = _close((), edges, Body(body), results, tuple(free), ())
    # The kernel's axis table: the free axes and every loop the nest bound (reduce, sweep), by
    # name — the term names them, the kernel holds their extents.
    axes = {axis.name: axis for axis in (*free, *(loop.axis for loop in Body.coerce(cell).loops))}
    axes.update((axis.name, axis) for spec in output_specs for axis in spec.sweep)  # a renamed sibling sweep
    return TileOp(
        op=Fold(operands=edges, lift=lift),
        name=name,
        place=Placement(free=tuple(axis for axis in free if axis not in serial), serial=serial),
        axes=tuple(axes.values()),
        inputs=dict(op.inputs),
        output_specs=output_specs,
    )


def seed_index(cell: tuple[Expression, ...], shape: tuple) -> tuple[Expression, ...]:
    """A seed read at ``cell``, one coordinate per dim of the seed tensor (``shape``). The state's cells
    are its axes alone: a size-one dim, which normalization pins to ``0`` in every index over the state,
    holds no cell (``_carried_from_loop``). The seed keeps that dim, so it reads ``0`` there."""
    if len(shape) == len(cell):
        return cell
    units = [dim for dim, extent in enumerate(shape) if extent == 1]
    if len(shape) - len(cell) != len(units):
        raise ValueError(f"a seed of shape {shape} is not its state's cells {len(cell)} with size-one dims beside them")
    rest = iter(cell)
    return tuple(Literal(0, "int") if dim in units else next(rest) for dim in range(len(shape)))


def lift_serial(op: LoopOp, *, name: str, prefix: str) -> tuple[TileOp, dict[str, tuple]]:
    """A kernel that carries a state lifted as a serial kernel: its carried states become state
    buffers (:func:`states_as_buffers`, named under ``prefix``) and the loop that carries them the
    kernel's time. Returns the tile and the state buffers' shapes."""
    body, serial, shapes = states_as_buffers(op.body, prefix, op.inputs)
    return lift_loop_op(op, name=name, body=body, serial=serial), shapes


def serial_form(tile: TileOp, prefix: str) -> TileOp:
    """A kernel that carries a state as the classic schedule realizes it: its Loop IR lifted again
    with every carried state a buffer the kernel owns — the port the lift added, named under
    ``prefix`` — and the carrying loop the kernel's serial launch axis (``lift_serial``): one
    launch per step, each reading the previous launch's stores. The term the classic tiers then
    tile is the STEP, a projection over ordinary slabs; the register schedule reads the carrying
    fold itself."""
    formed, _ = lift_serial(LoopOp(body=tile.loop_body, inputs=tile.inputs), name=tile.name, prefix=prefix)
    # The kernel's own knobs and its BOUND I/O: a standalone loop op seeds placeholder tensors, and
    # the schedule reads operand dtypes, output shapes and the ``with_io`` identity off these maps.
    return replace(formed, knobs=tile.knobs, inputs=tile.inputs, outputs=tile.outputs)


__all__ = [
    "carrying_loops",
    "fold_from_loop",
    "lift_body",
    "lift_loop_op",
    "lift_serial",
    "seed_index",
    "serial_form",
    "states_as_buffers",
    "stored_states",
]
