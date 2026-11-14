"""``Lambda`` — the ONE binder kind, shared by every IR level.

A lambda is a PURE term: explicit binders over the reused stmt vocabulary, an A-normal-form
let-chain in ``body``, the returned defs named by ``results``. It binds; it never executes.
Nothing here is a :class:`~emmy.compiler.ir.stmt.Statement` — see ``ir/ARCHITECTURE.md`` for the
invariant and for how a pure term reaches a statement position (it is RENDERED to stmts, never
spliced in as one).
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Var, apply
from emmy.compiler.ir.stmt import (
    Assign,
    Body,
    exposed_defines,
    normalize_lambda_body,
    one_operation_per_definition,
    pretty_body,
)
from emmy.compiler.wire import Wire
from emmy.emmy_ir import rename_lambda


@dataclass(frozen=True)
class Lambda(Wire):
    """Explicit binders over the REUSED stmt vocabulary — the ONE binder kind, common to every IR
    level. Not a second expression language: ``body`` is a :class:`Body` of PURE stmts only
    (A-normal form ≙ a let-chain), ``params`` the binders, ``results`` the returned defs
    (replacing every ``out`` / last-def convention).

    ``__post_init__`` validates the LOCAL formation invariant: every body stmt passes the
    :attr:`Statement.pure` trait (declared on the interface, conservative ``False`` default —
    ``Load`` / ``Assign`` and the structural nodes opt in; ``Accum`` / ``Write`` / ``Init`` /
    ``Loop`` never do; no isinstance whitelist, so a new statement type is excluded until it declares
    itself), every result is defined, and every value or coordinate read is bound. Scope-aware
    ``Body.free`` supplies the construction check, ``normalize_lambda_body`` the statement order and the canonical
    value spellings.

    α-invariance is CANONICAL RENUMBERING (the existing rename machinery), not de Bruijn:
    :meth:`canonical` renumbers params (``_p0…``) and internal defs (``_v0…``) in walk order,
    leaving free names untouched; :meth:`alpha_eq` compares canonical forms."""

    params: tuple[str, ...]
    body: Body
    results: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple):
            object.__setattr__(self, "params", tuple(self.params))
        body = normalize_lambda_body(self.body, self.params)
        if not isinstance(self.body, Body) or body != self.body:
            object.__setattr__(self, "body", body)
        if not isinstance(self.results, tuple):
            object.__setattr__(self, "results", tuple(self.results))
        impure = [type(s).__name__ for s in self.body if not s.pure]
        if impure:
            raise ValueError(f"Lambda body must be pure — impure statement type(s): {impure}")
        defined = set(self.params) | self.body.defined
        missing = [r for r in self.results if r not in defined]
        if missing:
            raise ValueError(f"Lambda results {missing} are not defined by the body or params")
        # CLOSED. A lambda is a function, so every name its body reads — a value or an index
        # coordinate, the same ``Var`` either way — is a param or one of its own defs. Which params
        # are coordinates is the binder's knowledge (a ``Fold`` reads them as its free coordinates
        # past the operand binding); a bare lambda cannot tell. :meth:`closing` FORMS a closed
        # lambda; this only refuses.
        free = self.body.free - defined
        if free:
            raise ValueError(
                f"Lambda body reads {sorted(free)} it does not bind. Pass them as params — "
                f"Lambda.closing(params, body, results) appends whatever the body still reads."
            )

    @classmethod
    def closing(cls, params: tuple[str, ...], body, results: tuple) -> Lambda:
        """Build a CLOSED lambda from its parts — the one former.

        A term is a function, so it has no free names. The caller supplies the params it knows —
        a fold's iteration var, the names its operand edges bind — and every remaining name the
        body reads, an index coordinate included, is appended as a TRAILING param. Trailing, never
        interleaved: the operand correspondence is the param PREFIX, so appending leaves every
        positional read of it intact.

        Built in ONE step: the body is normalized here, so the residual is computed against the
        same body the lambda stores rather than against a throwaway built open and rebuilt.

        Callers form; :meth:`__post_init__` refuses. They stay separate because a constructor that
        repaired its own input would enforce nothing."""
        body = normalize_lambda_body(body, tuple(params))
        bound = set(params) | body.defined
        # Reads the body does not define, plus any RESULT it does not define either: a write may
        # pass an enclosing value straight through (``o[j] = acc`` over an already-reduced
        # accumulator), and that result has no def to name, so it binds as a param like any read.
        residual = body.free | set(results)
        return cls(params=(*params, *sorted(residual - bound)), body=body, results=tuple(results))

    @classmethod
    def componentwise(cls, ops, names: tuple[str, ...]) -> Lambda:
        """The componentwise binary program — each result its own ⊕ over its two operands,
        ``nᵢ = ⊕ᵢ(nᵢ, nᵢ__o)``: ``S × S → S`` with the second operand spelled ``<n>__o``, a plain
        fold's combine, the shape :meth:`components` reads back."""
        other = tuple(f"{name}__o" for name in names)
        body = Body(tuple(Assign(name, apply(op, name, second)) for name, op, second in zip(names, ops, other, strict=True)))
        return cls(params=(*names, *other), body=body, results=tuple(names))

    def components(self) -> tuple[ElementwiseImpl, ...] | None:
        """The per-result ops when this program is componentwise — every result ``rᵢ = ⊕ᵢ(pᵢ, pₙ₊ᵢ)``
        on its own, in either argument order for a commutative ⊕ — else ``None``: the planar-vs-
        twisted reading of a fold's combine (a cross-component read or a rescale temp fails it)."""
        n = len(self.results)
        if len(self.body) != n or len(self.params) != 2 * n:
            return None
        definitions = self.body.definitions
        ops = []
        for index, result in enumerate(self.results):
            stmt, pair = definitions.get(result), (self.params[index], self.params[n + index])
            op = stmt.value.operation if isinstance(stmt, Assign) else None
            operands = tuple(operand.name if isinstance(operand, Var) else None for operand in stmt.value.operands) if op else ()
            if op is None or (operands != pair and not (op.commutative and operands == pair[::-1])):
                return None
            ops.append(op)
        return tuple(ops)

    @property
    def defined(self) -> frozenset[str]:
        """Every name this lambda binds — params plus every def its body exposes (deep)."""
        return frozenset(self.params) | self.body.defined

    def __getstate__(self):
        """Pickle the stored fields only — memoized reads recompute after transport."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    def cone(self, name: str) -> Lambda:
        """The closed cone of one definition — the statements of the body ``name`` depends on, as a
        lambda over what they read (a param names itself: an empty body returning it). What a
        reader asks of one result without the rest of the body: the score a fold's lift computes,
        the value one component of a projection denotes. Params in :meth:`closing`'s order.

        A PARAM NAMES ITSELF, and that has to hold for the reads inside the cone too. A stored
        combine is not SSA in its states: it takes ``acc`` in as a param and writes ``acc`` back on
        the way out to spell its result. :meth:`Body.backward_cone` resolves by name over a body it
        documents as SSA, so a read of the INCOMING ``acc`` resolves forward to that trailing write
        and drags it into the cone — which is how a weight cone came to carry the pivot's own state
        advance, and the kernel to read an uninitialized accumulator. Hiding the re-bindings makes
        the walk see the binder's scoping instead: under a param's spelling the body reads the
        value that came in, so the read lands in ``external_reads`` and closes back as that param.
        """
        if name in self.params:
            return Lambda.closing((), Body(()), (name,))
        params = set(self.params)
        visible = Body(tuple(s for s in self.body if not (exposed_defines(s) & params) or name in exposed_defines(s)))
        members = tuple(visible.backward_cone((name,)).members)
        return Lambda.closing((), Body(members), (name,))

    def rename(self, names) -> Lambda:
        """α-rename params, body and results through one map — a mapping or a callable — the
        lockstep every consumer of a stored lambda applies: the fold's rewrite handler, the
        combine's state rename, a recipe instantiated onto a term's names. The rename runs in Rust
        over the whole lambda; one that keeps every name apart keeps this lambda's formation
        checks true, so its result is formed without running them again."""
        params, body, results, injective = rename_lambda(self.params, self.body, self.results, names)
        if not injective:
            return Lambda(params=params, body=body, results=results)
        formed = object.__new__(Lambda)
        object.__setattr__(formed, "params", params)
        object.__setattr__(formed, "body", body)
        object.__setattr__(formed, "results", results)
        return formed

    def canonical(self) -> Lambda:
        """The α-canonical form: every definition one operation (a result stays a statement, as a reader outside the
        body reads it), then params renumbered to ``_p0…`` and internal defs to ``_v0…`` in walk order; free names (and
        float results) pass through unchanged. Deterministic and blind to whether a program was spelled as a chain of
        named temps or as one tree, up to the normal form's single-reader guard (a sum two statements read stays a
        definition), so two lambdas computing one thing have EQUAL canonical forms: the α-invariant equality/hash
        substrate (:meth:`alpha_eq`)."""
        split = Lambda(params=self.params, body=one_operation_per_definition(self.body, frozenset(self.results)), results=self.results)
        mapping = {p: f"_p{i}" for i, p in enumerate(self.params)}
        n = 0
        for s in split.body.iter():
            for d in s.defines():
                if d not in mapping:
                    mapping[d] = f"_v{n}"
                    n += 1
        return split.rename(mapping)

    def alpha_eq(self, other: Lambda) -> bool:
        """α-invariant equality — canonical forms compared structurally."""
        return isinstance(other, Lambda) and self.canonical() == other.canonical()

    def pretty(self, indent: str = "") -> list[str]:
        rs = ", ".join(self.results)
        head = f"{indent}λ({', '.join(self.params)}) -> ({rs})"
        return [head, *pretty_body(self.body, indent + "    ")]
