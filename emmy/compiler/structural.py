"""Structural identity protocol — one convention for every type whose
instances we want to compare or dedup by their structure rather than by
Python identity / field equality.

Implementers return a hex sha256 digest folded over the bits of state
that affect downstream behavior (codegen output, dataflow semantics,
compilation result). Bits that are name-only / advisory / ambient I/O
are deliberately excluded — see each implementer's docstring for the
exact include/exclude list.

``Dim``, ``Graph``, ``Body`` and ``Context`` implement it.

``Op.identity_key`` layers on these: one lattice over the op's canonical
Loop-IR body, typing its buffer roles and folding in the knob dict for the
deploy join (the deploy identity (``identity_key(with_io=True)``)) and the tuning / cubin caches
(``identity_key(with_io=True, with_knobs=True)``).

The cache layer in the autotuning loop keys candidates by these digests,
so adding a field to an implementer is an explicit decision: include it
in the digest only if it changes generated code or dataflow semantics.

``digest(*parts)`` is the canonical fold helper — pass a mix of strings,
ints, bytes, and pre-computed child digests; the helper canonicalizes
via ``repr`` and returns a hex sha256 string. Composite implementers use
it to fold child digests with their own discriminating fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Structural(Protocol):
    """Anything whose structural identity is comparable as a hex digest.

    Two instances that should be treated as equivalent for caching /
    dedup purposes return the same string from :meth:`structural_key`;
    two instances that differ in any codegen- or dataflow-relevant way
    return different strings.

    **Inherit it and you get the default**: :func:`form` over the instance, which is the right
    answer for any frozen dataclass whose fields ARE its identity (every ``Stmt``, which holds this
    default as a class attribute instead of inheriting the protocol's metaclass). Override it
    when the type owns a canonicalization the field walk cannot know — ``Fold``'s α-invariance,
    ``Body``'s normalize-and-collapse, ``Graph``'s Merkle walk over its nodes.

    :func:`form` reads that distinction directly (``type(x).structural_key is
    Structural.structural_key``) and delegates only to an override, which is what keeps the
    default from calling itself forever. There is no marker to set and none to forget.
    """

    def structural_key(self) -> str:
        return digest(form(self))


def instance_memo(obj, slot: str) -> dict:
    """The named per-instance memo table riding an IMMUTABLE object: a derived read caches on the
    term it derives from (the table is created on first use via ``object.__setattr__``, so frozen
    dataclasses take it), and the owner's ``__getstate__`` strips it so no cache — an id-keyed one
    especially — crosses a process boundary. A memo holds ONLY values derivable from the object;
    never decisions, never mutable policy.

    DEPRECATED, and closed to new callers. The undeclared attribute this stashes into is what
    STYLE.md's "Derived values are first-class members" rule forbids: it is invisible to readers of
    the class, dodges the pickling rules, and smears one type's functionality across whichever
    module caches for it. A derived read belongs on the type that owns it — a ``cached_property``
    returning the whole keyed family, a ``cached_method`` for one keyed by an argument, or a field
    computed in ``__post_init__``. The remaining callers (the path spelling tables, the normalize
    fixpoint stamp) are to be converted."""
    table = obj.__dict__.get(slot)
    if table is None:
        table = {}
        object.__setattr__(obj, slot, table)
    return table


def form(value: object) -> object:
    """The STRUCTURAL rendering of ``value`` — a nested ``(class name, *fields)`` tuple.

    The one alternative to ``repr`` for anything that enters an identity: how a statement PRINTS is a presentation
    choice, and a ``__repr__`` edit must not re-key every stored golden.

    Primitives and ``None`` pass through, so the leaves stay comparable by value; tuples and lists render
    elementwise; sets render SORTED, because their iteration order is not stable. Any other value renders by the
    rule of its class, chosen once per class, the first of:

    - the class supplies its own form in ``structural_form()``: the one way a class that is not a dataclass, or
      whose form is not its fields, enters an identity (``Body`` renders elementwise, as a tuple does). A class
      with fields calls :func:`record_form`, so it is keyed the way a dataclass is;
    - the class OVERRIDES :meth:`Structural.structural_key`: ``(class, its key)`` — it owns a canonicalization
      this walk cannot know (``Graph``'s Merkle walk), and a type that merely inherits the default must fall
      through, or the default would call itself forever;
    - a dataclass: :func:`record_form` over its fields in declaration order, so a new node kind is covered the day
      it is added. ``DataType`` takes this route, so its numpy dtype rides along as a leaf: stable, since equal
      dtypes render equal, just noisier than the name alone;
    - a ``str`` ``name``: ``(class, name)`` — ``ElementwiseImpl``, whose name IS its identity;
    - anything else RAISES. There is no ``repr`` fallback: a silent one would hide the moment a new field type
      started keying kernels on its ``__repr__``. A type that lands here supplies ``structural_form``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(form(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((form(v) for v in value), key=repr))
    cls = type(value)
    rule = _RULES.get(cls)
    if rule is None:
        rule = _RULES[cls] = _rule_of(cls)
    return rule(value)


_RULES: dict[type, Callable[[object], object]] = {}


def _rule_of(cls: type) -> Callable[[object], object]:
    if (own := getattr(cls, "structural_form", None)) is not None:
        return own
    if getattr(cls, "structural_key", Structural.structural_key) is not Structural.structural_key:
        return lambda value: (cls.__name__, value.structural_key())
    if dataclasses.is_dataclass(cls):
        declared = [(field.name, field.default) for field in dataclasses.fields(cls)]
        return lambda value: record_form(cls.__name__, [(getattr(value, name), default) for name, default in declared])
    return _named


def _named(value: object) -> tuple:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return (type(value).__name__, name)
    raise TypeError(
        f"structural.form has no rule for {type(value).__name__}, so this value cannot enter an identity. Give the "
        "class a structural_form that renders the part of it that IS the identity. Do not fall back to repr."
    )


def record_form(name: str, fields: Sequence[tuple[object, object]]) -> tuple:
    """``(name, *fields)`` less the TRAILING fields left at their default. ``fields`` pairs each value with its
    default, ``dataclasses.MISSING`` where it has none.

    A field a value does not use says nothing about what that value IS, and rendering it anyway means a class that
    GROWS one moves the identity of every value in the tree — every kernel, every recorded row, every corpus case —
    for a field none of them set. Trailing only, so the rendering stays positional: a default in the middle still
    holds its place.
    """
    kept = len(fields)
    while kept:
        value, default = fields[kept - 1]
        if default is dataclasses.MISSING or value != default:
            break
        kept -= 1
    return (name, *(form(value) for value, _ in fields[:kept]))


def digest(*parts: object) -> str:
    """Fold ``parts`` into a hex sha256 digest. Each part is rendered
    via ``repr`` and joined; pass child digests (already strings),
    primitive fields (ints, tuples, names), or any ``repr``-stable
    object. Order is significant — callers control canonicalization."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()
