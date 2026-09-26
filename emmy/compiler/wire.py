"""One wire for every object a golden or a tune DB row stores.

A class that lives on the wire mixes in :class:`Wire`. A dataclass gets its wire for free: its init fields, minus
the ones it names in ``wire_skip``, written by their annotations. A field whose annotation names one wire class
holds that class's payload bare; a field whose annotation is a base class, a union of classes or ``Any`` holds
``{tag: payload}``, so the reader knows which class to build; a field at its default is omitted; a field annotated
``dict`` is an opaque payload kept as it is (a program pool's entries). A class whose wire is not its fields — a dim,
a tensor, a body, a graph, a leaf type — overrides ``to_wire`` / ``from_wire``. ``wire_tag`` names the class on the
wire, the class name by default; every wire class registers its tag when it is defined, so a tagged payload decodes
without anyone listing the classes. The walker refuses an unknown or missing key by path.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, fields
from types import UnionType
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

_SCALARS = (bool, int, float, str)
_REGISTRY: dict[str, type] = {}
_TAG_OF: dict[type, str] = {}
#: The names an annotation may spell: every wire class, plus the aliases modules register (``Expr``).
_NAMESPACE: dict[str, object] = {}
_HINTS: dict[type, dict[str, object]] = {}


class Wire:
    """What any wire object can do: write itself as YAML-safe data, and come back from it."""

    #: The name this class goes by on the wire where the reader cannot tell it from the field's annotation.
    wire_tag: ClassVar[str | None] = None
    #: Init fields that never travel: runtime state a constructor accepts but the wire does not define.
    wire_skip: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        tag = cls.__dict__.get("wire_tag") or cls.__name__
        if tag in _REGISTRY and _REGISTRY[tag] is not cls:
            other = _REGISTRY[tag]
            raise TypeError(f"wire tag {tag!r} is claimed by both {other.__module__}.{other.__name__} and {cls.__module__}.{cls.__name__}")
        _REGISTRY[tag] = cls
        _TAG_OF[cls] = tag
        _NAMESPACE[cls.__name__] = cls

    @classmethod
    def from_wire(cls, value: object, where: str | None = None):
        """``value``, as parsed off the wire, as an instance of this class."""
        return _fields_from_wire(cls, value, where or cls.__name__)

    def to_wire(self):
        """The YAML-safe data the wire stores for this object."""
        return _fields_to_wire(self)


def tag_of(cls: type) -> str:
    """The tag ``cls`` goes by on the wire."""
    return _TAG_OF[cls]


def wire_class(tag: object) -> type | None:
    """The class a tag names, or ``None``."""
    return _REGISTRY.get(tag) if isinstance(tag, str) else None


def alias(name: str, value: object) -> None:
    """Let annotations spell ``name`` — a union alias such as ``Expr``, which is no class and registers nothing."""
    _NAMESPACE[name] = value


def encode(value: object):
    """``value`` as tagged wire data, typed by what it is: what a field with no usable annotation holds, and what a
    standalone object (an expression, a dim) is written as."""
    return _to_wire(value, Any, type(value).__name__)


def decode(value: object):
    """The object a tagged payload spells (:func:`encode`'s inverse)."""
    return _from_wire(Any, value, "wire")


def _hints(cls: type) -> dict[str, object]:
    hints = _HINTS.get(cls)
    if hints is None:
        try:
            hints = _HINTS[cls] = get_type_hints(cls, localns=_NAMESPACE)
        except NameError as exc:
            raise TypeError(f"{cls.__name__}: an annotation names {exc.name!r}, which is no wire class or alias") from exc
    return hints


def _wire_fields(cls: type):
    return [f for f in fields(cls) if f.init and f.name not in cls.wire_skip]


def _default_of(cls: type, f) -> object:
    """The value ``f`` takes when the wire leaves it out: the dataclass default, or the class's own constructor's
    when it defines ``__init__`` itself (a Load's dtype, a Write's atomic flag)."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return _init_defaults(cls).get(f.name, MISSING)


_INIT_DEFAULTS: dict[type, dict[str, object]] = {}


def _init_defaults(cls: type) -> dict[str, object]:
    """The defaults ``cls.__init__`` declares, read once per class: the signature is costly to build."""
    if cls not in _INIT_DEFAULTS:
        parameters = inspect.signature(cls.__init__).parameters.values()
        _INIT_DEFAULTS[cls] = {p.name: p.default for p in parameters if p.default is not inspect.Parameter.empty}
    return _INIT_DEFAULTS[cls]


def _fields_to_wire(obj: Wire):
    cls = type(obj)
    hints, declared = _hints(cls), _wire_fields(cls)
    out = {}
    for f in declared:
        item = getattr(obj, f.name)
        default = _default_of(cls, f)
        if default is not MISSING and item == default:
            continue
        out[f.name] = _to_wire(item, hints[f.name], f"{cls.__name__}.{f.name}")
    return out


def _fields_from_wire(cls: type, value: object, where: str):
    hints, declared = _hints(cls), _wire_fields(cls)
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    names = {f.name: f for f in declared}
    if unknown := set(value) - set(names):
        raise ValueError(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")
    required = {name for name, f in names.items() if _default_of(cls, f) is MISSING}
    if missing := required - set(value):
        raise ValueError(f"{where} missing {', '.join(sorted(missing))}")
    kwargs = {name: _from_wire(hints[name], item, f"{where}.{name}") for name, item in value.items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _scalar_ok(value: object, members: Iterable[type]) -> bool:
    members = tuple(members)
    if isinstance(value, bool):
        return bool in members
    if isinstance(value, int):
        return int in members or float in members
    if isinstance(value, float):
        return float in members
    return isinstance(value, str) and str in members


def _is_wire_class(tp: object) -> bool:
    return isinstance(tp, type) and issubclass(tp, Wire)


def _to_wire(value: object, tp: object, where: str):
    origin, args = get_origin(tp), get_args(tp)
    if tp is Any or tp is object:
        return _tagged(value, where)
    if tp is dict:
        return value
    if origin in (UnionType, typing.Union):
        members = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        if all(member in _SCALARS for member in members):
            return value
        if len(members) == 1:
            return _to_wire(value, members[0], where)
        return _tagged(value, where)
    if _is_wire_class(tp):
        if type(value) is tp or tp.to_wire is not Wire.to_wire:
            return value.to_wire()
        return _tagged(value, where)
    if origin in (tuple, list):
        if origin is tuple and (len(args) != 2 or args[1] is not Ellipsis):
            return [_to_wire(item, arg, f"{where}[{index}]") for index, (arg, item) in enumerate(zip(args, value, strict=True))]
        return [_to_wire(item, args[0], f"{where}[{index}]") for index, item in enumerate(value)]
    if origin is frozenset:
        return sorted((_to_wire(item, args[0], where) for item in value), key=repr)
    if origin is dict:
        if args[0] is str:
            return {key: _to_wire(item, args[1], f"{where}.{key}") for key, item in value.items()}
        return [[_to_wire(key, args[0], where), _to_wire(item, args[1], where)] for key, item in value.items()]
    if tp in _SCALARS:
        return value
    if tp in (tuple, list, frozenset, Mapping):
        return _tagged(value, where)
    raise TypeError(f"{where}: no wire for annotation {tp!r}")


def _tagged(value: object, where: str):
    """Wire data typed by the value itself, tagged where the reader could not tell."""
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, Wire):
        return {_TAG_OF[type(value)]: value.to_wire()}
    if isinstance(value, tuple):
        return {"tuple": [_tagged(item, where) for item in value]}
    if isinstance(value, frozenset):
        return {"frozenset": sorted((_tagged(item, where) for item in value), key=repr)}
    if isinstance(value, list):
        return [_tagged(item, where) for item in value]
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _tagged(item, where) for key, item in value.items()}
        return {"mapping": [[_tagged(key, where), _tagged(item, where)] for key, item in value.items()]}
    raise TypeError(f"{where}: no wire for a {type(value).__name__}")


def _from_wire(tp: object, value: object, where: str):
    origin, args = get_origin(tp), get_args(tp)
    if tp is Any or tp is object:
        return _untagged(value, where)
    if tp is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a mapping")
        return value
    if origin in (UnionType, typing.Union):
        members = [arg for arg in args if arg is not type(None)]
        if value is None and len(members) < len(args):
            return None
        if all(member in _SCALARS for member in members):
            if not _scalar_ok(value, members):
                raise ValueError(f"{where} must be a {' or '.join(member.__name__ for member in members)} value, got {value!r}")
            return value
        if len(members) == 1:
            return _from_wire(members[0], value, where)
        decoded = _untagged(value, where)
        if not isinstance(decoded, tuple(member for member in members if isinstance(member, type))):
            raise ValueError(f"{where} holds a {type(decoded).__name__}, not one of {', '.join(m.__name__ for m in members)}")
        return decoded
    if _is_wire_class(tp):
        # A base class annotation holds a tagged subclass payload — an op among a constant's load ops — while a
        # concrete class, or one whose wire is its own, holds its payload bare.
        if tp.from_wire.__func__ is Wire.from_wire.__func__ and isinstance(value, Mapping) and len(value) == 1:
            ((tag, payload),) = value.items()
            if (cls := _REGISTRY.get(tag)) is not None and cls is not tp and issubclass(cls, tp):
                return cls.from_wire(payload, f"{where}.{tag}")
        return tp.from_wire(value, where)
    if origin in (tuple, list, frozenset):
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list")
        if origin is tuple and (len(args) != 2 or args[1] is not Ellipsis):
            if len(value) != len(args):
                raise ValueError(f"{where} must be a list of {len(args)}")
            pairs = enumerate(zip(args, value, strict=True))
            return tuple(_from_wire(arg, item, f"{where}[{index}]") for index, (arg, item) in pairs)
        items = [_from_wire(args[0], item, f"{where}[{index}]") for index, item in enumerate(value)]
        return origin(items) if origin is not list else items
    if origin is dict:
        if args[0] is str:
            if not isinstance(value, Mapping):
                raise ValueError(f"{where} must be a mapping")
            return {
                _from_wire(str, key, f"{where} key {key!r}"): _from_wire(args[1], item, f"{where}.{key}") for key, item in value.items()
            }
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list of key/value pairs")
        return {_from_wire(args[0], key, where): _from_wire(args[1], item, where) for key, item in value}
    if tp in _SCALARS:
        if not _scalar_ok(value, (tp,)):
            raise ValueError(f"{where} must be a {tp.__name__}")
        return value
    if tp in (tuple, list, frozenset, Mapping):
        return _untagged(value, where)
    raise TypeError(f"{where}: no wire for annotation {tp!r}")


def _untagged(value: object, where: str):
    """The object a tagged payload spells (:func:`_tagged`'s inverse)."""
    if isinstance(value, list):
        return [_untagged(item, where) for item in value]
    if not isinstance(value, Mapping):
        return value
    if len(value) == 1:
        ((tag, payload),) = value.items()
        if tag == "tuple":
            return tuple(_untagged(item, where) for item in payload)
        if tag == "frozenset":
            return frozenset(_untagged(item, where) for item in payload)
        if tag == "mapping":
            return {_untagged(key, where): _untagged(item, where) for key, item in payload}
        if (cls := _REGISTRY.get(tag)) is not None:
            return cls.from_wire(payload, f"{where}.{tag}")
    return {key: _untagged(item, where) for key, item in value.items()}


# --- graphs and kernels on the wire -----------------------------------------------------------


def intern_wire(pool: list[dict], wire: dict) -> int:
    """``wire``'s index in ``pool``, appended when no equal wire is there."""
    for index, current in enumerate(pool):
        if current == wire:
            return index
    pool.append(wire)
    return len(pool) - 1


def intern(pool: list[dict], graph) -> int:
    """``graph``'s wire in ``pool``, added when absent: the index a golden's config refers to it by."""
    return intern_wire(pool, graph.to_wire())


def kernel_tile(op):
    """The tile kernel ``op`` lowered from — the first ``TileOp`` on its source chain — or ``None`` for
    a kernel no tile stands behind. Every kernel the tuner and ``run --bench`` measure has one, the
    kernel-cache replay included (a cached kernel keeps its chain); the deploy identity a golden
    receipt names and the definition a ``kernel`` row stores are both read off it."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    return next((ancestor for ancestor in op.source_chain() if isinstance(ancestor, TileOp)), None)


def formed_from(tile):
    """The loop op ``tile`` was formed from — the fused loop op the lift threads in as a kernel's ``source``,
    the loop nest a cut or split piece is re-formed through — or ``None`` for a kernel formed from no loop op:
    a piece carved from a twisted tree, whose derived body the lift does not take back."""
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415

    return next((op for op in tile.source_chain() if isinstance(op, LoopOp)), None)


def kernel_wire(tile) -> dict:
    """The Loop IR wire of one tile kernel: a one-node program holding the body the kernel was formed from
    (:func:`formed_from`), bound to the tile's own buffers. The lowering passes take that body back to the
    kernel — the lift, the twist and the identity strategy give it the same exact identity and ``S_*`` stamps
    — so a kernel row's definition re-lowers on its own, a piece a cut minted included, rather than as "its
    parent plus the route". A kernel formed from no loop op holds its derived ``loop_body`` instead: it
    decodes to the kernel's identities, but the lift does not take it back and the Loop passes would normalize
    a size-one axis away and mint another kernel — only its parent's program reaches such a kernel."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415

    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], outputs=(tensor,), node_id=name)
    # A node's primary buffer is named after the node, so the kernel node takes its primary output's
    # buffer name as id — the shape a golden's standalone slice has too.
    primary, *_ = tile.outputs
    formed = formed_from(tile)
    body = formed.body if formed is not None else tile.loop_body
    graph.add_node(LoopOp(body=body, name=tile.name), list(tile.inputs), outputs=tuple(tile.outputs.values()), node_id=primary)
    graph.inputs = list(tile.inputs)
    graph.outputs = list(tile.outputs)
    return graph.to_wire()


def symbolic_vars(wire: dict) -> set[str]:
    """The symbolic dim vars a kernel wire's buffers name — the keys a measurement of it binds
    (:func:`kernel_bindings`), and what a piece keeps of its parent's bindings."""
    out: set[str] = set()
    for node in wire["nodes"]:
        for _name, _dtype, dims in node["outputs"]:
            for dim in dims:
                if isinstance(dim, dict) and "sym" in dim:
                    out.add(str(dim["sym"]))
                elif isinstance(dim, dict) and "expr" in dim:
                    out |= set(decode(dim["expr"]).free_vars())
    return out


def symbolic_bindings(tensors: Iterable) -> dict[str, int]:
    """Each symbolic dim var of ``tensors`` bound to the size a bench runs it at: the dim's hint
    (``DEFAULT_SEQ_HINT`` for a bare seq axis), which is what the backend binds when no input is
    supplied. The first hint a var is seen with wins."""
    from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim  # noqa: PLC0415

    bindings: dict[str, int] = {}
    for tensor in tensors:
        for dim in tensor.shape:
            if isinstance(dim, Dim) and not dim.is_static:
                for var in dim.expr.free_vars():
                    bindings.setdefault(var, dim.hint or DEFAULT_SEQ_HINT)
    return bindings


def kernel_bindings(op) -> dict[str, int]:
    """The sizes a bench of ``op`` binds its symbolic dims to (:func:`symbolic_bindings` over its
    buffers) — what a ``perf`` row of a dynamic kernel records so rows at two sizes stay apart."""
    return symbolic_bindings((*op.inputs.values(), *op.outputs.values()))
