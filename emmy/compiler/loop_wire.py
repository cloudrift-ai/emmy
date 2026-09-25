"""Repository-local wire payload for post-fusion Loop IR fallbacks.

Torch provenance remains the preferred golden target because frontend IR is
the stable persistence boundary.  A traced kernel that has no frontend origin
is instead stored as its standalone Loop IR slice so inventory generation is
complete rather than lossy.  Golden files in this repository are regenerated
when this implementation-level Loop IR representation changes.

The same codec spells a measured KERNEL's definition (:func:`kernel_wire`): the tune DB stores one
wire per kernel identity, whether the kernel is the fused kernel of a slice or a piece a cut or a
split minted, so the same kernel reached from two parents has one definition and one candidate set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.dtype import DataType
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Carry,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    SelectBranch,
    StridedLoop,
    Write,
    ZeroPrologue,
)
from emmy.compiler.torch_wire import (
    dim_from_wire,
    dim_to_wire,
    expr_from_wire,
    expr_to_wire,
    op_from_wire,
    op_to_wire,
    tensor_from_wire,
    tensor_to_wire,
)

_EXPR_TYPES = (Var, Literal, BinaryExpr, Builtin, FuncCallExpr, TernaryExpr, CastExpr)
_DATA_CLASSES = (
    Axis,
    Window,
    Accum,
    Assign,
    Pre,
    Carry,
    Cond,
    Init,
    Lambda,
    Let,
    Load,
    Loop,
    Select,
    SelectBranch,
    StridedLoop,
    Write,
    ZeroPrologue,
)
_CLASS_BY_NAME = {cls.__name__: cls for cls in _DATA_CLASSES}


def _value_to_wire(value: Any) -> Any:
    if isinstance(value, Dim):
        return {"dim": dim_to_wire(value)}
    if isinstance(value, _EXPR_TYPES):
        return {"expr": expr_to_wire(value)}
    if isinstance(value, DataType):
        return {"dtype": value.name}
    if isinstance(value, ElementwiseImpl):
        return {"elementwise": value.name}
    if isinstance(value, Body):
        return {"body": [_value_to_wire(item) for item in value]}
    if isinstance(value, tuple):
        return {"tuple": [_value_to_wire(item) for item in value]}
    if isinstance(value, frozenset):
        return {"frozenset": [_value_to_wire(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [_value_to_wire(item) for item in value]
    if isinstance(value, dict):
        return {"mapping": [[_value_to_wire(key), _value_to_wire(item)] for key, item in value.items()]}
    if is_dataclass(value) and type(value) in _DATA_CLASSES:
        return {
            "class": type(value).__name__,
            "fields": {field.name: _value_to_wire(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Enum):
        raise TypeError(f"Loop IR wire does not support enum {type(value).__name__}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Loop IR wire does not support {type(value).__name__}")


def _value_from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_value_from_wire(item) for item in value]
    if not isinstance(value, dict) or (len(value) != 1 and set(value) != {"class", "fields"}):
        if isinstance(value, dict):
            raise ValueError("Loop IR value must be a tagged mapping")
        return value
    if set(value) == {"dim"}:
        return dim_from_wire(value["dim"])
    if set(value) == {"expr"}:
        return expr_from_wire(value["expr"])
    if set(value) == {"dtype"}:
        return get_dtype(value["dtype"])
    if set(value) == {"elementwise"}:
        return ElementwiseImpl(str(value["elementwise"]))
    if set(value) == {"body"}:
        payload = value["body"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR body must be a list")
        return Body(_value_from_wire(item) for item in payload)
    if set(value) == {"tuple"}:
        payload = value["tuple"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR tuple must be a list")
        return tuple(_value_from_wire(item) for item in payload)
    if set(value) == {"frozenset"}:
        payload = value["frozenset"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR frozenset must be a list")
        return frozenset(_value_from_wire(item) for item in payload)
    if set(value) == {"mapping"}:
        payload = value["mapping"]
        if not isinstance(payload, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in payload):
            raise ValueError("Loop IR mapping must be a list of key/value pairs")
        return {_value_from_wire(key): _value_from_wire(item) for key, item in payload}
    if set(value) == {"class", "fields"}:
        class_name = value["class"]
        if not isinstance(class_name, str):
            raise ValueError("Loop IR class name must be a string")
        cls = _CLASS_BY_NAME.get(class_name)
        if cls is None:
            raise ValueError(f"Loop IR value has unknown class {class_name!r}")
        payload = value["fields"]
        if not isinstance(payload, dict):
            raise ValueError(f"Loop IR {class_name} fields must be a mapping")
        expected = {field.name for field in fields(cls)}
        # A loop's ``role`` was an annotation a stored wire may still carry; a loop folds iff its
        # body carries an ``Accum``, so the field is dropped rather than refused.
        payload = {name: item for name, item in payload.items() if not (name == "role" and cls in (Loop, StridedLoop))}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"Loop IR {class_name} has unknown fields: {', '.join(sorted(unknown))}")
        try:
            return cls(**{name: _value_from_wire(item) for name, item in payload.items()})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Loop IR {class_name} is invalid: {exc}") from exc
    raise ValueError("Loop IR value has an unknown tag")


def loop_graph_to_wire(graph: Graph) -> dict:
    """Serialize a standalone Loop IR graph to YAML-safe data."""
    nodes = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if isinstance(node.op, LoopOp):
            op = "loop"
            attrs = {"body": _value_to_wire(node.op.body)}
            if node.op.name:
                attrs["name"] = node.op.name
        elif isinstance(node.op, (InputOp, ConstantOp)):
            encoded = op_to_wire(node.op)
            op, attrs = encoded["op"], encoded["attrs"]
        else:
            raise TypeError(f"Loop IR graph contains unsupported compute op {type(node.op).__name__}")
        item = {"id": node_id, "op": op}
        if attrs:
            item["attrs"] = attrs
        if node.inputs:
            item["inputs"] = list(node.inputs)
        item["outputs"] = [tensor_to_wire(tensor) for tensor in node.outputs]
        nodes.append(item)
    return {"inputs": list(graph.inputs), "outputs": list(graph.outputs), "nodes": nodes}


def loop_graph_from_wire(value: object) -> Graph:
    """Decode and validate a standalone post-fusion kernel slice."""
    if not isinstance(value, dict):
        raise ValueError("Loop IR program must be a mapping")
    unknown = set(value) - {"inputs", "outputs", "nodes"}
    if unknown:
        raise ValueError(f"Loop IR program has unknown fields: {', '.join(sorted(unknown))}")
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Loop IR program nodes must be a list")
    graph = Graph()
    for index, item in enumerate(nodes):
        if not isinstance(item, dict):
            raise ValueError(f"Loop IR node {index} must be a mapping")
        unknown = set(item) - {"id", "op", "attrs", "inputs", "outputs"}
        if unknown:
            raise ValueError(f"Loop IR node {index} has unknown fields: {', '.join(sorted(unknown))}")
        node_id = item.get("id")
        inputs = item.get("inputs", [])
        outputs = item.get("outputs")
        attrs = item.get("attrs", {})
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Loop IR node {index} requires a non-empty id")
        if not isinstance(inputs, list) or not all(isinstance(name, str) for name in inputs):
            raise ValueError(f"Loop IR node {node_id!r} inputs must be string names")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"Loop IR node {node_id!r} outputs must be a non-empty list")
        if not isinstance(attrs, dict):
            raise ValueError(f"Loop IR node {node_id!r} attrs must be a mapping")
        if item.get("op") == "loop":
            unknown_attrs = set(attrs) - {"body", "name"}
            if unknown_attrs or "body" not in attrs:
                raise ValueError(f"Loop IR node {node_id!r} has invalid loop attrs")
            try:
                body = _value_from_wire(attrs["body"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Loop IR node {node_id!r} body is invalid: {exc}") from exc
            if not isinstance(body, Body):
                raise ValueError(f"Loop IR node {node_id!r} body did not decode to Body")
            op = LoopOp(body=body, name=str(attrs.get("name", "")))
        else:
            op = op_from_wire({"op": item.get("op"), "attrs": attrs})
            if not isinstance(op, (InputOp, ConstantOp)):
                raise ValueError(f"Loop IR node {node_id!r} boundary must be input or constant")
        try:
            graph.add_node(op, inputs, outputs=tuple(tensor_from_wire(output) for output in outputs), node_id=node_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Loop IR node {node_id!r} is invalid: {exc}") from exc
    graph_inputs = value.get("inputs")
    graph_outputs = value.get("outputs")
    if not isinstance(graph_inputs, list) or not all(isinstance(name, str) for name in graph_inputs):
        raise ValueError("Loop IR program inputs must be string names")
    if not isinstance(graph_outputs, list) or not all(isinstance(name, str) for name in graph_outputs):
        raise ValueError("Loop IR program outputs must be string names")
    graph.inputs = list(graph_inputs)
    graph.outputs = list(graph_outputs)
    compute = [node for node in graph.nodes.values() if not isinstance(node.op, (InputOp, ConstantOp))]
    if not compute or any(not isinstance(node.op, LoopOp) for node in compute):
        raise ValueError("Loop IR program must contain only LoopOp compute nodes")
    output_producers = [graph.producer(name) for name in graph.outputs]
    if not output_producers or any(node is None or not isinstance(node.op, LoopOp) for node in output_producers):
        raise ValueError("Loop IR program outputs must be produced by LoopOp nodes")
    for name in (*graph.inputs, *graph.outputs):
        if graph.buffer(name) is None:
            raise ValueError(f"Loop IR program references unknown boundary buffer {name!r}")
    graph.topological_order()
    return graph


def kernel_tile(op):
    """The tile kernel ``op`` lowered from — the first ``TileOp`` on its source chain — or ``None`` for
    a kernel no tile stands behind. Every kernel the tuner and ``run --bench`` measure has one, the
    kernel-cache replay included (a cached kernel keeps its chain); the deploy identity a golden
    receipt names and the definition a ``kernel`` row stores are both read off it."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    return next((ancestor for ancestor in op.source_chain() if isinstance(ancestor, TileOp)), None)


def formed_from(tile) -> LoopOp | None:
    """The loop op ``tile`` was formed from — the fused loop op the lift threads in as a kernel's ``source``,
    the loop nest a cut or split piece is re-formed through — or ``None`` for a kernel formed from no loop op:
    a piece carved from a twisted tree, whose derived body the lift does not take back."""
    return next((op for op in tile.source_chain() if isinstance(op, LoopOp)), None)


def kernel_wire(tile) -> dict:
    """The Loop IR wire of one tile kernel: a one-node program holding the body the kernel was formed from
    (:func:`formed_from`), bound to the tile's own buffers. The lowering passes take that body back to the
    kernel — the lift, the twist and the identity strategy give it the same exact identity and ``S_*`` stamps
    — so a kernel row's definition re-lowers on its own, a piece a cut minted included, rather than as "its
    parent plus the route". A kernel formed from no loop op holds its derived ``loop_body`` instead: it
    decodes to the kernel's identities, but the lift does not take it back and the Loop passes would normalize
    a size-one axis away and mint another kernel — only its parent's program reaches such a kernel."""
    formed = formed_from(tile)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], outputs=(tensor,), node_id=name)
    # A node's primary buffer is named after the node, so the kernel node takes its primary output's
    # buffer name as id — the shape a golden's standalone slice has too.
    primary, *_ = tile.outputs
    body = formed.body if formed is not None else tile.loop_body
    graph.add_node(LoopOp(body=body, name=tile.name), list(tile.inputs), outputs=tuple(tile.outputs.values()), node_id=primary)
    graph.inputs = list(tile.inputs)
    graph.outputs = list(tile.outputs)
    return loop_graph_to_wire(graph)


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
                    out |= set(expr_from_wire(dim["expr"]).free_vars())
    return out


def symbolic_bindings(tensors: Iterable) -> dict[str, int]:
    """Each symbolic dim var of ``tensors`` bound to the size a bench runs it at: the dim's hint
    (``DEFAULT_SEQ_HINT`` for a bare seq axis), which is what the backend binds when no input is
    supplied. The first hint a var is seen with wins."""
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


def intern_loop_program(programs: list[dict], graph: Graph) -> int:
    payload = loop_graph_to_wire(graph)
    for index, current in enumerate(programs):
        if current == payload:
            return index
    programs.append(payload)
    return len(programs) - 1


def validate_loop_program_pool(programs: object) -> list[dict]:
    if programs is None:
        return []
    if not isinstance(programs, list):
        raise ValueError("golden loops must be a list")
    out: list[dict] = []
    for index, payload in enumerate(programs):
        if not isinstance(payload, dict):
            raise ValueError(f"golden Loop IR program {index} must be a mapping")
        try:
            loop_graph_from_wire(payload)
        except ValueError as exc:
            raise ValueError(f"golden Loop IR program {index}: {exc}") from exc
        out.append(payload)
    return out
