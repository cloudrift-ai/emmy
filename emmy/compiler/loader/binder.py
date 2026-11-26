"""Apply ``ConstantOp.load_ops`` to a source ndarray via the reference
NumPy backend.

The loader produces raw source tensors (from safetensors or
``module.named_parameters()``); this binder is the small adapter that
runs each constant's ``load_ops`` chain on its source array. Reusing
the existing ``Backend.run`` interpreter (``backend/base.py``) means
every op already has a numpy ``forward()`` — no new code path.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp


def apply_load_ops(source: np.ndarray, load_ops: tuple, dtype: str | None = None) -> np.ndarray:
    """Run ``load_ops`` over ``source`` using the NumPy backend.

    Builds a tiny single-input graph and dispatches it through the
    default ``Backend.run`` interpreter. Each load op must already have
    a working ``Op.forward`` (true for ``TransposeOp`` / ``ReshapeOp``
    by construction — those are the only ops the fold pass produces).

    ``dtype`` overrides the chain's tensor dtype token — needed for
    bits-carrier sources (an fp8 constant's raw uint8 bits), whose numpy
    dtype name (``uint8``) is not a registered ``DataType``.
    """
    if not load_ops:
        return np.ascontiguousarray(source)

    g = Graph()
    src_dtype = dtype or str(source.dtype)
    in_id = g.add_node(op=InputOp(), inputs=[], output=Tensor("src", tuple(source.shape), src_dtype))
    g.inputs.append(in_id)

    cur = in_id
    cur_shape = tuple(source.shape)
    for i, op in enumerate(load_ops):
        cur_shape = op.infer_output_shape([cur_shape])
        nid = g.add_node(op=op, inputs=[cur], output=Tensor(f"step_{i}", cur_shape, src_dtype))
        cur = nid
    g.outputs.append(cur)

    from emmy.compiler.backend.base import Backend

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result, _ = _NumpyInterp().run(g, input_data={in_id: np.ascontiguousarray(source)})
    return np.ascontiguousarray(result.outputs[cur])


def assemble_source(op, sources: dict[str, np.ndarray]) -> np.ndarray | None:
    """Assemble a constant's raw pre-chain source array from ``sources``: the single
    ``source_path`` lookup, or the axis-0 concat of its ``source_parts`` (the
    a multi-source weight concat). ``None`` when any source is missing.
    Duck-typed over ``ConstantOp`` and ``backend.plan.WeightSpec`` alike.
    A plan ``WeightSpec.generated`` carries a deterministic source-free bind record already
    evaluated to ``(dtype, shape, bytes)``; reconstruct it without an external lookup.
    Other ``source_graph`` bind records are the loaders' own business
    (:func:`evaluate_source_graph`) — this helper answers ``None`` for them.
    """
    generated = getattr(op, "generated", None)
    if generated is not None:
        dtype, shape, data = generated
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
    if op.source_parts:
        parts = [sources.get(path) for path, _shape in op.source_parts]
        if any(p is None for p in parts):
            return None
        return np.concatenate([np.ascontiguousarray(p) for p in parts], axis=0)
    return sources.get(op.source_path) if op.source_path is not None else None


def evaluate_source_graph(record, sources: dict[str, np.ndarray]) -> np.ndarray | None:
    """Evaluate a ``ConstantOp.source_graph`` bind record against per-path ``sources``.

    Binds every leaf constant of the record through :func:`bind_constants` (each leaf's own
    dtype rules apply — an f8-dtype leaf binds raw uint8 bits, and leaf ``load_ops`` replay),
    then runs the record through the reference NumPy backend. ``None`` when any leaf source is
    missing from ``sources`` — the caller skips the constant, same as a missing plain source.
    """
    leaf_data = bind_constants(record, sources)
    if any(nid not in leaf_data for nid, _op in record.loadable_constants()):
        return None

    from emmy.compiler.backend.base import Backend  # noqa: PLC0415

    class _NumpyInterp(Backend):
        def compile(self, graph):
            return graph

    result, _ = _NumpyInterp().run(record, input_data=leaf_data)
    return np.ascontiguousarray(result.outputs[record.outputs[0]])


def bind_constants(graph: Graph, sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the per-node ``input_data`` dict for every ``ConstantOp``.

    ``sources`` maps each ``ConstantOp.source_path`` (or ``source_parts`` part path) to its
    raw source ndarray (typically read from safetensors or pulled from a live
    ``nn.Module``). For each constant, this assembles the pre-chain source
    (:func:`assemble_source`), runs ``load_ops`` over it, and stores the result keyed by
    the constant's node id. Scalar constants (``value is not None``) are skipped — the
    backend materializes them on its own. Constants without a source are skipped too
    (synthetic constants emitted by passes carry their ``value`` directly).
    """
    out: dict[str, np.ndarray] = {}
    for nid, op in graph.loadable_constants():
        if op.source_graph is not None:
            # Folded static computation: evaluate the record over the same per-path
            # sources, then run this constant's own trailing chain.
            val = evaluate_source_graph(op.source_graph, sources)
            if val is not None:
                out[nid] = apply_load_ops(val, op.load_ops)
            continue
        src = assemble_source(op, sources)
        if src is None:
            continue
        # An f8-dtype constant binds raw uint8 BITS (the in-graph decode-cone form — the
        # graph's own algebra owns the value semantics); its chain runs under the f8 token
        # because the carrier's numpy dtype (uint8) is not a registered DataType. Mirrors the
        # safetensors loader's bits-bind convention.
        dt = graph.nodes[nid].output.dtype
        tok = dt.name if getattr(dt, "name", None) in ("f8e4m3", "f8e5m2") and src.dtype == np.uint8 else None
        out[nid] = apply_load_ops(src, op.load_ops, dtype=tok)
    return out


def bind_constants_from_module(graph: Graph, module) -> dict[str, np.ndarray]:  # noqa: ANN001 — torch.nn.Module, duck-typed
    """Bind every parameter/buffer ``ConstantOp`` from a *live* ``nn.Module``.

    The whole-model trace wrapper carries computed buffers — the precomputed
    rotary ``cos``/``sin`` and the causal mask — that aren't in the checkpoint's
    safetensors, so :func:`load_constants_from_safetensors` can't supply them.
    Binding from the traced module's own ``state_dict`` covers those *and* the
    weights uniformly: each ``ConstantOp.source_path`` is the module-attribute
    path captured at trace time, so it matches a ``state_dict`` key verbatim.

    ``state_dict`` (not ``named_parameters``) is the right source because it
    lists *tied* weights under every name they're registered as — e.g. a model
    with ``tie_word_embeddings=True`` traces an ``lm_head.weight`` constant, but
    ``named_parameters`` dedups it down to the shared ``embed_tokens.weight``
    only, leaving the final projection unbound (→ zero logits). Tensors are cast
    to float32 numpy (the backend dtype; also sidesteps reading bf16 checkpoints
    through numpy). NOTE: a live module cannot supply an fp8 checkpoint's raw
    bits or scale tensors — constants whose sources only exist in the checkpoint
    (f8-dtype leaves, ``source_graph`` records over them) stay unbound here and
    must come from the safetensors loader."""
    sources: dict[str, np.ndarray] = {}
    for name, t in module.state_dict().items():
        sources[name] = t.detach().cpu().float().numpy()
    return bind_constants(graph, sources)


def bind_inputs(compiled, module, example_args, example_kwargs, checkpoint=None):
    """Match graph inputs and constants to tensors from ``module`` / call args.

    Activations come from the call's positional/keyword tensors. Constants
    come from ``module.named_parameters()`` / ``named_buffers()`` keyed
    by each ``ConstantOp.source_path`` recorded at trace time. Each
    constant's ``load_ops`` chain is replayed via the NumPy backend
    (see ``compiler.loader.binder``), so any compile-time-folded
    transpose / reshape is honored uniformly.

    ``checkpoint`` (the model id / path, when the caller has one) covers the
    constants a live module cannot supply: an fp8 checkpoint's weights bind as
    raw bits + scale sources (or a folded ``source_graph`` record over them)
    that exist only in the safetensors shards, so any still-unbound constant is
    re-tried through the safetensors loader before the hard error below.
    """
    import torch

    flat_inputs: list[torch.Tensor] = []
    for v in example_args:
        flat_inputs.extend(flatten_tensors(v))
    for v in example_kwargs.values():
        flat_inputs.extend(flatten_tensors(v))

    input_ids = list(compiled.inputs)
    if len(input_ids) != len(flat_inputs):
        # Raise, don't sys.exit: this also runs inside the bench worker child, where an
        # exit(1) surfaces to the parent as an opaque ``SystemExit(1)`` with the real
        # cause stranded in the child's log stream.
        raise RuntimeError(f"Input arity mismatch: graph has {len(input_ids)} inputs, code provided {len(flat_inputs)}")

    input_data: dict[str, np.ndarray] = {}
    torch_f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    for nid, tensor in zip(input_ids, flat_inputs, strict=True):
        dtype = compiled.nodes[nid].output.dtype
        np_dtype = dtype.np
        tensor = tensor.detach().cpu()
        # An fp8 activation/weight INPUT binds the raw bit pattern on its uint8 carrier —
        # numpy has no fp8 scalar type, and a value cast would decode the bits (the same
        # rule the constant side and the serving expert feed already follow).
        if tensor.dtype in torch_f8:
            tensor = tensor.view(torch.uint8)
        if dtype.name == "bf16":
            from emmy.compiler.dtype import encode_bf16  # noqa: PLC0415

            if tensor.dtype == torch.bfloat16:
                input_data[nid] = tensor.contiguous().view(torch.uint16).numpy()
            else:
                input_data[nid] = encode_bf16(tensor.numpy())
        else:
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            input_data[nid] = tensor.numpy().astype(np_dtype, copy=False)

    # ``remove_duplicate=False`` so tied weights (e.g. a model whose lm_head
    # shares the embedding matrix) are surfaced under *every* name, including
    # the ``source_path`` the tracer recorded — otherwise the alias the trace
    # picked may be the one torch dedups away and the constant won't bind.
    sources: dict[str, np.ndarray] = {}
    for path, tensor in module.named_parameters(remove_duplicate=False):
        tensor = tensor.detach().cpu()
        sources[path] = (
            tensor.contiguous().view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy().astype(np.float32, copy=False)
        )
    for path, tensor in module.named_buffers(remove_duplicate=False):
        tensor = tensor.detach().cpu()
        sources[path] = (
            tensor.contiguous().view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy().astype(np.float32, copy=False)
        )

    input_data.update(bind_constants(compiled, sources))

    if checkpoint is not None and any(nid not in input_data for nid, _op in compiled.loadable_constants()):
        from emmy.compiler.trace.huggingface import quantized_checkpoint_dir  # noqa: PLC0415

        quant_dir = quantized_checkpoint_dir(checkpoint)
        if quant_dir is not None:
            from emmy.compiler.loader.safetensors import load_constants_from_safetensors  # noqa: PLC0415

            for nid, arr in load_constants_from_safetensors(compiled, str(quant_dir)).items():
                input_data.setdefault(nid, arr)

    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp) or nid in input_data:
            continue
        if node.op.value is not None:
            continue  # backend materializes scalars from node.op.value
        raise RuntimeError(f"Could not bind constant {nid} (source_path={node.op.source_path!r})")
    return input_data


def flatten_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(flatten_tensors(v))
        return out
    return []
