"""Device-resident coded output heads (EXL3, NVFP4) spelled through the ordinary compiler pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn

#: An NVFP4 head's leaves: packed e2m1 codes, raw e4m3 block scales and the f32 global scale.
NVFP4_HEAD_LEAVES = ("weight", "weight_scale", "weight_scale_2")
#: The rows the head program is compiled and measured at: one sampled row, a single request's decode step.
HEAD_ROWS = 1
#: The tensor only an NVFP4 head stores.
_NVFP4_HEAD_MARK = "lm_head.weight_scale_2"


@dataclass(frozen=True)
class Exl3HeadSpec:
    bits: int
    codebook: int
    hidden_size: int
    stored_vocab_size: int
    vocab_size: int
    #: The factorized spelling reads f16 activations (a torch dtype name).
    dtype = "float16"

    @classmethod
    def from_source(
        cls,
        source: dict,
        *,
        hidden_size: int,
        vocab_size: int,
        tensor_parallel_size: int,
        tied: bool,
    ) -> Exl3HeadSpec | None:
        """Return the coded compiler-head contract, or ``None`` for the decoded fallback."""
        if tied or tensor_parallel_size != 1:
            return None
        if "lm_head.mcg" in source and "lm_head.mul1" in source:
            raise ValueError("EXL3 coded lm_head carries both mcg and mul1 codebook markers")
        codebook = 2 if "lm_head.mul1" in source else 1 if "lm_head.mcg" in source else 0
        if codebook != 2:
            return None
        trellis = np.asarray(source["lm_head.trellis"])
        suh = np.asarray(source["lm_head.suh"])
        svh = np.asarray(source["lm_head.svh"])
        if trellis.dtype != np.int16 or trellis.ndim != 3 or trellis.shape[-1] % 16:
            raise ValueError(f"EXL3 coded lm_head has invalid trellis storage {trellis.dtype}{trellis.shape}")
        bits = trellis.shape[-1] // 16
        stored_hidden, stored_vocab = trellis.shape[0] * 16, trellis.shape[1] * 16
        if suh.dtype != np.float16 or suh.shape != (stored_hidden,):
            raise ValueError(f"EXL3 coded lm_head has invalid suh storage {suh.dtype}{suh.shape}")
        if svh.dtype != np.float16 or svh.shape != (stored_vocab,):
            raise ValueError(f"EXL3 coded lm_head has invalid svh storage {svh.dtype}{svh.shape}")
        if stored_hidden != hidden_size or stored_vocab < vocab_size:
            raise ValueError(f"EXL3 coded lm_head stores {(stored_vocab, stored_hidden)}, expected at least {(vocab_size, hidden_size)}")
        return cls(bits, codebook, stored_hidden, stored_vocab, vocab_size)

    def spell(self, graph, x: str) -> str:
        """Spell the logits of ``x`` as factorized algebra over the trellis codes; return the logits node."""
        from emmy.compiler.graph import Tensor
        from emmy.compiler.ir.base import ConstantOp
        from emmy.compiler.loader.trellis import spell_factored_linear

        shapes = {
            "trellis": ((self.hidden_size // 16, self.stored_vocab_size // 16, 16 * self.bits), "i16"),
            "suh": ((self.hidden_size,), "f16"),
            "svh": ((self.stored_vocab_size,), "f16"),
        }
        for leaf, (shape, dtype) in shapes.items():
            # The node id is a C identifier (graph buffer names become kernel parameter names);
            # the checkpoint pointer stays on ``source_path``.
            nid = f"lm_head_{leaf}"
            graph.add_node(
                ConstantOp(name=nid, source_path=f"lm_head.{leaf}", source_shape=shape, source_dtype=dtype),
                [],
                Tensor(nid, shape, dtype),
                node_id=nid,
            )
        rows = graph.nodes[x].output.shape[0]
        return spell_factored_linear(
            graph,
            "lm_head_trellis",
            "lm_head_suh",
            "lm_head_svh",
            cb=self.codebook,
            weight_shape=(self.stored_vocab_size, self.hidden_size),
            x=x,
            bias=None,
            out=Tensor("logits", (rows, self.stored_vocab_size), "f32"),
            weight_name="lm_head",
        )

    def feed(self, source: dict) -> dict:
        """The arrays the head program's coded constants bind, checked by :meth:`from_source` already."""
        return {f"lm_head_{leaf}": np.ascontiguousarray(source[f"lm_head.{leaf}"]) for leaf in ("trellis", "suh", "svh")}


@dataclass(frozen=True)
class Nvfp4HeadSpec:
    """An NVFP4 head stores exactly the vocab: two e2m1 codes per byte, an e4m3 scale per 16 and one f32 scale."""

    hidden_size: int
    vocab_size: int
    #: The activation dtype the head reads, the trunk's (a torch dtype name).
    dtype: str

    @classmethod
    def from_config(cls, *, hidden_size: int, vocab_size: int, tensor_parallel_size: int, tied: bool, dtype: str) -> Nvfp4HeadSpec | None:
        """Return the coded compiler-head contract, or ``None`` for the decoded fallback. The codes are read at
        load time, where :meth:`feed` checks them against this contract."""
        if tied or tensor_parallel_size != 1:
            return None
        return cls(hidden_size, vocab_size, dtype)

    def spell(self, graph, x: str) -> str:
        """Spell the logits of ``x`` against the decoded head's own weights: each value decoded in-graph exactly
        (:func:`~emmy.compiler.loader.quant.dequantize_nvfp4`) and rounded once to the model dtype, as the decoded
        fallback loads its parameter. Logits accumulate and return in f32. Returns the logits node."""
        from emmy.compiler.graph import Tensor
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.ir.frontend.ir import LinearOp
        from emmy.compiler.ir.tensor.ir import ElementwiseOp
        from emmy.compiler.loader.quant import NVFP4_BLOCK, spell_nvfp4_inputs

        n, k = self.vocab_size, self.hidden_size
        rows = graph.nodes[x].output.shape[0]
        graph.add_node(InputOp(), [], Tensor("lm_head_weight", (n, k), "f32"), node_id="lm_head_weight")
        graph.inputs.append("lm_head_weight")
        # The product runs in f32 over values the model dtype holds exactly, so a 16-bit tensor-core tile is exact.
        rounded = graph.add_node(ElementwiseOp(op="copy"), ["lm_head_weight"], Tensor("lm_head_weight_rounded", (n, k), self.dtype))
        weight = graph.add_node(ElementwiseOp(op="copy"), [rounded], Tensor("lm_head_weight_f32", (n, k), "f32"))
        x32 = graph.add_node(ElementwiseOp(op="copy"), [x], Tensor("x32", (rows, k), "f32"))
        logits = graph.add_node(LinearOp(), [x32, weight], Tensor("logits", (rows, n), "f32"))
        spell_nvfp4_inputs(graph, {"lm_head_weight": ((n, k // 2), (n, k // NVFP4_BLOCK), (1, 1))})
        return logits

    def feed(self, source: dict) -> dict:
        """The arrays the head program's coded inputs bind: the stored codes, the raw block-scale bits and the one
        global scale. Codes that do not store this head raise."""
        from emmy.compiler.loader.quant import NVFP4_BLOCK

        packed, scale, scale_2 = (np.asarray(source[f"lm_head.{leaf}"]) for leaf in NVFP4_HEAD_LEAVES)
        n, k = self.vocab_size, self.hidden_size
        expected = (np.uint8, (n, k // 2), np.uint8, (n, k // NVFP4_BLOCK), 1)
        if (packed.dtype, packed.shape, scale.dtype, scale.shape, scale_2.size) != expected:
            raise ValueError(
                f"NVFP4 lm_head stores codes {packed.dtype}{packed.shape}, block scales {scale.dtype}{scale.shape} and "
                f"global scale {scale_2.shape}, but the head is {(n, k)}"
            )
        return {"lm_head_weight": packed, "lm_head_weight_scale": scale, "lm_head_weight_scale_2": scale_2.reshape(1, 1)}


def nvfp4_head_dir(model_id: str) -> str | None:
    """The checkpoint directory when this checkpoint's head is NVFP4 (it stores a global ``weight_scale_2``), else
    ``None``: one scalar read, no codes."""
    from emmy.compiler.loader.safetensors import load_sources_by_path
    from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

    qdir = quantized_checkpoint_dir(model_id)
    return str(qdir) if qdir is not None and load_sources_by_path(str(qdir), [_NVFP4_HEAD_MARK]) else None


def nvfp4_head_stored(names) -> bool:
    """Whether a checkpoint's tensor names hold an NVFP4 output head: its global scale is stored by no other head."""
    return _NVFP4_HEAD_MARK in names


def head_graph(spec: Exl3HeadSpec | Nvfp4HeadSpec):
    """The head program: f32 logits of ``x``, any number of ``(rows, hidden)`` rows (the ``num_tokens`` axis
    serving's any-width programs share), compiled and measured at :data:`HEAD_ROWS`."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim("num_tokens", hint=HEAD_ROWS), spec.hidden_size), spec.dtype), node_id="x")
    graph.inputs = ["x"]
    graph.outputs = [spec.spell(graph, "x")]
    return graph


class CodedHead(nn.Module):
    """One persistent coded output head in the model's ``lm_head`` slot, so a speculative drafter that shares the
    target's head shares this one. The spec spells the logits over its codes, which upload once and stay
    pointer-stable; a call runs the sampled rows through the program in groups of at most ``capacity``, one launch
    per group."""

    def __init__(self, spec: Exl3HeadSpec | Nvfp4HeadSpec, source: dict, *, capacity: int):
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.cuda.program import CompiledProgram
        from emmy.compiler.backend.gpu_lock import gpu_lock
        from emmy.compiler.backend.plan import plan_from_graph

        super().__init__()
        plan = plan_from_graph(CudaBackend(tune_db="auto").compile(head_graph(spec)))
        if not plan.launches:
            raise RuntimeError("coded lm_head produced no compiler launches")
        # Built at capacity: a call sizes the launch grids to its rows, and the buffers never move.
        feed = {"x": np.zeros((capacity, spec.hidden_size), dtype=np.float32), **spec.feed(source)}
        with gpu_lock():
            self.program = CompiledProgram.build_from_plan(plan, feed)
        self.spec, self.capacity = spec, capacity

    def forward(self, hidden_states):
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        rows = hidden_states.reshape(-1, self.spec.hidden_size)
        if rows.dtype != getattr(torch, self.spec.dtype) or not rows.is_cuda:
            raise ValueError(f"coded lm_head requires CUDA {self.spec.dtype} input, got {rows.device}/{rows.dtype}")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and rows.shape[0] > self.capacity:
            raise ValueError(f"captured coded lm_head takes at most {self.capacity} rows, got {rows.shape[0]}")
        output = []
        with gpu_lock(), self.program.on_stream(torch.cuda.current_stream()):
            for group in rows.split(self.capacity):
                sizes = {"num_tokens": group.shape[0]}
                self.program.set_sym_values(sizes)
                self.program.upload_prefix_device({"x": group.contiguous()})
                self.program.run_once()
                logits = self.program.output_prefix_device(sizes)["logits"][:, : self.spec.vocab_size]
                output.append(logits if capturing else logits.clone())
        return output[0] if len(output) == 1 else torch.cat(output, dim=0)
