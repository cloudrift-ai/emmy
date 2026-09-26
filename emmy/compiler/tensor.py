"""``Tensor`` — symbolic descriptor of a tensor-shaped buffer.

Holds the three things every consumer asks for: a name, a shape, and a
:class:`DataType`. Reused as the per-node ``Node.output`` value in the
graph, as the per-buffer descriptor on ``KernelOp`` (kernel signature),
and as the render-time ``tensors`` map for index flattening.

``dtype`` accepts a :class:`DataType` directly or any string spelling
that :func:`emmy.compiler.dtype.get` resolves (canonical name,
PyTorch alias, etc.); ``__post_init__`` coerces to the canonical
:class:`DataType` so downstream code never sees a bare string.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dim import Dim, to_dim
from emmy.compiler.dtype import F32, DataType
from emmy.compiler.dtype import get as _get_dtype
from emmy.compiler.wire import Wire


@dataclass
class Tensor(Wire):
    """Multidimensional array descriptor.

    ``shape`` is a tuple of :class:`Dim`. Construction coerces bare
    ``int`` / ``str`` elements to ``Dim`` so existing call sites that
    pass ``(1, 32, 2048)`` keep working.

    ``constant`` marks tensors whose value is fixed at compile time
    (``ConstantOp.output`` — weights, RoPE tables, scalar literals).
    ``value`` carries the captured scalar when the constant is a
    0-D float (``ConstantOp.value is not None``); otherwise ``None``.
    Together they let downstream consumers (cuda lowering, the load
    vectorizer) recognize a scalar-literal buffer without re-querying
    the graph for ``ConstantOp`` predecessors.

    ``transient`` marks a buffer that only ever existed inside a rewrite
    fragment — a decomposition's private intermediate, a split's f32
    workspace. It has no storage in the program the trace described, so
    its ``dtype`` carries SHAPE, not a rounding boundary: fusing the
    buffer away deletes no rounding the reference performed. A
    non-transient buffer is a tensor the source program materialized, so
    a producer that computes wider than ``dtype`` rounds there and
    fusion has to keep that rounding (``ir/loop/splicer.py``).
    :meth:`Graph.splice` is the ONE place that answers this — every
    fragment-internal buffer is transient and a replacement inherits the
    replaced buffer's answer — so no rewrite rule restates it and no
    consumer reconstructs it from op history.
    """

    def to_wire(self) -> list:
        """``[name, dtype, shape]`` — a buffer's identity; whether it is constant or transient is the graph's."""
        return [self.name, self.dtype.name, [dim.to_wire() for dim in self.shape]]

    @classmethod
    def from_wire(cls, value: object, where: str = "tensor") -> Tensor:
        if not isinstance(value, list) or len(value) != 3 or not isinstance(value[2], list):
            raise ValueError(f"{where} must be [name, dtype, shape]")
        name, dtype, shape = value
        if not isinstance(name, str) or not name or not isinstance(dtype, str) or not dtype:
            raise ValueError(f"{where} needs a non-empty name and dtype")
        return cls(name=name, dtype=dtype, shape=tuple(Dim.from_wire(dim, f"{where}.shape") for dim in shape))

    name: str
    shape: tuple[Dim, ...]
    dtype: DataType = F32
    constant: bool = False
    value: float | None = None
    transient: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DataType):
            self.dtype = _get_dtype(self.dtype)
        if any(not isinstance(d, Dim) for d in self.shape):
            self.shape = tuple(to_dim(d) for d in self.shape)
