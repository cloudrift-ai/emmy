"""Loop IR → LLVM IR for one kernel.

``generate`` first tries a parallel strategy chosen from the loop nest's shape (pointwise, full reduction, row
reduction, contraction, split-K contraction). A kernel outside those shapes lowers through ``_serial``: the body as
written, split across threads only on an outermost loop whose iterations are independent. A kernel neither can
express raises :class:`Unsupported`. Every strategy checks its preconditions before it emits anything.

Every kernel is ``part(bufs, lo, hi, partial, sizes)`` over ``[lo, hi)`` of one axis. Reductions over that axis
write per-chunk partials and add ``finish(bufs, partials, nchunks, sizes)``, which combines them and runs the
epilogue. ``sizes`` holds the runtime symbolic extents in :attr:`Plan.sizes` order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import llvmlite.binding as llvm
import llvmlite.ir as ir

F32, I64, I32, I16, I1 = ir.FloatType(), ir.IntType(64), ir.IntType(32), ir.IntType(16), ir.IntType(1)
HALF, PTR, VOID = ir.HalfType(), ir.PointerType(), ir.VoidType()
STORAGE = {"f32": F32, "f16": HALF, "bf16": I16}
IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": float("-inf"), "minimum": float("inf")}
# Only the accumulating instruction may reassociate, so the vectorizer can split a sum into lanes while
# every other operation keeps IEEE rounding.
REASSOC = ("reassoc", "nsz")
EXPENSIVE_OPS = frozenset({"exp", "exp_fast", "log", "erf", "tanh", "sin", "cos", "sigmoid", "silu", "rsqrt", "sqrt"})
# Waking the pool costs tens of microseconds, so smaller kernels stay on one thread. Work is loop
# iterations x statements, a transcendental counting as eight.
PARALLEL_MIN_COST = 1 << 20
MAX_ACC_ARRAY = 1 << 16
# Each thread's partials start on their own 128-byte line (Apple silicon's cache line), so threads
# finishing a reduction never write the same line.
PARTIAL_ALIGN_FLOATS = 32


def _partial_stride(floats: int) -> int:
    return -(-floats // PARTIAL_ALIGN_FLOATS) * PARTIAL_ALIGN_FLOATS


class Unsupported(Exception):
    """The kernel uses a construct this lowering does not cover."""


@dataclass(frozen=True)
class Plan:
    strategy: str
    mode: str  # "range": independent chunks; "reduce": chunks write partials, then ``finish``
    extent: int | str  # length of the split axis; a name is a runtime size
    partial_floats: int  # one thread's partials, padded to whole cache lines
    parallel: bool
    sizes: tuple[str, ...] = ()


@dataclass
class _Nest:
    loops: list
    inner: list
    pre: dict[int, list] = field(default_factory=dict)
    post: dict[int, list] = field(default_factory=dict)


def _kind(s) -> str:
    return type(s).__name__


def _ext(loop):
    return loop.axis.extent.value


def _cost(stmts) -> int:
    total = 0
    for s in stmts:
        if _kind(s) in ("Loop", "StridedLoop"):
            extent = s.axis.extent
            total += (extent.value if isinstance(extent.value, int) else (extent.hint or 512)) * _cost(s.body)
        elif _kind(s) == "Cond":
            total += _cost(s.body) + _cost(s.else_body or ())
        elif _kind(s) == "Assign" and s.op.name in EXPENSIVE_OPS:
            total += 8
        else:
            total += 1
    return total


def _loop_sizes(stmts) -> set[str]:
    names: set[str] = set()
    for s in stmts:
        if _kind(s) in ("Loop", "StridedLoop") and isinstance(_ext(s), str):
            names.add(_ext(s))
        for attr in ("body", "else_body"):
            names |= _loop_sizes(getattr(s, attr, None) or ())
    return names


def _check_leaves(stmts) -> None:
    for s in stmts:
        if _kind(s) not in ("Load", "Assign", "Accum", "Write"):
            raise Unsupported(f"statement {_kind(s)}")
        if _kind(s) == "Write" and s.atomic:
            raise Unsupported("atomic write")
        if _kind(s) == "Accum" and (s.base is not None or s.op.name not in IDENTITY):
            raise Unsupported(f"accumulator {s.op.name}")


def _nest(body) -> _Nest:
    """A chain of single nested loops, each level optionally preceded by loop-invariant loads/assigns."""
    loops, pre, post = [], {}, {}
    level_body, level = list(body), -1
    while True:
        inner = [s for s in level_body if _kind(s) == "Loop"]
        if not inner:
            return _Nest(loops, level_body, pre, post)
        if len(inner) > 1:
            raise Unsupported("sibling loops")
        idx = level_body.index(inner[0])
        if any(_kind(s) not in ("Load", "Assign") for s in level_body[:idx]):
            raise Unsupported("statements before a nested loop")
        pre[level], post[level] = level_body[:idx], level_body[idx + 1 :]
        loops.append(inner[0])
        level_body, level = list(inner[0].body), level + 1


def _walks(load, axis: str) -> bool:
    """Whether stepping ``axis`` moves ``load`` along its contiguous (last) dimension."""
    return bool(load.index) and axis in load.index[-1].free_vars()


class _Fn:
    """One LLVM function: locals as allocas (mem2reg promotes them), loops as phi nodes, buffers by name."""

    def __init__(self, module, name, params, buffers, shapes, dtypes, size_names):
        self.m, self.shapes, self.dtypes = module, shapes, dtypes
        self.fn = ir.Function(module, ir.FunctionType(VOID, [PTR, *params]), name)
        for arg in self.fn.args:
            if isinstance(arg.type, ir.PointerType):
                arg.attributes.add("noalias")
        self.allocas = ir.IRBuilder(self.fn.append_basic_block("allocas"))
        self.b = ir.IRBuilder(self.fn.append_basic_block("body"))
        bufs, sizes = self.fn.args[0], self.fn.args[-1]
        self.buf = {
            n: self.b.load(self.b.gep(bufs, [ir.Constant(I64, i)], source_etype=PTR), typ=PTR, name=n) for i, n in enumerate(buffers)
        }
        self.size = {
            n: self.b.load(self.b.gep(sizes, [ir.Constant(I64, i)], source_etype=I64), typ=I64, name=n) for i, n in enumerate(size_names)
        }
        self.vars, self.axes, self.acc_arrays, self.lets = {}, {}, {}, {}
        self.open_accs: set[str] = set()

    def finalize(self):
        self.b.ret_void()
        self.allocas.branch(self.fn.basic_blocks[1])

    def dim(self, d):
        if isinstance(d, int):
            return ir.Constant(I64, d)
        if d not in self.size:
            raise Unsupported(f"runtime size {d!r} not bound")
        return self.size[d]

    def slot(self, name):
        if name not in self.vars:
            self.vars[name] = self.allocas.alloca(F32, name=name)
        return self.vars[name]

    def acc_array(self, name, n):
        self.acc_arrays[name] = self.allocas.alloca(F32, size=ir.Constant(I64, n), name=name)

    def addr(self, name, acc_index=None):
        if acc_index and name in acc_index:
            return self.b.gep(self.acc_arrays[name], [acc_index[name]], source_etype=F32)
        return self.slot(name)

    def loop(self, axis, lo, hi, body):
        b, pre = self.b, self.b.block
        head, inner, done = (self.fn.append_basic_block(f"{axis}.{s}") for s in ("head", "body", "done"))
        b.branch(head)
        b.position_at_end(head)
        i = b.phi(I64, name=axis)
        i.add_incoming(lo, pre)
        b.cbranch(b.icmp_signed("<", i, hi), inner, done)
        b.position_at_end(inner)
        self.axes[axis] = i
        body()
        i.add_incoming(b.add(i, ir.Constant(I64, 1)), b.block)
        b.branch(head)
        b.position_at_end(done)
        del self.axes[axis]

    def full(self, loop_stmt, body):
        self.loop(loop_stmt.axis.name, ir.Constant(I64, 0), self.dim(_ext(loop_stmt)), body)

    def branch(self, predicate, then, otherwise=None):
        with self.b.if_else(predicate) as (on_true, on_false):
            with on_true:
                then()
            with on_false:
                if otherwise is not None:
                    otherwise()

    def expr(self, e):
        """An index, predicate or scalar expression: i64 for integers, i1 for predicates, f32 for SSA values."""
        k = _kind(e)
        b = self.b
        if k == "Literal":
            if isinstance(e.value, bool):
                return ir.Constant(I1, int(e.value))
            if isinstance(e.value, int):
                return ir.Constant(I64, e.value)
            return ir.Constant(F32, float(e.value))
        if k == "Var":
            if e.name in self.axes:
                return self.axes[e.name]
            if e.name in self.lets:
                return self.lets[e.name]
            if e.name in self.vars:
                return b.load(self.vars[e.name], typ=F32)
            raise Unsupported(f"unbound variable {e.name}")
        if k == "CastExpr":
            v = self.expr(e.expr)
            if e.dtype in ("float", "f32"):
                return v if v.type == F32 else b.sitofp(self.integer(v), F32)
            if e.dtype == "int":
                return b.fptosi(v, I64) if v.type == F32 else self.integer(v)
            raise Unsupported(f"cast to {e.dtype}")
        if k == "TernaryExpr":
            on_true, on_false = self.expr(e.if_true), self.expr(e.if_false)
            if F32 in (on_true.type, on_false.type):
                on_true, on_false = self.real(on_true), self.real(on_false)
            else:
                on_true, on_false = self.integer(on_true), self.integer(on_false)
            return b.select(self.pred(e.cond), on_true, on_false)
        if k == "BinaryExpr":
            op = e.op
            if op in ("&&", "||"):
                return (b.and_ if op == "&&" else b.or_)(self.pred(e.left), self.pred(e.right))
            left, right = self.expr(e.left), self.expr(e.right)
            if F32 in (left.type, right.type):
                left, right = self.real(left), self.real(right)
                arith = {"+": b.fadd, "-": b.fsub, "*": b.fmul, "/": b.fdiv}
                if op in arith:
                    return arith[op](left, right)
                if op in ("<", "<=", ">", ">=", "==", "!="):
                    return b.fcmp_ordered(op, left, right) if op != "!=" else b.fcmp_unordered(op, left, right)
            else:
                left, right = self.integer(left), self.integer(right)
                if op in ("/", "//", "%"):
                    return self.floor_div_mod(left, right, op == "%")
                arith = {"+": b.add, "-": b.sub, "*": b.mul, "^": b.xor}
                if op in arith:
                    return arith[op](left, right)
                if op in ("<", "<=", ">", ">=", "==", "!="):
                    return b.icmp_signed(op, left, right)
        raise Unsupported(f"expression {k} {getattr(e, 'op', '')}".strip())

    def floor_div_mod(self, a, d, mod):
        """Python's integer ``//`` and ``%``, which round toward negative infinity where ``sdiv`` truncates."""
        b, zero = self.b, ir.Constant(I64, 0)
        q, r = b.sdiv(a, d), b.srem(a, d)
        adjust = b.and_(b.icmp_signed("!=", r, zero), b.icmp_signed("!=", b.icmp_signed("<", r, zero), b.icmp_signed("<", d, zero)))
        if mod:
            return b.select(adjust, b.add(r, d), r)
        return b.select(adjust, b.sub(q, ir.Constant(I64, 1)), q)

    def integer(self, v):
        if v.type == F32:
            raise Unsupported("float value used as an index")
        return self.b.zext(v, I64) if v.type == I1 else v

    def real(self, v):
        return v if v.type == F32 else self.b.sitofp(self.integer(v), F32)

    def index(self, e):
        return self.integer(self.expr(e))

    def pred(self, e):
        v = self.expr(e)
        if v.type == F32:
            return self.b.fcmp_unordered("!=", v, ir.Constant(F32, 0.0))
        return v if v.type == I1 else self.b.icmp_signed("!=", v, ir.Constant(I64, 0))

    def element(self, buf, index_exprs):
        """Row-major address of ``buf[index...]``; a runtime dimension makes the stride a runtime product."""
        shape = self.shapes[buf]
        if len(index_exprs) not in (0, len(shape)):
            raise Unsupported(f"{buf}: {len(index_exprs)} indices for rank {len(shape)}")
        offset, stride = ir.Constant(I64, 0), 1
        for e, d in zip(reversed(index_exprs), reversed(shape), strict=False):
            term = self.index(e)
            if not (isinstance(stride, int) and stride == 1):
                term = self.b.mul(term, ir.Constant(I64, stride) if isinstance(stride, int) else stride)
            offset = self.b.add(offset, term)
            if isinstance(stride, int) and isinstance(d, int):
                stride *= d
            else:
                stride = self.b.mul(ir.Constant(I64, stride) if isinstance(stride, int) else stride, self.dim(d))
        return self.b.gep(self.buf[buf], [offset], source_etype=STORAGE[self.dtypes[buf]])

    def widen(self, v, dtype):
        if dtype == "f16":
            return self.b.fpext(v, F32)
        if dtype == "bf16":
            return self.b.bitcast(self.b.shl(self.b.zext(v, I32), ir.Constant(I32, 16)), F32)
        return v

    def narrow(self, v, dtype):
        if dtype == "f16":
            return self.b.fptrunc(v, HALF)
        if dtype == "bf16":
            b = self.b
            bits = b.bitcast(v, I32)
            odd = b.and_(b.lshr(bits, ir.Constant(I32, 16)), ir.Constant(I32, 1))
            rounded = b.lshr(b.add(b.add(bits, ir.Constant(I32, 0x7FFF)), odd), ir.Constant(I32, 16))
            quiet_nan = b.or_(b.lshr(bits, ir.Constant(I32, 16)), ir.Constant(I32, 0x40))
            return b.trunc(b.select(b.fcmp_unordered("uno", v, v), quiet_nan, rounded), I16)
        return v

    def round_to(self, v, dtype):
        return self.widen(self.narrow(v, dtype), dtype) if dtype in ("f16", "bf16") else v

    def value(self, arg, acc_index=None):
        if isinstance(arg, str) and (arg in self.vars or arg in self.acc_arrays):
            return self.b.load(self.addr(arg, acc_index), typ=F32)
        if isinstance(arg, str) and arg in self.lets:
            return self.real(self.lets[arg])
        try:
            return ir.Constant(F32, float(arg))
        except (TypeError, ValueError):
            raise Unsupported(f"operand {arg!r}") from None

    def intrinsic(self, name):
        arity = 2 if name in ("llvm.maxnum", "llvm.minnum", "llvm.pow") else 1
        return self.m.declare_intrinsic(name, [F32], ir.FunctionType(F32, [F32] * arity))

    def libm(self, name):
        if name not in self.m.globals:
            ir.Function(self.m, ir.FunctionType(F32, [F32]), name)
        return self.m.globals[name]

    def elementwise(self, op, args):
        b, one = self.b, ir.Constant(F32, 1.0)
        binary = {"add": b.fadd, "subtract": b.fsub, "multiply": b.fmul, "divide": b.fdiv}
        if op in binary:
            return binary[op](*args)
        if op in ("copy", "pad"):
            return args[0]
        if op == "next":
            return args[1]
        if op == "where":
            return b.select(b.fcmp_unordered("!=", args[0], ir.Constant(F32, 0.0)), args[1], args[2])
        if op == "reciprocal":
            return b.fdiv(one, args[0])
        if op == "square":
            return b.fmul(args[0], args[0])
        if op == "power":
            return b.call(self.intrinsic("llvm.pow"), args)
        if op in ("floor", "ceil"):
            return b.call(self.intrinsic("llvm." + op), args)
        if op == "softplus":
            x = args[0]
            tail = b.call(self.libm("log1pf"), [self.exp(b.fneg(b.call(self.intrinsic("llvm.fabs"), [x])))])
            return b.fadd(b.call(self.intrinsic("llvm.maxnum"), [x, ir.Constant(F32, 0.0)]), tail)
        if op == "gelu":
            x = args[0]
            cdf = b.fadd(one, b.call(self.libm("erff"), [b.fmul(x, ir.Constant(F32, 0.7071067811865476))]))
            return b.fmul(b.fmul(ir.Constant(F32, 0.5), x), cdf)
        if op == "gelu_tanh":
            x = args[0]
            inner = b.fadd(x, b.fmul(ir.Constant(F32, 0.044715), b.fmul(x, b.fmul(x, x))))
            t = b.call(self.libm("tanhf"), [b.fmul(ir.Constant(F32, 0.7978845608028654), inner)])
            return b.fmul(b.fmul(ir.Constant(F32, 0.5), x), b.fadd(one, t))
        if op == "negative":
            return b.fneg(args[0])
        if op == "abs":
            return b.call(self.intrinsic("llvm.fabs"), args)
        if op == "relu":
            return b.call(self.intrinsic("llvm.maxnum"), [args[0], ir.Constant(F32, 0.0)])
        if op in ("maximum", "minimum"):
            return b.call(self.intrinsic("llvm.maxnum" if op == "maximum" else "llvm.minnum"), args)
        if op in ("exp", "exp_fast"):
            return self.exp(args[0])
        if op == "sigmoid":
            return b.fdiv(one, b.fadd(one, self.exp(b.fneg(args[0]))))
        if op == "silu":
            return b.fdiv(args[0], b.fadd(one, self.exp(b.fneg(args[0]))))
        if op in ("sqrt", "log"):
            return b.call(self.intrinsic("llvm." + op), args)
        if op == "rsqrt":
            return b.fdiv(one, b.call(self.intrinsic("llvm.sqrt"), args))
        if op in ("erf", "tanh", "sin", "cos"):
            return b.call(self.libm(op + "f"), args)
        raise Unsupported(f"elementwise op {op}")

    def exp(self, x):
        """exp(x) = 2^n * e^r, n = rint(x / ln2), e^r from the Cephes expf polynomial.
        Plain arithmetic, so the loop vectorizer can widen it on any CPU, unlike a libm call."""
        b, c = self.b, (lambda v: ir.Constant(F32, v))
        clamped = b.call(self.intrinsic("llvm.minnum"), [b.call(self.intrinsic("llvm.maxnum"), [x, c(-87.33654)]), c(88.72283)])
        xc = b.select(b.fcmp_unordered("uno", x, x), x, clamped)
        n = b.call(self.intrinsic("llvm.rint"), [b.fmul(xc, c(1.44269504088896341))])
        r = b.fsub(b.fsub(xc, b.fmul(n, c(0.693359375))), b.fmul(n, c(-2.12194440e-4)))
        poly = c(1.9875691500e-4)
        for k in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1):
            poly = b.fadd(b.fmul(poly, r), c(k))
        er = b.fadd(b.fadd(b.fmul(b.fmul(poly, r), r), r), c(1.0))
        # 2^n as two halves: n reaches 128 just below the overflow bound, past the largest float exponent.
        n_int = b.fptosi(n, I32)
        half = b.ashr(n_int, ir.Constant(I32, 1))

        def pow2(e):
            return b.bitcast(b.shl(b.add(e, ir.Constant(I32, 127)), ir.Constant(I32, 23)), F32)

        scaled = b.fmul(b.fmul(er, pow2(half)), pow2(b.sub(n_int, half)))
        return b.select(b.fcmp_ordered("<", x, c(-87.33654)), c(0.0), scaled)

    def combine(self, op, cur, v):
        if op == "add":
            return self.b.fadd(cur, v, flags=REASSOC)
        if op == "multiply":
            return self.b.fmul(cur, v, flags=REASSOC)
        if op in ("maximum", "minimum"):
            return self.b.call(self.intrinsic("llvm.maxnum" if op == "maximum" else "llvm.minnum"), [cur, v], fastmath=("nnan", "nsz"))
        raise Unsupported(f"accumulator {op}")

    def leaves(self, stmts, acc_index=None, *, serial=False):
        for s in stmts:
            k = _kind(s)
            if k == "Load":
                if len(s.names) != 1:
                    raise Unsupported("vector load")
                dt = self.dtypes[s.input]
                raw = self.b.load(self.element(s.input, s.index), typ=STORAGE[dt])
                self.b.store(self.widen(raw, dt), self.slot(s.names[0]))
            elif k == "Assign":
                result = self.elementwise(s.op.name, [self.value(a, acc_index) for a in s.args])
                self.b.store(self.round_to(result, str(s.dtype)), self.slot(s.name))
            elif k == "Accum":
                where = self.addr(s.name, acc_index)
                cur = self.value(s.base, acc_index) if s.base is not None else self.b.load(where, typ=F32)
                self.b.store(self.combine(s.op.name, cur, self.value(s.value, acc_index)), where)
            elif k == "Select":
                result = self.value(s.branches[-1].value, acc_index)
                for branch in reversed(s.branches[:-1]):
                    result = self.b.select(self.pred(branch.select), self.value(branch.value, acc_index), result)
                self.b.store(result, self.slot(s.name))
            elif k == "Let":
                v = self.expr(s.value)
                if v.type == F32:
                    self.b.store(v, self.slot(s.name))
                else:
                    self.lets[s.name] = v
            elif k == "Write":
                if len(s.values) != 1:
                    raise Unsupported("vector write")
                dt = self.dtypes[s.output]
                where = self.element(s.output, s.index)
                v = self.value(s.values[0], acc_index)
                if s.atomic:
                    if not serial:
                        raise Unsupported("atomic write")
                    v = self.b.fadd(self.widen(self.b.load(where, typ=STORAGE[dt]), dt), v)
                self.b.store(self.narrow(v, dt), where)
            else:
                raise Unsupported(f"statement {k}")

    def init_accs(self, accums, acc_index=None):
        for a in accums:
            self.b.store(ir.Constant(F32, IDENTITY[a.op.name]), self.addr(a.name, acc_index))

    def serial(self, stmts):
        """The body as written. An accumulator resets at the outermost loop over one of its reduce axes, or, when
        it names no axes, at the loop directly holding it."""
        for s in stmts:
            k = _kind(s)
            if k == "Loop":
                if getattr(s, "carries", None):
                    raise Unsupported("loop-carried state")
                opened = [
                    a
                    for a in _accums_in(s.body)
                    if a.name not in self.open_accs and (s.axis.name in a.axes or (not a.axes and any(a is t for t in s.body)))
                ]
                for a in opened:
                    if a.op.name not in IDENTITY:
                        raise Unsupported(f"accumulator {a.op.name}")
                self.init_accs(opened)
                self.open_accs |= {a.name for a in opened}
                self.full(s, lambda s=s: self.serial(s.body))
                self.open_accs -= {a.name for a in opened}
            elif k == "Cond":
                self.branch(
                    self.pred(s.cond), lambda s=s: self.serial(s.body), (lambda s=s: self.serial(s.else_body)) if s.else_body else None
                )
            else:
                self.leaves([s], serial=True)


def _accums_in(stmts):
    for s in stmts:
        if _kind(s) == "Accum":
            yield s
        for attr in ("body", "else_body"):
            yield from _accums_in(getattr(s, attr, None) or ())


def _nest_outer(fn: _Fn, loops, innermost, pre=None, level=0):
    if not loops:
        innermost()
        return
    if pre is not None:
        pre(level)
    fn.full(loops[0], lambda: _nest_outer(fn, loops[1:], innermost, pre, level + 1))


def _split_index(loops) -> int:
    """The loop a range strategy splits across threads: the first of more than one iteration, so a leading
    batch axis of 1 does not leave every thread but one idle."""
    return next((k for k, lp in enumerate(loops) if _ext(lp) != 1), 0)


def _split_nest(fn: _Fn, loops, lo, hi, innermost, pre=None):
    """Open ``loops`` with the split loop restricted to ``[lo, hi)``. Returns the split loop."""
    split = _split_index(loops)

    def open_from(k):
        if k > 0 and pre is not None:
            pre(k - 1)
        if k == split:
            fn.loop(loops[k].axis.name, lo, hi, lambda: _nest_outer(fn, loops[k + 1 :], innermost, pre, k))
        else:
            fn.full(loops[k], lambda: open_from(k + 1))

    open_from(0)
    return loops[split]


def generate(loop, shapes: dict, dtypes: dict) -> tuple[str, Plan]:
    """LLVM IR for one ``LoopOp`` and the plan for running it. ``shapes`` maps every buffer to a tuple of
    ints or runtime size names; ``dtypes`` to ``"f32"``, ``"f16"`` or ``"bf16"``."""
    buffers = [*loop.inputs, *loop.outputs]
    if any(dtypes[b] not in STORAGE for b in buffers):
        raise Unsupported(f"buffer dtypes {sorted({dtypes[b] for b in buffers})}")
    size_names = tuple(sorted({d for b in buffers for d in shapes[b] if isinstance(d, str)} | _loop_sizes(loop.body)))
    try:
        return _parallel(loop, buffers, shapes, dtypes, size_names)
    except Unsupported:
        return _serial(loop, buffers, shapes, dtypes, size_names)


def _module():
    m = ir.Module(name="emmy_cpu")
    m.triple = llvm.get_default_triple()
    return m


def _writes_in(stmts):
    for s in stmts:
        if _kind(s) == "Write":
            yield s
        for attr in ("body", "else_body"):
            yield from _writes_in(getattr(s, attr, None) or ())


def _disjoint(writes, loop_stmt) -> bool:
    """Each iteration of ``loop_stmt`` writes its own cells: every write's index depends on the loop's axis. A write
    that ignores the axis (``out[j]`` under a loop over ``i``) would race; an index that merely folds the axis
    (``out[i // 2]``) is not caught, and Emmy's lowered nests write each cell from one iteration."""
    axis = loop_stmt.axis.name
    return _ext(loop_stmt) == 1 or all(not w.atomic and any(axis in e.free_vars() for e in w.index) for w in writes)


def _require_disjoint(writes, loops) -> None:
    if not _disjoint(writes, loops[_split_index(loops)]):
        raise Unsupported("a write does not index the split axis")


def _independent(loop_stmt) -> bool:
    """Iterations of ``loop_stmt`` touch disjoint output cells and share no running state."""
    axis = loop_stmt.axis.name
    if getattr(loop_stmt, "carries", None) or any(axis in a.axes for a in _accums_in(loop_stmt.body)):
        return False
    return _disjoint(list(_writes_in(loop_stmt.body)), loop_stmt)


def _observes_running_value(stmts, accums) -> bool:
    """Whether anything but the folds themselves reads an accumulator, or writes, inside its reduce loop."""
    names = {a.name for a in accums}
    for s in stmts:
        if _kind(s) == "Write":
            return True
        reads = {s.value} | ({s.base} - {s.name} if s.base else set()) if _kind(s) == "Accum" else set(s.deps())
        if names & reads:
            return True
    return False


def _serial(loop, buffers, shapes, dtypes, size_names):
    """The body as written; its outermost independent loop splits across threads."""
    m = _module()
    part = _Fn(m, "part", [I64, I64, PTR, PTR], buffers, shapes, dtypes, size_names)
    lo, hi = part.fn.args[1:3]
    body = list(loop.body)
    loops = [s for s in body if _kind(s) == "Loop"]
    head = body[: body.index(loops[0])] if len(loops) == 1 else []
    split = loops[0] if len(loops) == 1 and body[len(head) + 1 :] == [] else None
    while split is not None and _ext(split) == 1 and len(split.body) == 1 and _kind(split.body[0]) == "Loop":
        part.axes[split.axis.name] = ir.Constant(I64, 0)
        split = split.body[0]
    if split is None or any(_kind(s) not in ("Load", "Assign", "Let") for s in head) or not _independent(split):
        part.serial(body)
        part.finalize()
        return str(m), Plan("serial", "range", 1, 0, False, size_names)
    part.serial(head)
    part.loop(split.axis.name, lo, hi, lambda: part.serial(list(split.body)))
    part.finalize()
    return str(m), Plan("serial", "range", _ext(split), 0, _cost(loop.body) >= PARALLEL_MIN_COST, size_names)


def _parallel(loop, buffers, shapes, dtypes, size_names):
    """Pick a strategy from the nest's shape. Every check runs before anything is emitted."""
    parallel = _cost(loop.body) >= PARALLEL_MIN_COST
    writes = list(_writes_in(loop.body))

    def start():
        m = _module()
        part = _Fn(m, "part", [I64, I64, PTR, PTR], buffers, shapes, dtypes, size_names)
        return m, part, *part.fn.args[1:4]

    def range_plan(strategy, extent):
        return Plan(strategy, "range", extent, 0, parallel, size_names)

    nest = _nest(loop.body)
    if not nest.loops:
        raise Unsupported("no loops")
    for stmts in (*nest.pre.values(), *nest.post.values()):
        _check_leaves(stmts)
    if any(_kind(s) == "Accum" for stmts in nest.post.values() for s in stmts):
        raise Unsupported("accumulation after a loop")
    _check_leaves(nest.inner)
    accums = [s for s in nest.inner if _kind(s) == "Accum"]

    if not accums:
        if any(nest.post.values()):
            raise Unsupported("writes outside the innermost loop")
        _require_disjoint(writes, nest.loops)
        m, part, lo, hi, _ = start()

        def pre(level):
            part.leaves(nest.pre.get(level, []))

        pre(-1)
        split = _split_nest(part, nest.loops, lo, hi, lambda: part.leaves(nest.inner), pre)
        part.finalize()
        return str(m), range_plan("pointwise", _ext(split))

    red_loop, par_loops = nest.loops[-1], nest.loops[:-1]
    red = red_loop.axis.name
    if {ax for a in accums for ax in a.axes} != {red}:
        raise Unsupported("reduction is not over exactly the innermost loop")
    post_level = len(par_loops) - 1
    if any(stmts for lvl, stmts in nest.post.items() if lvl != post_level):
        raise Unsupported("work after a loop at an unsupported level")
    post = nest.post.get(post_level, [])
    # Splitting the reduce loop gives each chunk its own running value, so nothing inside may observe it.
    splittable = not _observes_running_value(nest.inner, accums)

    if not par_loops:
        if not splittable:
            raise Unsupported("the reduce loop reads its running value")
        return _full_reduction(start, nest, accums, red_loop, post, buffers, shapes, dtypes, size_names, parallel)

    def walked(axis):
        return any(_walks(s, axis) and not _walks(s, red) for s in nest.inner if _kind(s) == "Load")

    # The free loops commute, so with no work between them the one a load walks contiguously can move innermost.
    if not walked(par_loops[-1].axis.name) and not any(nest.pre.get(level) for level in range(len(par_loops))):
        walker = next((lp for lp in par_loops[:-1] if walked(lp.axis.name)), None)
        if walker is not None:
            par_loops = [lp for lp in par_loops if lp is not walker] + [walker]
    j_loop, outer = par_loops[-1], par_loops[:-1]
    tile = (
        walked(j_loop.axis.name)
        and isinstance(_ext(j_loop), int)
        and _ext(j_loop) <= MAX_ACC_ARRAY
        and not any(nest.pre.get(level) for level in range(len(outer), len(par_loops)))
    )
    unit_outer = all(_ext(lp) == 1 for lp in outer)
    if tile and unit_outer and splittable:
        return _split_k(start, nest, accums, red_loop, j_loop, outer, post, buffers, shapes, dtypes, size_names, parallel)
    if tile and not unit_outer and _disjoint(writes, outer[_split_index(outer)]):
        return _contraction(start, nest, accums, red_loop, j_loop, outer, post, range_plan)
    _require_disjoint(writes, par_loops)
    m, part, lo, hi, _ = start()

    def pre(level):
        part.leaves(nest.pre.get(level, []))

    def row():
        pre(post_level)
        part.init_accs(accums)
        part.full(red_loop, lambda: part.leaves(nest.inner))
        part.leaves(post)

    pre(-1)
    split = _split_nest(part, par_loops, lo, hi, row, pre)
    part.finalize()
    return str(m), range_plan("row reduction", _ext(split))


def _full_reduction(start, nest, accums, red_loop, post, buffers, shapes, dtypes, size_names, parallel):
    m, part, lo, hi, partial = start()
    part.leaves(nest.pre.get(-1, []))
    part.init_accs(accums)
    part.loop(red_loop.axis.name, lo, hi, lambda: part.leaves(nest.inner))
    for k, a in enumerate(accums):
        part.b.store(part.b.load(part.slot(a.name), typ=F32), part.b.gep(partial, [ir.Constant(I64, k)], source_etype=F32))
    part.finalize()
    _finish(m, buffers, shapes, dtypes, size_names, accums, 1, post, None, nest.pre.get(-1, []))
    return str(m), Plan("full reduction", "reduce", _ext(red_loop), _partial_stride(len(accums)), parallel, size_names)


def _contraction(start, nest, accums, red_loop, j_loop, outer, post, range_plan):
    """One accumulator per column of ``j_loop``, so the loads that walk ``j`` read memory in order."""
    m, part, lo, hi, _ = start()
    ext_j = _ext(j_loop)

    def pre(level):
        part.leaves(nest.pre.get(level, []))

    def acc_idx():
        return {a.name: part.axes[j_loop.axis.name] for a in accums}

    for a in accums:
        part.acc_array(a.name, ext_j)
    pre(-1)

    def tile():
        pre(len(outer) - 1)
        part.full(j_loop, lambda: part.init_accs(accums, acc_idx()))
        part.full(red_loop, lambda: part.full(j_loop, lambda: part.leaves(nest.inner, acc_idx())))
        part.full(j_loop, lambda: part.leaves(post, acc_idx()))

    split = _split_nest(part, outer, lo, hi, tile, pre)
    part.finalize()
    return str(m), range_plan("contraction", _ext(split))


def _split_k(start, nest, accums, red_loop, j_loop, outer, post, buffers, shapes, dtypes, size_names, parallel):
    """Every outer loop runs once: bind their axes to 0 and split the reduction across threads instead."""
    m, part, lo, hi, partial = start()
    ext_j = _ext(j_loop)

    def acc_idx():
        return {a.name: part.axes[j_loop.axis.name] for a in accums}

    for lp in outer:
        part.axes[lp.axis.name] = ir.Constant(I64, 0)
    invariant = [s for level in range(-1, len(outer)) for s in nest.pre.get(level, [])]
    for k, a in enumerate(accums):
        part.acc_arrays[a.name] = part.b.gep(partial, [ir.Constant(I64, k * ext_j)], source_etype=F32)
    part.leaves(invariant)
    part.full(j_loop, lambda: part.init_accs(accums, acc_idx()))
    part.loop(red_loop.axis.name, lo, hi, lambda: part.full(j_loop, lambda: part.leaves(nest.inner, acc_idx())))
    part.finalize()
    _finish(m, buffers, shapes, dtypes, size_names, accums, ext_j, post, j_loop, invariant, [lp.axis.name for lp in outer])
    return str(m), Plan("contraction, split-K", "reduce", _ext(red_loop), _partial_stride(len(accums) * ext_j), parallel, size_names)


def _finish(m, buffers, shapes, dtypes, size_names, accums, width, post, j_loop, invariant, unit_axes=()):
    fin = _Fn(m, "finish", [PTR, I64, PTR], buffers, shapes, dtypes, size_names)
    partials, nchunks = fin.fn.args[1:3]
    for name in unit_axes:
        fin.axes[name] = ir.Constant(I64, 0)
    fin.leaves(list(invariant))
    per_chunk = _partial_stride(len(accums) * width)

    def combine_and_post():
        jv = fin.axes[j_loop.axis.name] if j_loop is not None else ir.Constant(I64, 0)
        fin.init_accs(accums)

        def add_chunk():
            c = fin.axes["chunk"]
            for k, a in enumerate(accums):
                off = fin.b.add(fin.b.mul(c, ir.Constant(I64, per_chunk)), fin.b.add(ir.Constant(I64, k * width), jv))
                v = fin.b.load(fin.b.gep(partials, [off], source_etype=F32), typ=F32)
                fin.b.store(fin.combine(a.op.name, fin.b.load(fin.slot(a.name), typ=F32), v), fin.slot(a.name))

        fin.loop("chunk", ir.Constant(I64, 0), nchunks, add_chunk)
        fin.leaves(post)

    if j_loop is None:
        combine_and_post()
    else:
        fin.full(j_loop, combine_and_post)
    fin.finalize()
