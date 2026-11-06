"""The normal-form and identity memos, counted through a splice."""

from __future__ import annotations

from emmy.compiler.ir.expression import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import (
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Write,
    splice_loops,
)


def test_large_sigma_target_free_vars_walks_do_not_scale_with_producer_chain(monkeypatch):
    """A long producer chain reuses one large coordinate target without re-walking its tree."""

    def splice(chain_length):
        from emmy.compiler.ir.stmt.body import _normal_form
        from emmy.compiler.ir.stmt.identity import _identity

        # Both runs measure the computation, not a memo hit: the consumer is the same body both times.
        _normal_form.cache_clear()
        _identity.cache_clear()

        producer_axis = Axis("p", 4096)
        producer_stmts = [Load(name="x", input="X", index=(Var("p"),))]
        previous = "x"
        for i in range(chain_length):
            name = f"v{i}"
            producer_stmts.append(Assign(name=name, op="negative", args=(previous,)))
            previous = name
        producer_stmts.append(Write(output="P", index=(Var("p"),), value=previous))
        producer = LoopOp(body=(Loop(axis=producer_axis, body=tuple(producer_stmts)),))

        i_axis, j_axis = Axis("i", 8), Axis("j", 8)
        target = Var("i")
        for _ in range(128):
            target = target * 2 + Var("j")
        target = BinaryExpr("^", target, Literal(1, "int"))
        consumer = LoopOp(
            body=(
                Loop(
                    axis=i_axis,
                    body=(
                        Loop(
                            axis=j_axis,
                            body=(
                                Load(name="pv", input="P", index=(target,)),
                                Write(output="OUT", index=(Var("i"), Var("j")), value="pv"),
                            ),
                        ),
                    ),
                ),
            )
        )

        root_walks = 0
        original = BinaryExpr.free_vars

        def counted_free_vars(expression):
            nonlocal root_walks
            if expression.op == "^":
                root_walks += 1
            return original(expression)

        with monkeypatch.context() as patch:
            patch.setattr(BinaryExpr, "free_vars", counted_free_vars)
            merged = splice_loops(loops={"producer": producer, "consumer": consumer}, splice_edges={("consumer", "P"): ("producer", "P")})

        assert merged is not None
        return root_walks

    # Reserving free operands and normalization each inspect the target a fixed number of times.
    # Compare chain lengths so a constant amount of necessary analysis does not trip the guard.
    assert splice(128) <= splice(8)
