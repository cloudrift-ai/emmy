"""The normal-form and identity memos, counted through a splice."""

from __future__ import annotations

from emmy.compiler.ir.expression import BinaryExpr
from emmy.compiler.ir.loop import LoopOp, splice_loops
from emmy.compiler.ir.stmt import parse_body


def test_large_sigma_target_free_vars_walks_do_not_scale_with_producer_chain(monkeypatch):
    """A long producer chain reuses one large coordinate target without re-walking its tree."""

    def splice(chain_length):
        from emmy.compiler.ir.stmt.body import _normal_form
        from emmy.compiler.ir.stmt.identity import _identity

        # Both runs measure the computation, not a memo hit: the consumer is the same body both times.
        _normal_form.cache_clear()
        _identity.cache_clear()

        producer_stmts = ["x = load X[p]"]
        previous = "x"
        for i in range(chain_length):
            name = f"v{i}"
            producer_stmts.append(f"{name} = negative({previous})")
            previous = name
        producer_stmts.append(f"P[p] = {previous}")
        producer = LoopOp(body=parse_body("\n".join(("for p in 0..4096:", *(f"    {line}" for line in producer_stmts)))))

        target = "i"
        for _ in range(128):
            target = f"({target} * 2 + j)"
        consumer = LoopOp(
            body=parse_body(
                f"""
                for i in 0..8:
                    for j in 0..8:
                        pv = load P[{target} ^ 1]
                        OUT[i, j] = pv
                """
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
