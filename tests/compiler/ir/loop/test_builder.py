"""Loop body construction and fresh-name allocation."""

from emmy.compiler.ir.loop.builder import BodyBuilder


class _LookupBudget(set):
    """Set whose deterministic probe budget catches suffix rescans without wall-clock assertions."""

    def __init__(self, budget: int):
        super().__init__()
        self.budget = budget
        self.lookups = 0

    def __contains__(self, value):
        self.lookups += 1
        if self.lookups > self.budget:
            raise AssertionError(f"fresh-name allocation exceeded {self.budget} membership probes")
        return super().__contains__(value)


def test_many_same_hint_allocations_stay_within_a_linear_probe_budget():
    count = 50_000
    builder = BodyBuilder(set())
    used = _LookupBudget(3 * count)
    builder._used = used

    last = None
    for _ in range(count):
        last = builder.fresh("v")

    assert last == f"v_s{count - 1}"
    assert used.lookups < 2 * count


def test_fresh_names_keep_lowest_available_suffix_across_hint_collisions():
    builder = BodyBuilder({"v", "v_s1", "v_s3"})

    assert builder.fresh("v") == "v_s2"
    assert builder.fresh("v_s4") == "v_s4"
    assert builder.fresh("v") == "v_s5"


def test_insert_descends_by_axis_and_prepends_at_the_leaf():
    """The builder's tree contract: descent creates missing Loops, a repeated scope reuses its
    Loop, and each level keeps prepend order — newest first, a fresh nested Loop ahead of the
    stmts inserted before it."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.loop import Scope
    from emmy.compiler.ir.stmt import Body, parse_body
    from tests.compiler.ir.stmt import listing

    a, b = Axis("a", 4), Axis("b", 8)
    builder = BodyBuilder(set())
    builder.insert(*parse_body("v0 = copy(x)"), Scope(enclosing=(a,)))
    builder.insert(*parse_body("v1 = copy(x)"), Scope(enclosing=(a, b)))
    builder.insert(*parse_body("v2 = copy(x)"), Scope(enclosing=(a,)))
    builder.insert(*parse_body("v3 = copy(x)"), Scope(enclosing=()))

    assert str(Body(builder.finish())) == listing(
        """
        v3 = copy(x)
        for a in 0..4:
            v2 = copy(x)
            for b in 0..8:
                v1 = copy(x)
            v0 = copy(x)
        """
    )
