from emmy.compiler.ir.expression import BinaryExpr, Literal, Var
from emmy.compiler.ir.sigma import Sigma


def _deep_expression(depth: int) -> BinaryExpr:
    expression = Var("i")
    for _ in range(depth):
        expression = expression * 2 + Var("j")
    return BinaryExpr("^", expression, Literal(1, "int"))


def test_restrict_reuses_parent_canonical_entries(monkeypatch):
    target = _deep_expression(128)
    sigma = Sigma({"b": Var("b"), "a": target})
    root_walks = 0
    original = BinaryExpr.__str__

    def counted_text(expression):
        nonlocal root_walks
        if expression.op == "^":
            root_walks += 1
        return original(expression)

    monkeypatch.setattr(BinaryExpr, "__str__", counted_text)
    restricted = [sigma.restrict({"a"}) for _ in range(1_000)]

    assert root_walks == 0
    assert all(item == restricted[0] and hash(item) == hash(restricted[0]) for item in restricted)
    assert all(item._key[0][1] is sigma._key[0][1] for item in restricted)


def test_extend_reuses_unchanged_canonical_entries(monkeypatch):
    target = _deep_expression(128)
    sigma = Sigma({"a": target})
    root_walks = 0
    original = BinaryExpr.__str__

    def counted_text(expression):
        nonlocal root_walks
        if expression.op == "^":
            root_walks += 1
        return original(expression)

    monkeypatch.setattr(BinaryExpr, "__str__", counted_text)
    extended = sigma.extend("b", Var("j"))

    assert root_walks == 0
    assert extended == Sigma({"a": target, "b": Var("j")})
    assert hash(extended) == hash(Sigma({"a": target, "b": Var("j")}))
    assert extended._key[0][1] is sigma._key[0][1]
