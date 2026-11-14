"""``FragmentApply``'s global-memory operand kind — the fragment-tier bias: what it reads, under renaming. How each
operand kind is computed per element is the PTX compiler's (``crates/emmy-ptx/tests/fragment.rs``)."""

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel.ir import FRAG, FRAG_COL, FRAG_ROW, GMEM, FragmentApply, fragment_args


def test_a_gmem_operand_is_a_read_of_its_buffer() -> None:
    bias = FragmentApply(
        out="s",
        op=ElementwiseImpl("add"),
        args=fragment_args(("s", ("mask", (Literal(0, "int"), Var(FRAG_ROW), Var(FRAG_COL)))), (FRAG, GMEM)),
        in_place=True,
        row_base=Var("q"),
        col_base=Var("kv"),
    )
    assert bias.external_reads() == ("mask",)
    assert bias.rename_buffers({"mask": "m0"}).external_reads() == ("m0",)
