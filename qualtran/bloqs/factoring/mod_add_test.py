from qualtran.bloqs.factoring.mod_add import CtrlModAddK, CtrlScaleModAdd
from qualtran.resource_counting import SympySymbolAllocator


def test_ctrl_scale_mod_add():
    bloq = CtrlScaleModAdd(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'y += x*123 % 221'

    counts = bloq.bloq_counts(SympySymbolAllocator())
    assert counts[0][0] == 8


def test_ctrl_mod_add_k():
    bloq = CtrlModAddK(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'x += 123 % 221'

    counts = bloq.bloq_counts(SympySymbolAllocator())
    assert counts[0][0] == 5
