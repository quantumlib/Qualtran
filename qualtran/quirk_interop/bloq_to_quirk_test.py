import pytest

from qualtran import BloqBuilder, QAny
from qualtran.bloqs.basic_gates import Toffoli, XGate
from qualtran.bloqs.bookkeeping import Allocate, Join, Split
from qualtran.quirk_interop.bloq_to_quirk import (
    bloq_to_quirk,
    composite_bloq_to_quirk,
    SparseLineManager,
)


def _build_split_join_split_cbloq(n):
    bb = BloqBuilder()
    q1 = bb.add(Allocate(QAny(n)))
    q2 = bb.add(Allocate(QAny(n)))
    qs1 = bb.add(Split(QAny(n)), reg=q1)
    qs2 = bb.add(Split(QAny(n)), reg=q2)
    for i in range(n):
        qs1[i] = bb.add(XGate(), q=qs1[i])
    q1 = bb.add(Join(QAny(n)), reg=qs1)
    q2 = bb.add(Join(QAny(n)), reg=qs2)
    qs1 = bb.add(Split(QAny(n)), reg=q1)
    qs2 = bb.add(Split(QAny(n)), reg=q2)
    for i in range(n):
        qs2[i] = bb.add(XGate(), q=qs2[i])
    q2 = bb.add(Join(QAny(n)), reg=qs2)
    q1 = bb.add(Join(QAny(n)), reg=qs1)
    return bb.finalize(q1=q1, q2=q2)


@pytest.mark.parametrize("n", range(3, 6))
def test_sparse_line_manager_builds_dual_maps(n):
    cbloq = _build_split_join_split_cbloq(n)
    manager = SparseLineManager(cbloq)

    assert manager._join_to_split_id
    assert manager._split_to_join_id


@pytest.mark.parametrize("n", range(3, 6))
def test_composite_bloq_to_quirk_url_shape(n):
    cbloq = _build_split_join_split_cbloq(n)
    url = composite_bloq_to_quirk(cbloq)

    assert url.startswith('https://algassert.com/quirk#circuit={"cols":[')
    assert url.endswith(']}')


def test_composite_bloq_to_quirk():
    cbloq1 = _build_split_join_split_cbloq(1)
    url1 = composite_bloq_to_quirk(cbloq1)
    assert url1 == 'https://algassert.com/quirk#circuit={"cols":[["X",1],[1,"X"]]}'
    cbloq2 = _build_split_join_split_cbloq(2)
    url2 = composite_bloq_to_quirk(cbloq2)
    assert (
        url2
        == 'https://algassert.com/quirk#circuit={"cols":[["X",1,1,1],[1,"X",1,1],[1,1,"X",1],[1,1,1,"X"]]}'
    )


def test_bloq_to_quirk_on_atomic():
    url = bloq_to_quirk(Toffoli())
    assert url == 'https://algassert.com/quirk#circuit={"cols":[["•","•","X"]]}'
