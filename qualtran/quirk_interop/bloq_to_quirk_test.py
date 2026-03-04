import pytest

from qualtran import BloqBuilder, QAny, QUInt
from qualtran.bloqs.bookkeeping import Allocate, Join, Split
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.bloqs.arithmetic import Add, Negate
from qualtran.quirk_interop.bloq_to_quirk import (
    SparseLineManager,
    bloq_to_quirk,
    composite_bloq_to_quirk,
)


def _build_split_join_split_cbloq(n):
    bb = BloqBuilder()
    q = bb.add(Allocate(QAny(n)))
    qs = bb.add(Split(QAny(n)), reg=q)
    q_joined = bb.add(Join(QAny(n)), reg=qs)
    qs_again = bb.add(Split(QAny(n)), reg=q_joined)
    out = bb.add(Join(QAny(n)), reg=qs_again)
    return bb.finalize(out=out)


@pytest.mark.parametrize("n", range(3, 6))
def test_sparse_line_manager_builds_dual_maps(n):
    cbloq = _build_split_join_split_cbloq(n)
    manager = SparseLineManager(cbloq)

    assert manager._join_to_split_id
    assert manager._split_to_join_id


@pytest.mark.parametrize("n", range(3, 6))
def test_composite_bloq_to_quirk_url_shape(n):
    cbloq = MultiTargetCNOT(n).decompose_bloq().flatten()
    url = composite_bloq_to_quirk(cbloq)

    assert url.startswith('https://algassert.com/quirk#circuit={"cols":[')
    assert url.endswith(']}')


def test_bloq_to_quirk():
    url_add = bloq_to_quirk(Add(QUInt(5)))
    assert url_add.startswith('https://algassert.com/quirk#circuit={"cols":[')
    assert url_add.endswith(']}')
    url_mtcnot = bloq_to_quirk(MultiTargetCNOT(3))
    assert (
        url_mtcnot
        == 'https://algassert.com/quirk#circuit={"cols":[[1,"•",1,"X"],[1,"•","X",1],["•","X",1,1],[1,"•","X",1],[1,"•",1,"X"]]}'
    )


def test_negate_to_quirk():
    url = bloq_to_quirk(Negate(QUInt(2)))
    assert (
        url
        == 'https://algassert.com/quirk#circuit={"cols":[["X",1,1,1,1],[1,"X",1,1,1],[1,1,1,"X",1],[1,"•",1,"•","X"],["X",1,1,1,"•"],[1,"•",1,"•","X"],["X",1,"•",1,1],[1,"X",1,"•",1],[1,1,1,"X",1]]}'
    )


def test_bloq_to_quirk_on_atomic():
    url = bloq_to_quirk(Toffoli())
    assert url == 'https://algassert.com/quirk#circuit={"cols":[["•","•","X"]]}'
