import pytest

from qualtran.bloq_algos.sorting import BitonicSort, Comparator
from qualtran.jupyter_tools import execute_notebook


def _make_comparator():
    from qualtran.bloq_algos.sorting import Comparator

    return Comparator(bitsize=4)


def _make_bitonic_sort():
    from qualtran.bloq_algos.sorting import BitonicSort

    return BitonicSort(bitsize=8, k=8)


def test_comparator():
    bloq = Comparator(4)
    assert bloq.t_complexity().t == 88
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


def test_bitonic_sort():
    bitsize = 4
    k = 8
    bloq = BitonicSort(bitsize, k)
    assert bloq.t_complexity().t == 8 * 9 * 88
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


def test_notebook():
    execute_notebook('sorting')
