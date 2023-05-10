import pytest

from cirq_qubitization.bloq_algos.sorting import BitonicSort, Comparator
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_comparator():
    from cirq_qubitization.bloq_algos.sorting import Comparator

    return Comparator(bitsize=4)


def _make_bitonic_sort():
    from cirq_qubitization.bloq_algos.sorting import BitonicSort

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
