import pytest

from cirq_qubitization.bloq_algos.sorting import Comparator, BitonicSort
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_comparator():
    from cirq_qubitization.bloq_algos.sorting import Comparator

    return Comparator(nbits=4)


def _make_bitonic_sort():
    from cirq_qubitization.bloq_algos.sorting import BitonicSort

    return BitonicSort(nbits=8, k=8)


def test_comparator():
    bb = CompositeBloqBuilder()
    nbits = 4
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('b', nbits)
    anc = bb.add_register('anc', 1)
    a, b, anc = bb.add(Comparator(nbits), a=q0, b=q1, anc=anc)
    cbloq = bb.finalize(a=a, b=b, anc=anc)
    print(cbloq.t_complexity())
    assert cbloq.t_complexity().t == 88
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()


def test_bitonic_sort():
    nbits = 4
    k = 8
    bloq = BitonicSort(nbits, k)
    print(bloq.t_complexity())
    # k * log(k)^2 * Comparator(nbits)
    assert bloq.t_complexity().t == 8 * 9 * 88
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()
