import pytest

from cirq_qubitization.bloq_algos.sorting import Comparator, BitonicSort
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

def test_comparator():
    bb = CompositeBloqBuilder()
    nbits = 4
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('b', nbits)
    anc = bb.add_register('anc', 1)
    a, b, anc = bb.add(Comparator(nbits), a=q0, b=q1, anc=anc)
    cbloq = bb.finalize(a=a, b=b, anc=anc)
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()

def test_bitonic_sort():
    nbits = 4
    k = 8
    bloq = BitonicSort(nbits, k)
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()