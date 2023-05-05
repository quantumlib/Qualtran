import pytest

from cirq_qubitization.bloq_algos.arithmetic import Add, Product, Square, SumOfSquares
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_add():
    from cirq_qubitization.bloq_algos.arithmetic import Add

    return Add(nbits=4)


def _make_square():
    from cirq_qubitization.bloq_algos.arithmetic import Square

    return Square(nbits=8)


def _make_sum_of_squares():
    from cirq_qubitization.bloq_algos.arithmetic import SumOfSquares

    return SumOfSquares(nbits=8, k=4)


def _make_product():
    from cirq_qubitization.bloq_algos.arithmetic import Product

    return Product(nbits=4, mbits=6)


def test_add():
    bb = CompositeBloqBuilder()
    nbits = 4
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('b', nbits)
    a, b = bb.add(Add(nbits), a=q0, b=q1)
    cbloq = bb.finalize(a=a, b=b)
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()


def test_square():
    bb = CompositeBloqBuilder()
    nbits = 4
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('result', 2 * nbits)
    q0, q1 = bb.add(Square(nbits), a=q0, result=q1)
    cbloq = bb.finalize(a=q0, result=q1)
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()


def test_sum_of_squares():
    bb = CompositeBloqBuilder()
    nbits = 4
    k = 3
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('result', 2 * nbits)
    q0, q1 = bb.add(SumOfSquares(nbits, 9, k), inputs=q0, result=q1)
    cbloq = bb.finalize(a=q0, result=q1)
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()


def test_product():
    bb = CompositeBloqBuilder()
    nbits = 5
    mbits = 3
    q0 = bb.add_register('a', nbits)
    q1 = bb.add_register('b', mbits)
    q2 = bb.add_register('result', 2 * max(nbits, mbits))
    q0, q1, q2 = bb.add(Product(nbits, mbits), a=q0, b=q1, result=q2)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    with pytest.raises(NotImplementedError):
        cbloq.decompose_bloq()
