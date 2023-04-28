from cirq_qubitization.jupyter_tools import show_bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.bloq_algos.arithmetic_bloqs import Add


def _make_add():
    from cirq_qubitization.bloq_algos.arithmetic_bloqs import Add

    return Add(nbits=4)


def _make_square():
    from cirq_qubitization.bloq_algos.arithmetic_bloqs import Square

    return Square(nbits=8)


def _make_sum_of_squares():
    from cirq_qubitization.bloq_algos.arithmetic_bloqs import SumOfSquares

    return SumOfSquares(nbits=8, k=4)


def _make_product():
    from cirq_qubitization.bloq_algos.arithmetic_bloqs import Product

    return Product(nbits=4, mbits=6)


# def test_add():
#     bb = CompositeBloqBuilder()
#     # nbits = 4
#     q0 = bb.add_register('a', nbits)
#     q1 = bb.add_register('b', nbits)
#     qs, trg = bb.add(Add(4))
#     cbloq = bb.finalize(input=qs[0], output=qs[1])
