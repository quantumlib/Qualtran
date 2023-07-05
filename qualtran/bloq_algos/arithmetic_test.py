from qualtran import BloqBuilder, FancyRegister
from qualtran.bloq_algos.arithmetic import Add, GreaterThan, Product, Square, SumOfSquares
from qualtran.jupyter_tools import execute_notebook


def _make_add():
    from qualtran.bloq_algos.arithmetic import Add

    return Add(bitsize=4)


def _make_square():
    from qualtran.bloq_algos.arithmetic import Square

    return Square(bitsize=8)


def _make_sum_of_squares():
    from qualtran.bloq_algos.arithmetic import SumOfSquares

    return SumOfSquares(bitsize=8, k=4)


def _make_product():
    from qualtran.bloq_algos.arithmetic import Product

    return Product(a_bitsize=4, b_bitsize=6)


def _make_greater_than():
    from qualtran.bloq_algos.arithmetic import GreaterThan

    return GreaterThan(bitsize=4)


def test_add():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    a, b = bb.add(Add(bitsize), a=q0, b=q1)
    cbloq = bb.finalize(a=a, b=b)


def test_square():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('result', 2 * bitsize)
    q0, q1 = bb.add(Square(bitsize), a=q0, result=q1)
    cbloq = bb.finalize(a=q0, result=q1)


def test_sum_of_squares():
    bb = BloqBuilder()
    bitsize = 4
    k = 3
    inp = bb.add_register(FancyRegister("input", bitsize=bitsize, wireshape=(k,)))
    out = bb.add_register(FancyRegister("result", bitsize=2 * bitsize + 1))
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp, result=out)
    cbloq = bb.finalize(input=inp, result=out)


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q2 = bb.add_register('result', 2 * max(bitsize, mbits))
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1, result=q2)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)


def test_greater_than():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    anc = bb.add_register('result', 1)
    q0, q1, anc = bb.add(GreaterThan(bitsize), a=q0, b=q1, result=anc)
    cbloq = bb.finalize(a=q0, b=q1, result=anc)


def test_notebook():
    execute_notebook('arithmetic')
