import numpy as np
import pytest

from cirq_qubitization.algos.basic_gates import ONE, ONE_EFFECT, ZERO, ZERO_EFFECT, ZVector
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.quimb_sim import bloq_to_dense, cbloq_to_dense


def test_zero_state():
    bloq = ZERO
    assert bloq == ZVector(bit=False, state=True)
    vector = bloq_to_dense(bloq)
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def test_one_state():
    bloq = ONE
    assert bloq == ZVector(bit=True, state=True)
    vector = bloq_to_dense(bloq)
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)


def test_zero_effect():
    vector = bloq_to_dense(ZERO_EFFECT)

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def test_one_effect():
    vector = bloq_to_dense(ONE_EFFECT)

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)


@pytest.mark.parametrize('bit', [False, True])
def test_zero_state_effect(bit):
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(ZVector(bit, state=True))
    bb.add(ZVector(bit, state=False), q=q0)
    cbloq = bb.finalize()
    val = cbloq_to_dense(cbloq)

    should_be = 1
    np.testing.assert_allclose(should_be, val)
