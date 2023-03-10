import numpy as np
import pytest

from cirq_qubitization.bloq_algos.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.quimb_sim import bloq_to_dense, cbloq_to_dense


def test_zero_state():
    bloq = ZeroState()
    assert str(bloq) == 'ZeroState(n=1)'
    assert not bloq.bit
    vector = bloq_to_dense(bloq)
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def test_multiq_zero_state():
    # Verifying the attrs trickery that I can plumb through *some*
    # of the attributes but pre-specify others.
    with pytest.raises(NotImplementedError):
        _ = ZeroState(n=10)


def test_one_state():
    bloq = OneState()
    assert bloq.bit
    assert bloq.state
    vector = bloq_to_dense(bloq)
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)


def test_zero_effect():
    vector = bloq_to_dense(ZeroEffect())

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def test_one_effect():
    vector = bloq_to_dense(OneEffect())

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)


@pytest.mark.parametrize('bit', [False, True])
def test_zero_state_effect(bit):
    bb = CompositeBloqBuilder()

    if bit:
        state = OneState()
        eff = OneEffect()
    else:
        state = ZeroState()
        eff = ZeroEffect()

    (q0,) = bb.add(state)
    bb.add(eff, q=q0)
    cbloq = bb.finalize()
    val = cbloq_to_dense(cbloq)

    should_be = 1
    np.testing.assert_allclose(should_be, val)
