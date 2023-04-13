import numpy as np
import pytest

from cirq_qubitization.bloq_algos.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def test_zero_state():
    bloq = ZeroState()
    assert str(bloq) == 'ZeroState(n=1)'
    assert not bloq.bit
    vector = bloq.tensor_contract()
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    classical = bloq.apply_classical()
    assert classical['q'] == 0


def test_multiq_zero_state():
    # Verifying the attrs trickery that I can plumb through *some*
    # of the attributes but pre-specify others.
    with pytest.raises(NotImplementedError):
        _ = ZeroState(n=10)


def test_one_state():
    bloq = OneState()
    assert bloq.bit
    assert bloq.state
    vector = bloq.tensor_contract()
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    classical = bloq.apply_classical()
    assert classical['q'] == 1


def test_zero_effect():
    bloq = ZeroEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    classical = bloq.apply_classical(q=0)
    assert classical == {}

    with pytest.raises(AssertionError):
        bloq.apply_classical(q=1)

    with pytest.raises(ValueError, match=r'.*q should be an integer, not \[0\, 0\, 0\]'):
        # TODO: error checking by default
        bloq.as_composite_bloq().apply_classical(q=[0, 0, 0])


def test_one_effect():
    bloq = OneEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    classical = bloq.apply_classical(q=1)
    assert classical == {}

    with pytest.raises(AssertionError):
        bloq.apply_classical(q=0)


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
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)

    res = cbloq.apply_classical()
    assert res == {}
