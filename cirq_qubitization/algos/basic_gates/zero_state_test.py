import numpy as np

from cirq_qubitization.algos.basic_gates import ZeroEffect, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.quimb_sim import bloq_to_dense, cbloq_to_dense


def _make_zero_state():
    from cirq_qubitization.algos.basic_gates import ZeroState

    return ZeroState()


def test_zero_state():
    bloq = ZeroState()
    vector = bloq_to_dense(bloq)
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def _make_zero_effect():
    from cirq_qubitization.algos.basic_gates import ZeroEffect

    return ZeroEffect()


def test_zero_effect():
    bloq = ZeroEffect()
    vector = bloq_to_dense(bloq)

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track of your own shapes or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)


def test_zero_state_effect():
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(ZeroState())
    bb.add(ZeroEffect(), q=q0)
    cbloq = bb.finalize()
    val = cbloq_to_dense(cbloq)

    should_be = 1
    np.testing.assert_allclose(should_be, val)
