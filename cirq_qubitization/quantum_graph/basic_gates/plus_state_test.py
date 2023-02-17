import numpy as np

from cirq_qubitization.quantum_graph.basic_gates import PlusEffect, PlusState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.quimb_sim import bloq_to_dense, cbloq_to_dense, cbloq_to_quimb


def test_plus_state():
    bloq = PlusState()
    vector = bloq_to_dense(bloq)
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def test_plus_effect():
    bloq = PlusEffect()
    vector = bloq_to_dense(bloq)

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track of your own shapes or use compositebloq.
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def test_plus_state_effect():
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(PlusState())
    bb.add(PlusEffect(), q=q0)
    cbloq = bb.fancy_finalize()
    val = cbloq_to_dense(cbloq)

    should_be = 1
    np.testing.assert_allclose(should_be, val)
