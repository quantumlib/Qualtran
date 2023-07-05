import numpy as np

from qualtran.bloq_algos.basic_gates import PlusEffect, PlusState
from qualtran.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_plus_state():
    from qualtran.bloq_algos.basic_gates import PlusState

    return PlusState()


def test_plus_state():
    bloq = PlusState()
    vector = bloq.tensor_contract()
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def _make_plus_effect():
    from qualtran.bloq_algos.basic_gates import PlusEffect

    return PlusEffect()


def test_plus_effect():
    bloq = PlusEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def test_plus_state_effect():
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(PlusState())
    bb.add(PlusEffect(), q=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)
