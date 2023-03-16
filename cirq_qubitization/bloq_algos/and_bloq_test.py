import itertools

import numpy as np
import pytest

import cirq_qubitization.testing as cq_testing
from cirq_qubitization.bloq_algos.and_bloq import And
from cirq_qubitization.bloq_algos.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_and():
    from cirq_qubitization.bloq_algos.and_bloq import And

    return And()


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_truth_table(cv1, cv2):
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]

    for a, b in itertools.product([0, 1], repeat=2):
        bb = CompositeBloqBuilder()
        (q_a,) = bb.add(state[a])
        (q_b,) = bb.add(state[b])
        (q_a, q_b), res = bb.add(And(cv1, cv2), ctrl=[q_a, q_b])
        bb.add(eff[a], q=q_a)
        bb.add(eff[b], q=q_b)
        cbloq = bb.finalize(res=res)

        vec = cbloq.tensor_contract()
        if (a == cv1) and (b == cv2):
            np.testing.assert_allclose([0, 1], vec)
        else:
            np.testing.assert_allclose([1, 0], vec)


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_bad_adjoint(cv1, cv2):
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]
    and_ = And(cv1, cv2, adjoint=True)

    for a, b in itertools.product([0, 1], repeat=2):
        bb = CompositeBloqBuilder()
        (q_a,) = bb.add(state[a])
        (q_b,) = bb.add(state[b])
        if (a == cv1) and (b == cv2):
            (res,) = bb.add(ZeroState())
        else:
            (res,) = bb.add(OneState())

        ((q_a, q_b),) = bb.add(and_, ctrl=[q_a, q_b], target=res)
        bb.add(eff[a], q=q_a)
        bb.add(eff[b], q=q_b)
        cbloq = bb.finalize()

        val = cbloq.tensor_contract()
        assert np.abs(val) < 1e-8


def test_inverse():
    bb = CompositeBloqBuilder()
    q0 = bb.add_register('q0', 1)
    q1 = bb.add_register('q1', 1)
    qs, trg = bb.add(And(), ctrl=[q0, q1])
    (qs,) = bb.add(And(adjoint=True), ctrl=qs, target=trg)
    cbloq = bb.finalize(q0=qs[0], q1=qs[1])

    mat = cbloq.tensor_contract()
    np.testing.assert_allclose(np.eye(4), mat)


def test_notebook():
    cq_testing.execute_notebook('bloq_algos/and_bloq')
