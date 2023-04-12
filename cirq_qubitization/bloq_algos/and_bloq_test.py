import itertools

import numpy as np
import pytest

import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.bloq_algos.and_bloq import And, MultiAnd
from cirq_qubitization.bloq_algos.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_and():
    from cirq_qubitization.bloq_algos.and_bloq import And

    return And()


def _make_multi_and():
    from cirq_qubitization.bloq_algos.and_bloq import MultiAnd

    return MultiAnd(cvs=(1, 1, 1, 1))


def _iter_and_truth_table(cv1: int, cv2: int):
    # Iterate over And bra/ketted by all possible inputs
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
        yield cbloq, a, b


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_truth_table(cv1, cv2):
    for cbloq, a, b in _iter_and_truth_table(cv1, cv2):
        vec = cbloq.tensor_contract()
        if (a == cv1) and (b == cv2):
            np.testing.assert_allclose([0, 1], vec)
        else:
            np.testing.assert_allclose([1, 0], vec)


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_truth_table_classical(cv1, cv2):
    for cbloq, a, b in _iter_and_truth_table(cv1, cv2):
        out = cbloq.apply_classical()
        if (a == cv1) and (b == cv2):
            assert out['res'] == 1
        else:
            assert out['res'] == 0


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


def test_multi_truth_table():
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]

    n = 4
    rs = np.random.RandomState(52)
    all_cvs = rs.choice([0, 1], size=(2, n))
    # ctrl_strings = np.array(list(itertools.product([0,1], repeat=n)))
    ctrl_strings = rs.choice([0, 1], size=(10, n))

    for cvs in all_cvs:
        for ctrl_string in ctrl_strings:
            bb = CompositeBloqBuilder()
            ctrl_qs = [bb.add(state[c])[0] for c in ctrl_string]

            ctrl_qs, junk, res = bb.add_from(MultiAnd(cvs), ctrl=ctrl_qs)

            for c, q in zip(ctrl_string, ctrl_qs):
                bb.add(eff[c], q=q)

            cbloq = bb.finalize(junk=junk, res=res)

            # Tensor simulation
            vec = cbloq.tensor_contract()
            should_be = np.all(ctrl_string == cvs)
            *junk_is, res_i = np.where(vec.reshape((2,) * (n - 1)))
            assert res_i == should_be, ctrl_string

            # Classical simulation
            classical = cbloq.apply_classical()
            assert classical['res'] == should_be


def test_multiand_consitent_apply_classical():
    rs = np.random.RandomState(52)
    n = 5
    all_cvs = rs.choice([0, 1], size=(2, n))
    # ctrl_strings = np.array(list(itertools.product([0,1], repeat=n)))
    ctrl_strings = rs.choice([0, 1], size=(10, n))

    for cvs, ctrl_string in itertools.product(all_cvs, ctrl_strings):
        bloq = MultiAnd(cvs=cvs)
        cbloq = bloq.decompose_bloq()

        bloq_classical = bloq.apply_classical(ctrl=ctrl_string)
        cbloq_classical = cbloq.apply_classical(ctrl=ctrl_string)

        assert sorted(bloq_classical.keys()) == sorted(cbloq_classical.keys())
        for k in bloq_classical.keys():
            np.testing.assert_array_equal(bloq_classical[k], cbloq_classical[k])


def test_notebook():
    cq_testing.execute_notebook('and_bloq')
