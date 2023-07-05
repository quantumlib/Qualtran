import itertools

import cirq
import numpy as np
import pytest

from qualtran import Signature
from qualtran.bloq_algos.basic_gates import CNOT, PlusState, ZeroState
from qualtran.quantum_graph.composite_bloq import BloqBuilder
from qualtran.quantum_graph.musical_score import get_musical_score_data


def _make_CNOT():
    from qualtran.bloq_algos.basic_gates import CNOT

    return CNOT()


def test_cnot():
    bloq = CNOT()
    matrix = bloq.tensor_contract()
    # fmt: off
    should_be = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])
    # fmt: on
    np.testing.assert_allclose(should_be, matrix)


def test_cnot_cbloq():
    bb, soqs = BloqBuilder.from_registers(Signature.build(c=1, t=1))
    c, t = bb.add(CNOT(), ctrl=soqs['c'], target=soqs['t'])
    cbloq = bb.finalize(c=c, t=t)
    matrix = cbloq.tensor_contract()

    c_qs = cirq.LineQubit.range(2)
    c_circ = cirq.Circuit(cirq.CNOT(c_qs[0], c_qs[1]))
    c_matrix = c_circ.unitary(qubit_order=c_qs)

    np.testing.assert_allclose(c_matrix, matrix)


def test_bell_state():
    bb = BloqBuilder()

    (q0,) = bb.add(PlusState())
    (q1,) = bb.add(ZeroState())

    q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)

    cbloq = bb.finalize(q0=q0, q1=q1)
    matrix = cbloq.tensor_contract()

    should_be = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, matrix)


def test_classical_truth_table():
    truth_table = []
    for c, t in itertools.product([0, 1], repeat=2):
        out_c, out_t = CNOT().call_classically(ctrl=c, target=t)
        truth_table.append(((c, t), (out_c, out_t)))

    assert truth_table == [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))]

    with pytest.raises(ValueError):
        CNOT().call_classically(ctrl=2, target=0)


def test_cnot_musical_score():
    cbloq = CNOT().as_composite_bloq()
    msd = get_musical_score_data(cbloq)
    # soq[0] and [1] are the dangling symbols
    assert msd.soqs[2].json_dict()['symb_cls'] == 'Circle'
    assert msd.soqs[3].json_dict()['symb_cls'] == 'ModPlus'
