import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.chemistry import SelectChem, SubPrepareChem
from cirq_qubitization.cirq_infra.gate_with_registers import SelectionRegisters
from cirq_qubitization.jupyter_tools import execute_notebook


def test_select_t_complexity():
    N = 10
    select = SelectChem(num_spin_orb=N, control_val=1)
    cost = cq.t_complexity(select)
    assert cost.t == 168
    assert cost.rotations == 0


def test_sub_prepare():
    num_orb = 4
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * num_orb).reshape((4, num_orb))
    # not meant to be meaningful.
    lambda_H = np.sum(np.abs([Us, Ts, Vs]))
    sp = SubPrepareChem.build_from_coefficients(
        num_spin_orb=2 * num_orb, T=Ts, U=Us, V=Vs, Vx=Vxs, lambda_H=lambda_H
    )
    g = cq_testing.GateHelper(sp)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    data_size = 4 * num_orb
    # Because we iterate over U and V completely we have an increase T
    # complexity of 8N-4 for QROM as opposed to 3N - 4, if we only could pick
    # 00, 01, and 10 selection combinations.
    qrom = cq.QROM.build([sp.altU])
    assert cq_testing.t_complexity(qrom).t == 4 * data_size - 8
