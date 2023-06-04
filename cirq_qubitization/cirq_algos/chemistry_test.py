import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.chemistry import SelectChem, SubPrepareChem
from cirq_qubitization.cirq_infra.gate_with_registers import SelectionRegisters
from cirq_qubitization.jupyter_tools import execute_notebook


# @pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity():
    num_orb = 4
    select = SelectChem(num_spin_orb=num_orb)
    print(select.selection_registers['q'])
    q_selection_regs = SelectionRegisters.build(
        beta=(1, 2), q=(select.selection_registers['q'].bitsize, select.num_spin_orb)
    )
    target = select.target_registers
    select = cq_testing.GateHelper(SelectChem(num_spin_orb=num_orb, control_val=1))


def test_sub_prepare():
    num_orb = 4
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * num_orb).reshape((4, num_orb))
    # not meant to be meaningful.
    lambda_H = np.sum(np.abs([Us, Ts, Vs]))
    sp = SubPrepareChem.build(num_spin_orb=2 * num_orb, T=Ts, U=Us, V=Vs, Vx=Vxs, lambda_H=lambda_H)
    g = cq_testing.GateHelper(sp)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    data_size = 4 * num_orb
    qrom = cq.MultiIndexedQROM.build([sp.altU])
    assert cq_testing.t_complexity(qrom).t == 4 * data_size - 8
