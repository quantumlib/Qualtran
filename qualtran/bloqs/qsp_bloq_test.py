import numpy as np
from cirq_ft.algos.qubitization_walk_operator_test import get_walk_operator_for_1d_Ising_model

from qualtran.bloqs.qsp_bloq import QEVTCircuit, qsp_phase_factors
from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq


def test_compute_qsp_phase_factors():
    phases = qsp_phase_factors([0.5, 0.5], [0.5, -0.5])
    theta = phases['theta']
    phi = phases['phi']
    lambd = phases['lambda']
    assert (theta == [0, np.pi / 4]).all()
    assert (phi == [0, np.pi]).all()
    assert lambd == 0


def test_QEVT_circuit():
    U = get_walk_operator_for_1d_Ising_model(4, 2e-1)
    pU = QEVTCircuit(U, (0.5, 0.5), (0.5, -0.5))
    bloq = CirqGateAsBloq(pU)
