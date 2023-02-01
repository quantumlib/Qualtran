import numpy as np

from cirq_qubitization.surface_code.ccz2t_cost_model import get_ccz2t_costs
from cirq_qubitization.surface_code.factory import MagicStateCount


def test_vs_spreadsheet():
    re = get_ccz2t_costs(
        n_magic=MagicStateCount(t_count=10**8, ccz_count=10**8),
        n_algo_qubits=100,
        error_budget=0.01,
        phys_err=1e-3,
        cycle_time_us=1,
    )

    np.testing.assert_allclose(re.failure_prob, 0.0084, rtol=1e-3)
    np.testing.assert_allclose(re.footprint, 4.00e5, rtol=1e-3)
    np.testing.assert_allclose(re.duration_hr, 7.53, rtol=1e-3)
