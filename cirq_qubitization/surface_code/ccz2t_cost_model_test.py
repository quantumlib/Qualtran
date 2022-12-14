import numpy as np

from cirq_qubitization.surface_code.ccz2t_cost_model import CCZ2TCostModel


def test_vs_spreadsheet():
    re = CCZ2TCostModel(
        t_count=10**8,
        toffoli_count=10**8,
        n_alg_qubits=100,
        error_budget=0.01,
        physical_error_rate=1e-3,
        cycle_time_us=1,
    )

    np.testing.assert_allclose(re.failure_prob, 0.0084, rtol=1e-3)
    np.testing.assert_allclose(re.n_phys_qubits, 4.00e5, rtol=1e-3)
    np.testing.assert_allclose(re.duration_hr, 7.53, rtol=1e-3)


def test_invert_error_at():
    re = CCZ2TCostModel(
        t_count=10**8,
        toffoli_count=10**8,
        n_alg_qubits=100,
        error_budget=0.01,
        physical_error_rate=1e-3,
        cycle_time_us=1,
    )

    budgets = np.logspace(-1, -18)
    for budget in budgets:
        d = re._code_distance_from_budget(budget=budget)
        assert re.error_at(d=d) <= budget
