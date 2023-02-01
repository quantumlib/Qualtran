import numpy as np

from cirq_qubitization.surface_code.formulae import code_distance_from_budget, error_at


def test_invert_error_at():
    phys_err = 1e-3
    budgets = np.logspace(-1, -18)
    for budget in budgets:
        d = code_distance_from_budget(phys_err=phys_err, budget=budget)
        assert d % 2 == 1
        assert error_at(phys_err=phys_err, d=d) <= budget
        if d > 3:
            assert error_at(phys_err=phys_err, d=d - 2) > budget
