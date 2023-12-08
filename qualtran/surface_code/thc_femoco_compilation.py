from functools import partial

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.ccz2t_cost_model import (
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)

ERR_BUDGET = 0.01
N_TOFFOLI = 6665400000  # Pag. 26
N_QUBITS = 696  # Fig. 10
PHYS_ERR = 1e-3
N_FACTORIES = 4

n_magic = AlgorithmSummary(toffoli_gates=N_TOFFOLI)

best_cost, best_factory, best_data_block = get_ccz2t_costs_from_grid_search(
    n_magic=n_magic,
    n_algo_qubits=N_QUBITS,
    error_budget=ERR_BUDGET,
    phys_err=PHYS_ERR,
    factory_iter=partial(iter_ccz2t_factories, n_factories=N_FACTORIES),
    cost_function=(lambda pc: pc.duration_hr),
)

print(best_cost)
print(best_factory)
print(best_data_block)

distillation_error = best_factory.distillation_error(n_magic, PHYS_ERR)
data_error = best_data_block.data_error(
    n_algo_qubits=N_QUBITS, n_cycles=best_factory.n_cycles(n_magic), phys_err=PHYS_ERR
)

print(f"distillation error: {distillation_error:.3%}")  # ref: 0.1% per 1e10 Toffolis
print(f"data error: {data_error:.3%}")
print(f"wall time: {best_cost.duration_hr/24} days")  # ref: 3 days
print(
    f"footprint: {best_cost.footprint*1e-6:.2f} million qubits"
)  # ref: 4 million qubits
