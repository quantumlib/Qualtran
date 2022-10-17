# !!!! Do not modify imports !!!!
import cirq
import numpy as np
import cirq_qubitization
import cirq_qubitization.testing as cq_testing
from typing import *


# !!!! Do not modify imports !!!!

# This module contains functions that generate demo gate objects. Both the code and
# the objects they return are used to render some auto-generated cells in our jupyter
# notebooks.
#
# The above imports are the imports guaranteed to be in a jupyter notebook. For bespoke
# imports, use a local import in your function. This will be rendered into the notebook as well.
#
# These functions must have a globally unique name, take no arguments, and finish with
# a `return` statement from which we can extract an expression that will be rendered into
# a notebook template.


def _make_ApplyGateToLthQubit():
    from cirq_qubitization.apply_gate_to_lth_target import ApplyGateToLthQubit

    def _z_to_odd(n: int):
        if n & 1:
            return cirq.Z
        return cirq.I

    apply_z_to_odd = ApplyGateToLthQubit(
        selection_bitsize=3, target_bitsize=4, nth_gate=_z_to_odd, control_bitsize=2
    )

    return apply_z_to_odd


def _make_QROM():
    from cirq_qubitization import QROM

    return QROM([1, 2, 3, 4, 5])


def _make_MultiTargetCSwap():
    from cirq_qubitization.swap_network import MultiTargetCSwap

    return MultiTargetCSwap(3)


def _make_MultiTargetCSwapApprox():
    from cirq_qubitization.swap_network import MultiTargetCSwapApprox

    return MultiTargetCSwapApprox(2)


def _make_GenericSelect():
    from cirq_qubitization.generic_select import GenericSelect
    from cirq_qubitization.generic_select_test import get_1d_ising_hamiltonian

    num_sites = 4
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites

    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus, 2 * num_sites Pauli ops
    selection_bitsize = int(np.ceil(np.log(num_select_unitaries)))

    target = cirq.LineQubit.range(target_bitsize)  # placeholder
    ham = get_1d_ising_hamiltonian(target, 1, 1)
    dense_ham = [tt.dense(target) for tt in ham]
    return GenericSelect(selection_bitsize, target_bitsize, select_unitaries=dense_ham)


def _make_GenericSubPrepare():
    from cirq_qubitization.generic_subprepare import GenericSubPrepare

    def get_1d_ising_hamiltonian(
        qubits: Sequence[cirq.Qid], j_zz_strength: float = 1.0, gamma_x_strength: float = -1
    ) -> cirq.PauliSum:
        n_sites = len(qubits)
        terms = [
            cirq.PauliString(
                {qubits[k]: cirq.Z, qubits[(k + 1) % n_sites]: cirq.Z}, coefficient=j_zz_strength
            )
            for k in range(n_sites)
        ]
        terms.extend([cirq.PauliString({q: cirq.X}, coefficient=gamma_x_strength) for q in qubits])
        return cirq.PauliSum.from_pauli_strings(terms)

    spins = cirq.LineQubit.range(3)
    ham = get_1d_ising_hamiltonian(spins, np.pi / 3, np.pi / 7)
    coeffs = np.array([term.coefficient.real for term in ham])

    lcu_coeffs = coeffs / np.sum(coeffs)

    return GenericSubPrepare(lcu_coeffs, probability_epsilon=1e-2)
