# !!!! Do not modify imports !!!!
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from typing import *
import cirq
import numpy as np
import cirq_qubitization
import cirq_qubitization.testing as cq_testing

# pylint: enable=unused-import,wildcard-import,unused-wildcard-import
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

    target_bitsize = 4
    us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
    us = [cirq.DensePauliString(u) for u in us]
    selection_bitsize = int(np.ceil(np.log2(len(us))))
    return GenericSelect(selection_bitsize, target_bitsize, select_unitaries=us)


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
