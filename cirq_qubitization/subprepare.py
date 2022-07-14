"""
Implementation of all components for SUBPREPARE

UNIFORM_L on first selection register
H^{mu} on mu-sigma-register
QROM-alt-keep selection is on first selection alt-keep are on next mu and logL registers
LessThanEqualGate 
Coherent swap

Input to subprepare should be the LCU coefficients and the desired accuracy to represent
each probability (which sets mu size and keep/alt integers).

Total space will be 2 * log(L) + 2 mu + 1

The 1 ancilla is for the LessThanEqualGate followed by coherent swap
"""
from typing import List

import numpy as np


from openfermion.circuits.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from cirq_qubitization.arithmetic_gates import LessThanEqualGate
from cirq_qubitization.alt_keep_qrom import construct_alt_keep_qrom
from cirq_qubitization.prepare_uniform_superposition import PrepareUniformSuperposition
import cirq


class GENERICSUBPREPARE(cirq.Gate):

    def __init__(self, lcu_probabilities: List[float], *, probability_epsilon: float = 1.0E-5, num_controls: int = 0) -> None:
        """
        """
        self._lcu_probs = lcu_probabilities
        self._num_controls = num_controls
        self._probability_epsilon = probability_epsilon

        ## get the non-trivial gate components for SUPREPARE
        # QROM-alt-keep
        self._alt_keep_qrom = construct_alt_keep_qrom(self._lcu_probs, 
                                                self._probability_epsilon)
        self._alternates, self._keep_numers = self._alt_keep_qrom._data[0], self._alt_keep_qrom._data[1]
        self._mu = max([xx.bit_length() for xx in self._keep_numers])
        self._select_register_size = int(np.ceil(np.log2(len(self._lcu_probs))))
        # compute ancilla needed for qrom
        self._alt_keep_qrom_ancilla = self._alt_keep_qrom.num_qubits() - (2 * self._select_register_size + self._mu) - 1 # 1 is for control bit in qrom

        # uniform superposition
        self._uniform_op = PrepareUniformSuperposition(len(self._lcu_probs), num_controls=num_controls)
        self._uniform_op_num_ancilla = self._uniform_op.num_qubits() - self._select_register_size - self._uniform_op._num_controls

        # sigma_keep-less-tan
        self._lessthan_equal = LessThanEqualGate([2] * self._mu, [2] * self._mu)
        self._lessthan_equal_ancilla = 0  # Update later depending on how this is decomposed

    @property
    def selection_register(self) -> int:
        return self._select_register_size
    
    @property
    def sigma_mu_register(self) -> int:
        return self._mu
    
    @property
    def alternates_register(self) -> int:
        return self._select_register_size
    
    @property
    def keep_register(self) -> int:
        return self._mu
    
    @property
    def temp_register(self) -> int:
        return 2 * self._mu + self._select_register_size + 1

    def _num_qubits_(self) -> int:
        return None # how to compute this with a memory manager?

    def __repr__(self) -> str:
        return (
            f"cirq_qubitization.SUBPREPARE"
            f"num_controls={self._num_controls}"
            f")"
        )
    def _decompose_(self, qubits: List[cirq.Qid]) -> cirq.OP_TREE:
        # Bunch of yields
        pass


if __name__ == "__main__":
    # An example, of what is needed to simulate this
    from cirq_qubitization.generic_select_test import OneDimensionalIsingModel
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 4
    target_register_size = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_register_size = int(np.ceil(np.log(num_select_unitaries)))
    # Get paulistring terms
    # right now we only handle positive interaction term values
    target = cirq.LineQubit.range(num_sites)
    ising_inst = OneDimensionalIsingModel(num_sites, np.pi / 3,  np.pi / 7)
    pauli_string_hamiltonian = [*ising_inst.get_pauli_sum(target)]
    dense_pauli_string_hamiltonian = [
        tt.dense(target) for tt in pauli_string_hamiltonian
    ]
    qubitization_lambda = sum(
        xx.coefficient.real for xx in dense_pauli_string_hamiltonian
    )
    lcu_coeffs = (
        np.array([xx.coefficient.real for xx in dense_pauli_string_hamiltonian]) 
        / qubitization_lambda
    )
    epsilon = 1.0E-2  # precision value is kept low so we can simulate the output
    qrom = construct_alt_keep_qrom(lcu_coefficients=lcu_coeffs, probability_epsilon=epsilon)

    alternates, keep_numers = qrom._data
    mu = max([xx.bit_length() for xx in keep_numers])

    # for this test mu should be equal to 4
    assert mu == 4

    subprepare_gate = GENERICSUBPREPARE(lcu_probabilities=lcu_coeffs, probability_epsilon=epsilon, num_controls=0)
