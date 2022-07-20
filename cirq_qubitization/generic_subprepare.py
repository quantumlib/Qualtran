"""
Implementation of all components for SUBPREPARE

UNIFORM_L on first selection register
H^{mu} on mu-sigma-register
QROM-alt-keep selection is on first selection alt-keep are on next mu and logL registers
LessThanEqualGate 
Coherent swap

Input to subprepare should be the LCU coefficients and the desired accuracy to represent
each probability (which sets mu size and keep/alt integers).

Total space will be (2 * log(L) + 2 mu + 1) work qubits + log(L) ancillas for QROM.

The 1 ancilla in work qubits is for the `LessThanEqualGate` followed by coherent swap.
"""
from typing import List, Sequence

from openfermion.circuits.lcu_util import (
    preprocess_lcu_coefficients_for_reversible_sampling,
)

from cirq_qubitization.arithmetic_gates import LessThanEqualGate
from cirq_qubitization.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq_qubitization.qrom import QROM
import cirq


class GenericSubPrepare(cirq.Gate):
    """Implements generic sub-prepare defined in Fig 11 of https://arxiv.org/abs/1805.03662"""

    def __init__(
        self,
        lcu_probabilities: List[float],
        *,
        probability_epsilon: float = 1.0e-5,
    ) -> None:
        self._lcu_probs = lcu_probabilities
        self._probability_epsilon = probability_epsilon
        self._select_register_size = (len(self._lcu_probs) - 1).bit_length()
        (
            self._alt,
            self._keep,
            self._mu,
        ) = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=lcu_probabilities, epsilon=probability_epsilon
        )

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
        return (
            self.sigma_mu_register + self.alternates_register + self.keep_register + 1
        )

    @property
    def ancilla_register(self) -> int:
        return self._select_register_size

    def _num_qubits_(self) -> int:
        return self._select_register_size + self.temp_register + self.ancilla_register

    def __repr__(self) -> str:
        return (
            f"cirq_qubitization.GenericSubPrepare("
            f"lcu_probabilities={self._lcu_probs},"
            f"probability_epsilon={self._probability_epsilon}"
            f")"
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> cirq.OP_TREE:
        selection = qubits[: self.selection_register]
        temp = qubits[
            self.selection_register : self.selection_register + self.temp_register
        ]
        sigma_mu, alt, keep, less_than_equal = (
            temp[: self._mu],
            temp[self._mu : self._mu + self._select_register_size],
            temp[-(self._mu + 1) : -1],
            temp[-1],
        )
        ancilla = qubits[-self.selection_register :]
        yield PrepareUniformSuperposition(len(self._lcu_probs)).on(
            *selection, ancilla[0]
        )
        qrom = QROM(self._alt, self._keep, target_registers=[len(alt), len(keep)])
        yield qrom.on_registers(
            selection_register=selection,
            selection_ancilla=ancilla,
            target_register=[alt, keep],
        )
        yield cirq.H.on_each(*sigma_mu)
        yield LessThanEqualGate([2] * self._mu, [2] * self._mu).on(
            *sigma_mu, *keep, less_than_equal
        )
        yield [cirq.CSWAP(less_than_equal, a, s) for (a, s) in zip(alt, selection)]

    def on_registers(
        self,
        *,
        selection_register: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
        temp_register: Sequence[cirq.Qid],
    ) -> cirq.GateOperation:
        return cirq.GateOperation(
            self,
            list(selection_register) + list(temp_register) + list(selection_ancilla),
        )
