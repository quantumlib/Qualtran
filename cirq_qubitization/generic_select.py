from typing import Sequence, List
import numpy as np
import cirq
import cirq_qubitization


class GenericSelect(cirq_qubitization.UnaryIterationGate):
    """
    Gate that implements SELECT for a Hamiltonian expressed as an LCU.

    Recall: SELECT = \sum_{l}|l><l| \otimes U_{l}

    The first log(L) qubits is the index register and the last M qubits are the system
    register U_{l} is applied to
    """

    def __init__(
        self,
        selection_register_length: int,
        target_register_length: int,
        select_unitaries: List[cirq.DensePauliString],
    ):
        """
        An implementation of the SELECT unitary using the `UnaryIterationGate`

        Args:
            selection_length: Number of qubits needed for select register. This is ceil(log2(len(select_unitaries)))
            target_length: number of qubits in the target register.
            select_unitaries: List of Paulistrings to apply to target register

        Caveat: this is not really a gate since select_unitaries contain qubits and we require
                the user to correctly assign these qubits consistent with the "target_length" register
        """
        self.selection_length = selection_register_length
        self.target_length = target_register_length
        self.select_unitaries = select_unitaries
        if len(select_unitaries) <= int(np.log(self.selection_length)):
            raise ValueError(
                "Input select length is not consistent with select_unitaries"
            )

    @property
    def control_register(self) -> int:
        return 1

    @property
    def selection_register(self) -> int:
        return self.selection_length

    @property
    def target_register(self) -> int:
        return self.target_length

    @property
    def iteration_length(self) -> int:
        return len(self.select_unitaries)

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """

        :param n: takes on values [0, len(self.selection_unitaries))
        :param control: Qid that is the control qubit or qubits
        :param target: Target register qubits
        """
        if n < 0 or n >= 2**self.selection_length:
            raise ValueError("n is outside selection length range")
        return (
            self.select_unitaries[n]
            .on(*target)
            .with_coefficient(1)
            .controlled_by(control)
        )
