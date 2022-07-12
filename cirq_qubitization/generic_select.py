from typing import Sequence, List
import cirq
from cirq_qubitization import unary_iteration


class GenericSelect(unary_iteration.UnaryIterationGate):
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
            select_unitaries: List of DensePauliString's to apply to target register. Each dense
            pauli string must contain `target_register_length` terms.

        Raises:
            ValueError if any(len(dps) != target_register_length for dps in select_unitaries).
        """
        if any(len(dps) != target_register_length for dps in select_unitaries):
            raise ValueError(
                f"Each dense pauli string in `select_unitaries` should contain "
                f"{target_register_length} terms."
            )
        self.selection_length = selection_register_length
        self.target_length = target_register_length
        self.select_unitaries = select_unitaries
        if self.selection_length < (len(select_unitaries) - 1).bit_length():
            raise ValueError(
                "Input selection_register_length is not consistent with select_unitaries"
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
        Args:
             n: takes on values [0, len(self.selection_unitaries))
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        if n < 0 or n >= 2**self.selection_length:
            raise ValueError("n is outside selection length range")
        return (
            self.select_unitaries[n]
            .on(*target)
            .with_coefficient(1)
            .controlled_by(control)
        )
