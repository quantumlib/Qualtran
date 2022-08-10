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
        selection_bitsize: int,
        target_bitsize: int,
        select_unitaries: List[cirq.DensePauliString],
    ):
        """
        An implementation of the SELECT unitary using the `UnaryIterationGate`

        Args:
            selection_bitsize: Number of qubits needed for select register. This is ceil(log2(len(select_unitaries)))
            target_bitsize: number of qubits in the target register.
            select_unitaries: List of DensePauliString's to apply to target register. Each dense
                pauli string must contain `target_bitsize` terms.

        Raises:
            ValueError if any(len(dps) != target_bitsize for dps in select_unitaries).
        """
        if any(len(dps) != target_bitsize for dps in select_unitaries):
            raise ValueError(
                f"Each dense pauli string in `select_unitaries` should contain "
                f"{target_bitsize} terms."
            )
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self.select_unitaries = select_unitaries
        if self._selection_bitsize < (len(select_unitaries) - 1).bit_length():
            raise ValueError("Input selection_bitsize is not consistent with select_unitaries")

    @property
    def control_bitsize(self) -> int:
        return 1

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        return self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return len(self.select_unitaries)

    def nth_operation(self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Args:
             n: takes on values [0, len(self.selection_unitaries))
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        if n < 0 or n >= 2**self.selection_bitsize:
            raise ValueError("n is outside selection length range")
        return self.select_unitaries[n].on(*target).with_coefficient(1).controlled_by(control)
