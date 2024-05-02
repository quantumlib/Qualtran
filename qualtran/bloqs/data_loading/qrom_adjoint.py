import attrs
import cirq
from attr import field
from numpy._typing import NDArray

from qualtran import Signature
from qualtran.bloqs.data_loading import QROM
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.resource_counting.symbolic_counting_utils import SymbolicInt, is_symbolic, log2


@attrs.frozen
class QROM():
    QROM_Bloq: QROM = field()
    num_ancilla: SymbolicInt = field(default=1)

    @num_ancilla.validator
    def num_ancilla_validator(self, field, val):
        if is_symbolic(val):
            return
        if not log2(val).is_integer():
            raise ValueError(f"num_ancilla must be a power of 2, but got {val}")

    def signature(self) -> Signature:
        return self.QROM_Bloq.signature


    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        num_targets = len(self.QROM_Bloq.target_registers)
        for i in range(num_targets):
            targets = quregs[f'target{i}']
            for target in targets:
                yield cirq.H(target)
        for i in range(num_targets):
            targets = quregs[f'target{i}']
            for j, target in enumerate(targets):
                yield cirq.measure(target, key=f"{i}{j}")

