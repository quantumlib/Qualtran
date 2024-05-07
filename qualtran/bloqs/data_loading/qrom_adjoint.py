import dataclasses
from typing import List, Sequence

import attrs
import cirq
from attr import field
from cirq import Condition
from numpy._typing import NDArray

from qualtran import Signature
from qualtran.bloqs.data_loading import QROM
from qualtran.bloqs.data_loading.one_hot_encoding import OneHotEncoding
from qualtran.bloqs.data_loading.qrom import _to_tuple
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.resource_counting.symbolic_counting_utils import SymbolicInt, is_symbolic, log2

@dataclasses.dataclass(frozen=True)
class QROMAdjCondition(Condition):
    D_x: int


@attrs.define
class QROMWithClassicalControls(QROM):
    QROM_bloq: QROM = field(default=None)

    def D_x(self, x):


    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = {reg.name: kwargs[reg.name] for reg in self.target_registers}
        yield self._load_nth_data(selection_idx, lambda q: CNOT().on(control, q), **target_regs)

@attrs.frozen
class QROMAdj():
    QROM_Bloq: QROM = field()
    num_ancilla: SymbolicInt = field(default=1)
    mz_key: str = field(default="target_mzs")

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
                yield cirq.measure(target, key=f"target_mzs")
        ancillas = context.qubit_manager.qalloc(self.num_ancilla)
        if len(self.QROM_Bloq.selection_registers) == 1:
            selection_regs = quregs['selection']
        else:
            selection_regs = [quregs[f"selection{i}"] for i in range(len(self.QROM_Bloq.selection_registers))]
        selection_regs = selection_regs.flatten()
        binary_int_size = int(log2(self.num_ancilla))
        yield OneHotEncoding(binary_int_size).on_registers(a=selection_regs[-binary_int_size:], b=ancillas)

