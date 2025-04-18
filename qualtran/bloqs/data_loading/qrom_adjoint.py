import dataclasses
import itertools
from typing import List, Sequence, Tuple

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
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.resource_counting.symbolic_counting_utils import SymbolicInt, is_symbolic, log2

@dataclasses.dataclass(frozen=True)
class QROMAdjCondition(Condition):
    key: cirq.MeasurementKey
    dx: List[int]

    def replace_key(self, current: cirq.MeasurementKey, replacement: cirq.MeasurementKey):
        return QROMAdjCondition(replacement, self.dx) if self.key == current else self

    def resolve(self, classical_data: cirq.ClassicalDataStoreReader) -> bool:
        y = classical_data.get_digits(self.key)
        active = False
        for yi, dxi in zip(y, self.dx):
            active = not active if yi * dxi == 1 else active
        return active
    


@attrs.define
class QROMWithClassicalControls(QROM):
    QROM_bloq: QROM = field(default=None)
    mz_key: str = field(default="target_mzs")

    def calc_dx(self, x):
        bitstring = []
        x_start = 0
        for i in range(len(self.QROM_bloq.target_bitsizes)):
            bitsize = self.QROM_bloq.target_bitsizes[i]
            data = self.QROM_bloq.data[i][x[x_start:x_start + bitsize]]
            bitstring.extend(iter_bits(data, bitsize))
        return bitstring


    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx: int = kwargs[self.selection_registers[0].name]
        target_regs = {reg.name: kwargs[reg.name] for reg in self.target_registers}
        # yield self._load_nth_data(selection_idx, lambda q: CNOT().on(control, q), **target_regs)
        # for i, d in enumerate(self.data):
        #     target = target_regs.get(f'target{i}_', ())
        target = target_regs.get(f'target{0}_', ())
        # for q, bit in zip(target, f'{int(self.data[0][selection_idx]):0{len(target)}b}'):
        #     if int(bit):
        #         yield gate(q)
        N = int(log2(len(target)))
        selection_bits = iter_bits(selection_idx, self.selection_bitsizes[0])
        for i in range(len(target)):
            target_bits = iter_bits(i, N)
            dx = self.calc_dx(list(itertools.chain(selection_bits, target_bits)))
            yield cirq.X(target[i]).with_classical_controls(QROMAdjCondition(cirq.MeasurementKey(self.mz_key), dx))


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

