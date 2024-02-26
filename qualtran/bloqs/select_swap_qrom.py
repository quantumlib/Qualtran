#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import List, Optional, Sequence, Tuple

import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    BoundedQUInt,
    CompositeBloq,
    QAny,
    Register,
    Signature,
    Soquet,
)
from qualtran._infra.gate_with_registers import merge_qubits, split_qubits, total_bits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.drawing import Circle, TextBox, WireSymbol


def find_optimal_log_block_size(iteration_length: int, target_bitsize: int) -> int:
    """Find optimal block size, which is a power of 2, for SelectSwapQROM.

    This functions returns the optimal `k` s.t.
     - k is in an integer and k >= 0.
     - iteration_length/2^k + target_bitsize*(2^k - 1) is minimized.

    The corresponding block size for SelectSwapQROM would be 2^k.
    """
    k = 0.5 * np.log2(iteration_length / target_bitsize)
    if k < 0:
        return 1

    def value(kk: List[int]):
        return iteration_length / np.power(2, kk) + target_bitsize * (np.power(2, kk) - 1)

    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    return int(k_int[np.argmin(value(k_int))])  # obtain optimal k


@cirq.value_equality()
class SelectSwapQROM(Bloq):
    r"""Gate to load `data[l]` in the target register when the selection register stores integer `l`.

    Let `N` be the number of data elements to load; and `b` be the bit-size of the target
    register in which data elements should be loaded.

    The `SelectSwapQROM` is a hybrid of the following two existing primitives:

     - Unary Iteration based `QROM` requires O(N) T-gates to load `N` data
       elements into a `b`-sized target register. Note that the T-complexity is independent of `b`.
     - `SwapWithZeroGate` can swap a `b`-sized register at index `x` with a `b`-sized
       register at index `0` using O(b) T-gates, if the selection register stores integer `x`.
       Note that the swap complexity is independent of the iteration length `N`.

    The `SelectSwapQROM` uses a square root decomposition by combining the above two approaches to
    further optimize the T-gate complexity of loading `N` data elements, each into a `b`-sized
    target register as follows:

     - Divide the `N` data elements into batches of size `B` (a variable) and
       load each batch simultaneously into `B` distinct target signature using the conventional
       QROM. This has T-complexity `O(N / B)`.
     - Use `SwapWithZeroGate` to swap the `i % B`'th target register in batch number `i / B`
       to load `data[i]` in the 0'th target register. This has T-complexity `O(B * b)`.

    Thus, the final T-complexity of `SelectSwapQROM` is $O(B * b + N / B)$ T-gates; where $B$ is
    the block-size with an optimal value of $O(\sqrt(N / b))$.

    This improvement in T-complexity is achieved at the cost of using an additional $O(B * b)$
    ancilla qubits, with a nice property that these additional ancillas can be "dirty".
    They don't need to start in the |0> state and can be borrowed from other parts of the
    algorithm. The these dirty ancillas are restored to their original state after the operation.

    Args:
        data: Sequence of integers to load in the target register. If more than one sequence
            is provided, each sequence must be of the same length.
        target_bitsizes: Sequence of integers describing the size of target register for each
            data sequence to load. Defaults to `max(data[i]).bit_length()` for each i.
        block_size(B): Load batches of `B` data elements in each iteration of traditional QROM
            (N/B iterations required). Complexity of SelectSwap QROAM scales as
            `O(B * b + N / B)`, where `B` is the block_size. Defaults to optimal value of
            `O(sqrt{N / b})`.

    Registers:
        selection: The selection register of size `logN` to index into each `data`.
        target{i}_: The target registers, one for each `data` to load the data.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(
        self,
        *data: Sequence[int],
        target_bitsizes: Optional[Sequence[int]] = None,
        block_size: Optional[int] = None,
    ):
        # Validate input.
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        if target_bitsizes is None:
            target_bitsizes = [int(max(d)).bit_length() for d in data]
        assert len(target_bitsizes) == len(data)
        assert all(t >= int(max(d)).bit_length() for t, d in zip(target_bitsizes, data))
        self._num_sequences = len(data)
        self._target_bitsizes = tuple(target_bitsizes)
        self._iteration_length = len(data[0])
        if block_size is None:
            # Figure out optimal value of block_size
            block_size = 2 ** find_optimal_log_block_size(len(data[0]), sum(target_bitsizes))
        assert 0 < block_size <= self._iteration_length
        self._block_size = block_size
        self._num_blocks = int(np.ceil(self._iteration_length / self.block_size))
        self.selection_q, self.selection_r = tuple(
            (L - 1).bit_length() for L in [self.num_blocks, self.block_size]
        )
        self._data = tuple(tuple(d) for d in data)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        # See https://github.com/quantumlib/Qualtran/issues/556 for unusual placement of underscore.
        return tuple(
            Register(f'target{sequence_id}_', QAny(self._target_bitsizes[sequence_id]))
            for sequence_id in range(self._num_sequences)
        )

    @cached_property
    def signature(self) -> Signature:
        selection_reg = Register(
            'selection', BoundedQUInt(self.selection_q + self.selection_r, self._iteration_length)
        )
        return Signature([selection_reg, *self.target_registers])

    @property
    def data(self) -> Tuple[Tuple[int, ...], ...]:
        return self._data

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        # Divide each data sequence and corresponding target registers into
        # `self.num_blocks` batches of size `self.block_size`.
        selection, targets = quregs.pop('selection'), quregs
        qrom_data: List[NDArray] = []
        qrom_target_bitsizes: List[int] = []
        ordered_target_qubits: List[cirq.Qid] = []
        for block_id in range(self.block_size):
            for sequence_id in range(self._num_sequences):
                data = self.data[sequence_id]
                target_bitsize = self._target_bitsizes[sequence_id]
                ordered_target_qubits.extend(context.qubit_manager.qborrow(target_bitsize))
                data_for_current_block = data[block_id :: self.block_size]
                if len(data_for_current_block) < self.num_blocks:
                    zero_pad = (0,) * (self.num_blocks - len(data_for_current_block))
                    data_for_current_block = data_for_current_block + zero_pad
                qrom_data.append(np.array(data_for_current_block))
                qrom_target_bitsizes.append(target_bitsize)
        # Construct QROM, SwapWithZero and CX operations using the batched data and qubits.
        k = (self.block_size - 1).bit_length()
        q, r = selection[: self.selection_q], selection[self.selection_q :]
        qrom_gate = QROM(
            qrom_data,
            selection_bitsizes=(self.selection_q,),
            target_bitsizes=tuple(qrom_target_bitsizes),
        )
        qrom_targets = split_qubits(qrom_gate.target_registers, ordered_target_qubits)

        clean_targets = merge_qubits(self.target_registers, **targets)
        cnot_op = [CNOT().on(s, t) for s, t in zip(ordered_target_qubits, clean_targets)]

        if self.block_size > 1:
            swz_gate = SwapWithZero(k, total_bits(self.target_registers), self.block_size)
            swz_targets = split_qubits(swz_gate.target_registers, ordered_target_qubits)

            yield qrom_gate.on_registers(selection=q, **qrom_targets)
            yield swz_gate.on_registers(selection=r, **swz_targets)
            yield cnot_op
            yield swz_gate.adjoint().on_registers(selection=r, **swz_targets)
            yield qrom_gate.adjoint().on_registers(selection=q, **qrom_targets)
            yield swz_gate.on_registers(selection=r, **swz_targets)
            yield cnot_op
            yield swz_gate.adjoint().on_registers(selection=r, **swz_targets)
            return
        else:
            yield qrom_gate.on_registers(selection=q, **qrom_targets)
            yield cnot_op
            yield qrom_gate.adjoint().on_registers(selection=q, **qrom_targets)
            yield cnot_op

        context.qubit_manager.qfree(ordered_target_qubits)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In_q"] * self.selection_q
        wire_symbols += ["In_r"] * self.selection_r
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROAM_{i}"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        name = soq.reg.name
        if name == 'selection':
            return TextBox('In')
        elif 'target' in name:
            trg_indx = int(name.replace('target', '').replace('_', ''))
            # match the sel index
            subscript = trg_indx
            return TextBox(f'QROAM_{subscript}')
        elif name == 'control':
            return Circle()

    def short_name(self) -> str:
        return 'QROAM'

    def _value_equality_values_(self):
        return self.block_size, self._target_bitsizes, self.data


@bloq_example
def _ss_qrom() -> SelectSwapQROM:
    rs = np.random.RandomState(52)
    datas = rs.randint(0, 256, size=(3, 11))
    ss_qrom = SelectSwapQROM(*datas)
    return ss_qrom


_SS_QROM_DOC = BloqDocSpec(
    bloq_cls=SelectSwapQROM,
    import_line='from qualtran.bloqs.select_swap_qrom import SelectSwapQROM',
    examples=[_ss_qrom],
)
