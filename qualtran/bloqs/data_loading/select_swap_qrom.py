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
import numbers
from collections import defaultdict
from functools import cached_property
from typing import cast, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import ArrayLike, NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.qrom_base import QROMBase
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import ceil, is_symbolic, log2, prod, SymbolicInt

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def find_optimal_log_block_size(
    iteration_length: SymbolicInt, target_bitsize: SymbolicInt
) -> SymbolicInt:
    """Find optimal block size, which is a power of 2, for SelectSwapQROM.

    This functions returns the optimal `k` s.t.
        * k is in an integer and k >= 0.
        * iteration_length/2^k + target_bitsize*(2^k - 1) is minimized.
    The corresponding block size for SelectSwapQROM would be 2^k.
    """
    k = 0.5 * log2(iteration_length / target_bitsize)
    if is_symbolic(k):
        return ceil(k)

    if k < 0:
        return 0

    def value(kk: List[int]):
        return iteration_length / np.power(2, kk) + target_bitsize * (np.power(2, kk) - 1)

    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    return int(k_int[np.argmin(value(k_int))])  # obtain optimal k


@cirq.value_equality(distinct_child_types=True)
@attrs.frozen
class SelectSwapQROM(QROMBase, GateWithRegisters):  # type: ignore[misc]
    """Gate to load data[l] in the target register when the selection register stores integer l.

    Let
        N:= Number of data elements to load.
        b:= Bit-length of the target register in which data elements should be loaded.

    The `SelectSwapQROM` is a hybrid of the following two existing primitives:

    - Unary Iteration based `QROM` requires O(N) T-gates to load `N` data
    elements into a b-bit target register. Note that the T-complexity is independent of `b`.
    - `SwapWithZeroGate` can swap a `b` bit register indexed `x` with a `b`
    bit register at index `0` using O(b) T-gates, if the selection register stores integer `x`.
    Note that the swap complexity is independent of the iteration length `N`.

    The `SelectSwapQROM` uses square root decomposition by combining the above two approaches to
    further optimize the T-gate complexity of loading `N` data elements, each into a `b` bit
    target register as follows:

    - Divide the `N` data elements into batches of size `B` (a variable) and
    load each batch simultaneously into `B` distinct target signature using the conventional
    QROM. This has T-complexity `O(N / B)`.
    - Use `SwapWithZeroGate` to swap the `i % B`'th target register in batch number `i / B`
    to load `data[i]` in the 0'th target register. This has T-complexity `O(B * b)`.

    This, the final T-complexity of `SelectSwapQROM` is `O(B * b + N / B)` T-gates; where `B` is
    the block-size with an optimal value of `O(sqrt(N / b))`.

    This improvement in T-complexity is achieved at the cost of using an additional `O(B * b)`
    ancilla qubits, with a nice property that these additional ancillas can be `dirty`; i.e.
    they don't need to start in the |0> state and thus can be borrowed from other parts of the
    algorithm. The state of these dirty ancillas would be unaffected after the operation has
    finished.

    For more details, see the reference below:

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low, Kliuchnikov, Schaeffer. 2018.

        [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization](https://arxiv.org/abs/1902.02134).
            Berry et. al. (2019). Appendix A. and B.
    """

    log_block_sizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    use_dirty_ancilla: bool = True

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    # Builder methods and helpers.
    @log_block_sizes.default
    def _default_block_sizes(self) -> Tuple[SymbolicInt, ...]:
        target_bitsize = sum(self.target_bitsizes) * sum(
            prod(*shape) for shape in self.target_shapes
        )
        return tuple(find_optimal_log_block_size(ilen, target_bitsize) for ilen in self.data_shape)

    @classmethod
    def build_from_data(
        cls: Type['SelectSwapQROM'],
        *data: ArrayLike,
        target_bitsizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        use_dirty_ancilla: bool = True,
    ) -> 'SelectSwapQROM':
        qroam: 'SelectSwapQROM' = cls._build_from_data(
            *data, target_bitsizes=target_bitsizes, num_controls=num_controls
        )
        qroam = attrs.evolve(qroam, use_dirty_ancilla=use_dirty_ancilla)
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    @classmethod
    def build_from_bitsize(
        cls: Type['SelectSwapQROM'],
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        use_dirty_ancilla: bool = True,
    ) -> 'SelectSwapQROM':
        qroam: 'SelectSwapQROM' = cls._build_from_bitsize(
            data_len_or_shape,
            target_bitsizes,
            selection_bitsizes=selection_bitsizes,
            num_controls=num_controls,
        )
        qroam = attrs.evolve(qroam, use_dirty_ancilla=use_dirty_ancilla)
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    def with_log_block_sizes(
        self, log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None
    ) -> 'SelectSwapQROM':
        if log_block_sizes is None:
            return self
        if isinstance(log_block_sizes, (int, sympy.Basic, numbers.Number)):
            log_block_sizes = (log_block_sizes,)
        if not is_symbolic(*log_block_sizes):
            assert all(1 <= 2**bs <= ilen for bs, ilen in zip(log_block_sizes, self.data_shape))
        return attrs.evolve(self, log_block_sizes=log_block_sizes)

    @cached_property
    def block_sizes(self) -> Tuple[SymbolicInt, ...]:
        return tuple(2**log_K for log_K in self.log_block_sizes)

    @cached_property
    def batched_qrom_shape(self) -> Tuple[SymbolicInt, ...]:
        return tuple(ceil(N / K) for N, K in zip(self.data_shape, self.block_sizes))

    @cached_property
    def batched_qrom_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        return tuple(s - log_K for s, log_K in zip(self.selection_bitsizes, self.log_block_sizes))

    @cached_property
    def padded_data(self) -> List[np.ndarray]:
        pad_width = tuple(
            (0, ceil(N / K) * K - N) for N, K in zip(self.data_shape, self.block_sizes)
        )
        return [np.pad(d, pad_width) for d in self.data]

    @cached_property
    def batched_data_shape(self) -> Tuple[int, ...]:
        return cast(Tuple[int, ...], self.batched_qrom_shape + self.block_sizes)

    @cached_property
    def batched_data(self) -> List[np.ndarray]:
        # In SelectSwapQROM, for N-dimensional data (one or more datasets), you pick block sizes for
        # each dimension and load a batched N-dimensional output "at-once" using a traditional QROM read
        # followed by an N-dimensional SwapWithZero swap.
        #
        # For data[N1][N2] with block sizes (k1, k2), you load batches of size `(k1, k2)` at once.
        # Thus, you load batch[N1/k1][N2/k2] where batch[i][j] = data[i*k1:(i + 1)*k1][j*k2:(j + 1)*k2]
        batched_data = [np.empty(self.batched_data_shape) for _ in range(len(self.target_bitsizes))]
        block_slices = [slice(0, k) for k in self.block_sizes]
        for i, data in enumerate(self.padded_data):
            for batch_idx in np.ndindex(cast(Tuple[int, ...], self.batched_qrom_shape)):
                data_idx = [slice(x * k, (x + 1) * k) for x, k in zip(batch_idx, self.block_sizes)]
                batched_data[i][(*batch_idx, *block_slices)] = data[tuple(data_idx)]
        return batched_data

    @cached_property
    def qrom_bloq(self) -> QROM:
        return QROM.build_from_bitsize(
            self.batched_qrom_shape,
            self.target_bitsizes,
            target_shapes=(self.block_sizes,) * len(self.target_bitsizes),
            selection_bitsizes=self.batched_qrom_selection_bitsizes,
            num_controls=self.num_controls,
        )

    @cached_property
    def swap_with_zero_bloqs(self) -> List[SwapWithZero]:
        return [
            SwapWithZero(
                self.log_block_sizes,
                target_bitsize=target_bitsize,
                n_target_registers=self.block_sizes,
            )
            for target_bitsize in self.target_bitsizes
        ]

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        ret: Dict[Bloq, SymbolicInt] = defaultdict(lambda: 0)
        toggle_overhead = 2 if self.use_dirty_ancilla else 1
        ret[self.qrom_bloq] += 1
        ret[self.qrom_bloq.adjoint()] += 1
        ret[CNOT()] += toggle_overhead * total_bits(self.target_registers)
        for swz in self.swap_with_zero_bloqs:
            if any(is_symbolic(s) or s > 0 for s in swz.selection_bitsizes):
                ret[swz] += toggle_overhead
                ret[swz.adjoint()] += toggle_overhead
        return set(ret.items())

    def _alloc_qrom_target_qubits(
        self, reg: Register, qm: cirq.QubitManager
    ) -> NDArray[cirq.Qid]:  # type:ignore[type-var]
        qubits = (
            qm.qborrow(total_bits([reg]))
            if self.use_dirty_ancilla
            else qm.qalloc(total_bits([reg]))
        )
        return np.array(qubits).reshape(reg.shape + (reg.bitsize,))

    def decompose_from_registers(  # type: ignore[return]
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        # 1. Construct QROM to load the batched data.
        qrom = self.qrom_bloq.with_data(*self.batched_data)
        qrom_ctrls = {reg.name: quregs[reg.name] for reg in qrom.control_registers}
        qrom_selection = {
            qrom_reg.name: quregs[sel_reg.name][: qrom_reg.bitsize]
            for sel_reg, qrom_reg in zip(self.selection_registers, qrom.selection_registers)
        }
        qrom_targets = {
            reg.name: self._alloc_qrom_target_qubits(reg, context.qubit_manager)
            for reg in qrom.target_registers
        }
        qrom_op = qrom.on_registers(**qrom_ctrls, **qrom_selection, **qrom_targets)
        # 2. Construct SwapWithZero.
        swz_ops = []
        assert len(qrom_targets) == len(self.swap_with_zero_bloqs)
        for targets, swz in zip(qrom_targets.values(), self.swap_with_zero_bloqs):
            if len(targets) <= 1:
                continue
            swz_selection = {
                swz_reg.name: quregs[sel_reg.name][-swz_reg.bitsize :]
                for sel_reg, swz_reg in zip(self.selection_registers, swz.selection_registers)
            }
            swz_ops.append(swz.on_registers(**swz_selection, targets=targets))
        # 3. Construct CNOTs from 0th borrowed register to clean target registers.
        cnot_ops = []
        for qrom_batched_target, target_reg in zip(qrom_targets.values(), self.target_registers):
            cnot_ops += [
                [cirq.CNOT(a, b) for a, b in zip(qrom_batched_target[0], quregs[target_reg.name])]
            ]

        # Yield the operations in correct order.
        if any(b > 0 for b in self.block_sizes):
            yield qrom_op
            yield swz_ops
            yield cnot_ops
            yield cirq.inverse(swz_ops)
            yield cirq.inverse(qrom_op)
            if self.use_dirty_ancilla:
                yield swz_ops
                yield cnot_ops
                yield cirq.inverse(swz_ops)
        else:
            yield qrom_op
            yield cnot_ops
            yield cirq.inverse(qrom_op)
            yield cnot_ops

        context.qubit_manager.qfree(
            [q for targets in qrom_targets.values() for q in targets.flatten()]
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        from qualtran.cirq_interop._bloq_to_cirq import _wire_symbol_to_cirq_diagram_info

        return _wire_symbol_to_cirq_diagram_info(self, args)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('QROAM')
        name = reg.name
        if name == 'selection':
            return TextBox('In')
        elif 'target' in name:
            trg_indx = int(name.replace('target', '').replace('_', ''))
            # match the sel index
            subscript = chr(ord('a') + trg_indx)
            return TextBox(f'QROAM_{subscript}')
        elif name == 'control':
            return Circle()
        raise ValueError(f'Unknown register name {name}')

    def _value_equality_values_(self):
        return self.log_block_sizes, *super()._value_equality_values_()


@bloq_example
def _qroam_multi_data() -> SelectSwapQROM:
    data1 = np.arange(5)
    data2 = np.arange(5) + 1
    qroam_multi_data = SelectSwapQROM.build_from_data([data1, data2])
    return qroam_multi_data


@bloq_example
def _qroam_multi_dim() -> SelectSwapQROM:
    data1 = np.arange(25).reshape((5, 5))
    data2 = (np.arange(25) + 1).reshape((5, 5))
    qroam_multi_dim = SelectSwapQROM.build_from_data([data1, data2])
    return qroam_multi_dim


@bloq_example
def _qroam_symb_dirty_1d() -> SelectSwapQROM:
    N, b, k, c = sympy.symbols('N b k c')
    qroam_symb_dirty_1d = SelectSwapQROM.build_from_bitsize(
        (N,), (b,), log_block_sizes=(k,), num_controls=c
    )
    return qroam_symb_dirty_1d


@bloq_example
def _qroam_symb_dirty_2d() -> SelectSwapQROM:
    N, M, b1, b2, k1, k2, c = sympy.symbols('N M b1 b2 k1 k2 c')
    log_block_sizes = (k1, k2)
    qroam_symb_dirty_2d = SelectSwapQROM.build_from_bitsize(
        (N, M), (b1, b2), log_block_sizes=log_block_sizes, num_controls=c
    )
    return qroam_symb_dirty_2d


@bloq_example
def _qroam_symb_clean_1d() -> SelectSwapQROM:
    N, b, k, c = sympy.symbols('N b k c')
    qroam_symb_clean_1d = SelectSwapQROM.build_from_bitsize(
        (N,), (b,), log_block_sizes=(k,), num_controls=c, use_dirty_ancilla=False
    )
    return qroam_symb_clean_1d


@bloq_example
def _qroam_symb_clean_2d() -> SelectSwapQROM:
    N, M, b1, b2, k1, k2, c = sympy.symbols('N M b1 b2 k1 k2 c')
    log_block_sizes = (k1, k2)
    qroam_symb_clean_2d = SelectSwapQROM.build_from_bitsize(
        (N, M), (b1, b2), log_block_sizes=log_block_sizes, num_controls=c, use_dirty_ancilla=False
    )
    return qroam_symb_clean_2d


_SELECT_SWAP_QROM_DOC = BloqDocSpec(
    bloq_cls=SelectSwapQROM,
    import_line='from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM',
    examples=[
        _qroam_multi_data,
        _qroam_multi_dim,
        _qroam_symb_dirty_1d,
        _qroam_symb_dirty_2d,
        _qroam_symb_clean_1d,
        _qroam_symb_clean_2d,
    ],
)
