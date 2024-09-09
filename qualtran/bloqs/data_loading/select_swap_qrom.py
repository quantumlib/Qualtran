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
from typing import cast, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, TypeVar, Union

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import ArrayLike

from qualtran import bloq_example, BloqDocSpec, BQUInt, GateWithRegisters, Register, Signature
from qualtran.bloqs.arithmetic.bitwise import Xor
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.qrom_base import QROMBase
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import ceil, is_symbolic, log2, prod, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import Bloq, BloqBuilder, QDType, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

SelSwapQROM_T = TypeVar('SelSwapQROM_T', bound='SelectSwapQROM')


def find_optimal_log_block_size(
    iteration_length: SymbolicInt, target_bitsize: SymbolicInt, use_dirty_ancilla: bool = False
) -> SymbolicInt:
    """Find optimal block size, which is a power of 2, for SelectSwapQROM.

    This functions returns the optimal `k` s.t.
        * k is in an integer and k >= 0.
        * iteration_length/2^k + target_bitsize*(2^k - 1) is minimized if use_dirty_ancilla is False
        * iteration_length/2^k + 2*target_bitsize*(2^k - 1) is minimized if use_dirty_ancilla is True

    The corresponding block size for SelectSwapQROM would be 2^k.
    """
    if not use_dirty_ancilla:
        target_bitsize = 2 * target_bitsize
    k: SymbolicFloat = 0.5 * log2(iteration_length / target_bitsize)
    if is_symbolic(k):
        return ceil(k)

    if k < 0:
        return 0

    def value(kk: List[int]):
        return iteration_length / np.power(2, kk) + target_bitsize * (np.power(2, kk) - 1)

    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    return int(k_int[np.argmin(value(k_int))])  # obtain optimal k


def _find_optimal_log_block_size_helper(qrom: 'SelectSwapQROM') -> Tuple[SymbolicInt, ...]:
    target_bitsize = sum(qrom.target_bitsizes) * sum(prod(shape) for shape in qrom.target_shapes)
    return tuple(
        find_optimal_log_block_size(ilen, target_bitsize, qrom.use_dirty_ancilla)
        for ilen in qrom.data_shape
    )


def _alloc_anc_for_reg(
    bb: 'BloqBuilder', dtype: 'QDType', shape: Tuple[int, ...], dirty: bool
) -> 'SoquetT':
    if not shape:
        return bb.allocate(dtype=dtype, dirty=dirty)
    soqs = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        soqs[idx] = bb.allocate(dtype=dtype, dirty=dirty)
    return soqs


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
        if x is not None
        else x,
        default=None,
    )
    use_dirty_ancilla: bool = True

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.target_shapes != self._default_target_shapes():
            raise ValueError(
                f"{type(self)} currently only supports target registers of shape (). Found {self.target_shapes}"
            )
        if self.log_block_sizes is None:
            object.__setattr__(self, "log_block_sizes", _find_optimal_log_block_size_helper(self))

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    # Builder methods and helpers.
    def is_symbolic(self) -> bool:
        return super().is_symbolic() or is_symbolic(*self.log_block_sizes)

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
        if log_block_sizes is None:
            log_block_sizes = _find_optimal_log_block_size_helper(qroam)
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
        if log_block_sizes is None:
            log_block_sizes = _find_optimal_log_block_size_helper(qroam)
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    def with_log_block_sizes(
        self: SelSwapQROM_T,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
    ) -> 'SelSwapQROM_T':
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
        batched_data = [np.zeros(self.batched_data_shape, dtype=int) for _ in self.target_bitsizes]
        block_slices = [slice(0, k) for k in self.block_sizes]
        for i, data in enumerate(self.padded_data):
            for batch_idx in np.ndindex(cast(Tuple[int, ...], self.batched_qrom_shape)):
                data_idx = [slice(x * k, (x + 1) * k) for x, k in zip(batch_idx, self.block_sizes)]
                batched_data[i][(*batch_idx, *block_slices)] = data[tuple(data_idx)]
        return batched_data

    @cached_property
    def qrom_bloq(self) -> QROM:
        qrom = QROM.build_from_bitsize(
            self.batched_qrom_shape,
            self.target_bitsizes,
            target_shapes=(self.block_sizes,) * len(self.target_bitsizes),
            selection_bitsizes=self.batched_qrom_selection_bitsizes,
            num_controls=self.num_controls,
        )
        return qrom if is_symbolic(self) else qrom.with_data(*self.batched_data)

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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: Dict[Bloq, SymbolicInt] = defaultdict(lambda: 0)
        toggle_overhead = 2 if self.use_dirty_ancilla else 1
        ret[self.qrom_bloq] += 1
        ret[self.qrom_bloq.adjoint()] += 1
        for reg in self.target_registers:
            ret[Xor(reg.dtype)] += toggle_overhead * np.prod(reg.shape, dtype=int)
        for swz in self.swap_with_zero_bloqs:
            if any(is_symbolic(s) or s > 0 for s in swz.selection_bitsizes):
                ret[swz] += toggle_overhead
                ret[swz.adjoint()] += toggle_overhead
        return ret

    def _add_qrom_bloq(
        self,
        bb: 'BloqBuilder',
        ctrls: List['SoquetT'],
        sel_l: List['SoquetT'],
        targets: List['SoquetT'],
        uncompute: bool = False,
    ) -> Tuple[List['SoquetT'], List['SoquetT'], List['SoquetT']]:
        in_soqs = {reg.name: soq for reg, soq in zip(self.qrom_bloq.control_registers, ctrls)}
        in_soqs |= {reg.name: soq for reg, soq in zip(self.qrom_bloq.selection_registers, sel_l)}
        in_soqs |= {reg.name: soq for reg, soq in zip(self.qrom_bloq.target_registers, targets)}
        out_soqs = bb.add_d(self.qrom_bloq.adjoint() if uncompute else self.qrom_bloq, **in_soqs)
        ctrls = [out_soqs[reg.name] for reg in self.qrom_bloq.control_registers]
        sel_l = [out_soqs[reg.name] for reg in self.qrom_bloq.selection_registers]
        targets = [out_soqs[reg.name] for reg in self.qrom_bloq.target_registers]
        return ctrls, sel_l, targets

    def _add_swap_with_zero_bloq(
        self,
        bb: 'BloqBuilder',
        selection: List['SoquetT'],
        targets: List['SoquetT'],
        uncompute: bool = False,
    ) -> Tuple[List['SoquetT'], List['SoquetT']]:
        # Get soquets for SwapWithZero
        assert len(targets) == len(self.swap_with_zero_bloqs)
        sel_names = [reg.name for reg in self.swap_with_zero_bloqs[0].selection_registers]
        soqs = {sel_name: soq for sel_name, soq in zip(sel_names, selection)}
        out_targets: List['SoquetT'] = []
        for target, swz in zip(targets, self.swap_with_zero_bloqs):
            soqs['targets'] = target
            soqs = bb.add_d(swz.adjoint() if uncompute else swz, **soqs)
            out_targets.append(soqs['targets'])
        return [soqs[reg_name] for reg_name in sel_names], out_targets

    def _add_cnot(
        self, bb: 'BloqBuilder', qrom_targets: List['SoquetT'], target: List['SoquetT']
    ) -> Tuple[List['SoquetT'], List['SoquetT']]:
        for i, qrom_reg in enumerate(qrom_targets):
            assert isinstance(qrom_reg, np.ndarray)  # Make mypy happy.
            idx = np.unravel_index(0, qrom_reg.shape)
            qrom_reg[idx], target[i] = bb.add(
                Xor(self.target_registers[i].dtype), x=qrom_reg[idx], y=target[i]
            )
        return qrom_targets, target

    @cached_property
    def _partition_selection_reg_bloqs(self) -> List[Partition]:
        partition_bloqs = []
        for reg, k in zip(self.selection_registers, self.log_block_sizes):
            preg = (
                Register('l', BQUInt(reg.bitsize - k, 2 ** (reg.bitsize - k))),
                Register('k', BQUInt(k, 2**k)),
            )
            partition_bloqs.append(Partition(reg.bitsize, preg))
        return partition_bloqs

    def _partition_sel_register(
        self, bb: 'BloqBuilder', selection: List['SoquetT']
    ) -> Tuple[List['SoquetT'], List['SoquetT']]:
        sel_l, sel_k = [], []
        for sel, pbloq in zip(selection, self._partition_selection_reg_bloqs):
            sl, sk = bb.add(pbloq, x=sel)
            sel_l.append(sl)
            sel_k.append(sk)
        return sel_l, sel_k

    def _unpartition_sel_register(
        self, bb: 'BloqBuilder', sel_l: List['SoquetT'], sel_k: List['SoquetT']
    ) -> List['SoquetT']:
        selection = []
        for l, k, pbloq in zip(sel_l, sel_k, self._partition_selection_reg_bloqs):
            selection.append(bb.add(pbloq.adjoint(), l=l, k=k))
        return selection

    def _build_composite_bloq_with_swz(
        self,
        bb: 'BloqBuilder',
        ctrl: List['SoquetT'],
        selection: List['SoquetT'],
        target: List['SoquetT'],
        qrom_targets: List['SoquetT'],
    ) -> Tuple[List['SoquetT'], List['SoquetT'], List['SoquetT'], List['SoquetT']]:
        sel_l, sel_k = self._partition_sel_register(bb, selection)
        # Partition selection registers into l & k
        ctrl, sel_l, qrom_targets = self._add_qrom_bloq(bb, ctrl, sel_l, qrom_targets)
        sel_k, qrom_targets = self._add_swap_with_zero_bloq(bb, sel_k, qrom_targets)
        qrom_targets, target = self._add_cnot(bb, qrom_targets, target)
        sel_k, qrom_targets = self._add_swap_with_zero_bloq(bb, sel_k, qrom_targets, uncompute=True)
        ctrl, sel_l, qrom_targets = self._add_qrom_bloq(
            bb, ctrl, sel_l, qrom_targets, uncompute=True
        )
        if self.use_dirty_ancilla:
            sel_k, qrom_targets = self._add_swap_with_zero_bloq(bb, sel_k, qrom_targets)
            qrom_targets, target = self._add_cnot(bb, qrom_targets, target)
            sel_k, qrom_targets = self._add_swap_with_zero_bloq(
                bb, sel_k, qrom_targets, uncompute=True
            )
        # UnPartition sel_l, sel_k into selection
        selection = self._unpartition_sel_register(bb, sel_l, sel_k)
        return ctrl, selection, target, qrom_targets

    def _build_composite_bloq_without_swz(
        self,
        bb: 'BloqBuilder',
        ctrl: List['SoquetT'],
        selection: List['SoquetT'],
        target: List['SoquetT'],
        qrom_targets: List['SoquetT'],
    ) -> Tuple[List['SoquetT'], List['SoquetT'], List['SoquetT'], List['SoquetT']]:
        ctrl, selection, qrom_targets = self._add_qrom_bloq(bb, ctrl, selection, qrom_targets)
        qrom_targets, target = self._add_cnot(bb, qrom_targets, target)
        ctrl, selection, qrom_targets = self._add_qrom_bloq(
            bb, ctrl, selection, qrom_targets, uncompute=True
        )
        if self.use_dirty_ancilla:
            qrom_targets, target = self._add_cnot(bb, qrom_targets, target)
        return ctrl, selection, target, qrom_targets

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        # Get the ctrl and target register for the SelectSwapQROM.
        ctrl = [soqs.pop(reg.name) for reg in self.control_registers]
        selection = [soqs.pop(reg.name) for reg in self.selection_registers]
        target = [soqs.pop(reg.name) for reg in self.target_registers]
        # Allocate intermediate clean/dirty ancilla for the underlying QROM call.
        if is_symbolic(*self.block_sizes):
            raise ValueError(
                f"Cannot decompose SelectSwapQROM bloq with symbolic block sizes. Found {self.block_sizes=}"
            )
        block_sizes = cast(Tuple[int, ...], self.block_sizes)
        qrom_targets = [
            _alloc_anc_for_reg(bb, reg.dtype, block_sizes, self.use_dirty_ancilla)
            for reg in self.target_registers
        ]
        # Verify some of the assumptions are correct.
        assert not soqs, f"All registers must have been used by now. Found: {soqs}"
        assert len(self.qrom_bloq.target_shapes) == len(self.target_registers)
        for qrom_target, my_target in zip(self.qrom_bloq.target_registers, self.target_registers):
            assert qrom_target.dtype == my_target.dtype
            assert qrom_target.shape == block_sizes
        # Add the bloq decomposition
        if any(b > 1 for b in block_sizes):
            ctrl, selection, target, qrom_targets = self._build_composite_bloq_with_swz(
                bb, ctrl, selection, target, qrom_targets
            )
        else:
            ctrl, selection, target, qrom_targets = self._build_composite_bloq_without_swz(
                bb, ctrl, selection, target, qrom_targets
            )
        # Free the allocated register.
        for reg in qrom_targets:
            assert isinstance(reg, np.ndarray)
            for soq in reg.flat:
                bb.free(soq)
        # Add back target and control registers.
        soqs |= {reg.name: soq for reg, soq in zip(self.control_registers, ctrl)}
        soqs |= {reg.name: soq for reg, soq in zip(self.selection_registers, selection)}
        soqs |= {reg.name: soq for reg, soq in zip(self.target_registers, target)}
        # Return dictionary of final soquets.
        return soqs

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


@bloq_example
def _qroam_multi_data() -> SelectSwapQROM:
    data1 = np.arange(5)
    data2 = np.arange(5) + 1
    qroam_multi_data = SelectSwapQROM.build_from_data(data1, data2)
    return qroam_multi_data


@bloq_example
def _qroam_multi_dim() -> SelectSwapQROM:
    data1 = np.arange(25).reshape((5, 5))
    data2 = (np.arange(25) + 1).reshape((5, 5))
    qroam_multi_dim = SelectSwapQROM.build_from_data(data1, data2)
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
