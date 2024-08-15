#  Copyright 2024 Google LLC
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
import numpy as np
import sympy
from numpy.typing import ArrayLike

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.data_loading.qrom_base import QROMBase
from qualtran.symbolics import ceil, is_symbolic, log2, prod, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import Bloq, BloqBuilder, SoquetT, QDType
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator

from qualtran.bloqs.data_loading.select_swap_qrom import _alloc_anc_for_reg, SelectSwapQROM


def _alloc_anc_for_reg_except_first(
    bb: 'BloqBuilder', dtype: 'QDType', shape: Tuple[int, ...], dirty: bool
) -> 'SoquetT':
    if not shape:
        return bb.allocate(dtype=dtype, dirty=dirty)
    soqs = np.empty(shape, dtype=object)
    ndindex_iter = np.ndindex(shape)
    _ = next(ndindex_iter)  # Skip the first element.
    for idx in ndindex_iter:
        soqs[idx] = bb.allocate(dtype=dtype, dirty=dirty)
    return soqs


def qroam_cost(x, data_size: SymbolicInt, bitsize: SymbolicInt, adjoint: bool = False):
    # See appendix B of https://arxiv.org/pdf/1902.02134
    if adjoint:
        return data_size / x + x
    else:
        return data_size / x + bitsize * (x - 1)


def get_optimal_log_block_size_clean_ancilla(
    data_size: SymbolicInt,
    bitsize: SymbolicInt,
    adjoint: bool = False,
    qroam_block_size: Optional[SymbolicInt] = None,
) -> SymbolicInt:
    if qroam_block_size is None:
        if adjoint:
            k: SymbolicFloat = 0.5 * log2(data_size)
        else:
            k = 0.5 * log2(data_size / bitsize)
    else:
        k = log2(qroam_block_size)
    if is_symbolic(k):
        return k
    k_int = np.array([np.floor(k), np.ceil(k)])
    return int(k_int[np.argmin(qroam_cost(2**k_int, data_size, bitsize, adjoint))])


@attrs.frozen
class QROAMCleanAdjoint(QROMBase, GateWithRegisters):  # type: ignore[misc]
    log_block_sizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )

    @cached_property
    def _target_reg_side(self) -> Side:
        return Side.LEFT

    @classmethod
    def build_from_data(
        cls: Type['QROAMCleanAdjoint'],
        *data: ArrayLike,
        target_bitsizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
    ) -> 'QROAMCleanAdjoint':
        qroam: 'QROAMCleanAdjoint' = cls._build_from_data(
            *data,
            target_bitsizes=target_bitsizes,
            target_shapes=target_shapes,
            num_controls=num_controls,
        )
        return qroam.with_log_block_sizes(log_block_sizes)

    @classmethod
    def build_from_bitsize(
        cls: Type['QROAMCleanAdjoint'],
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
    ) -> 'QROAMCleanAdjoint':
        qroam: 'QROAMCleanAdjoint' = cls._build_from_bitsize(
            data_len_or_shape,
            target_bitsizes,
            selection_bitsizes=selection_bitsizes,
            target_shapes=target_shapes,
            num_controls=num_controls,
        )
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    @log_block_sizes.default
    def _default_log_block_sizes(self) -> Tuple[SymbolicInt, ...]:
        target_bitsize = sum(self.target_bitsizes) * sum(
            prod(shape) for shape in self.target_shapes
        )
        return tuple(
            get_optimal_log_block_size_clean_ancilla(ilen, target_bitsize, adjoint=True)
            for ilen in self.data_shape
        )

    def with_log_block_sizes(
        self, log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None
    ) -> 'QROAMCleanAdjoint':
        if log_block_sizes is None:
            return self
        if isinstance(log_block_sizes, (int, sympy.Basic, numbers.Number)):
            log_block_sizes = (log_block_sizes,)
        if not is_symbolic(*log_block_sizes):
            assert all(1 <= 2**bs <= ilen for bs, ilen in zip(log_block_sizes, self.data_shape))
        return attrs.evolve(self, log_block_sizes=log_block_sizes)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        X = prod([2**k for k in self.log_block_sizes])
        N = prod(self.data_shape)
        B = 2
        cost = qroam_cost(X, ceil(N / 2), B, adjoint=True)
        return {(Toffoli(), cost)}

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def adjoint(self) -> 'QROAMClean':
        return QROAMClean(**attrs.asdict(self))


@attrs.frozen
class QROAMClean(SelectSwapQROM):
    log_block_sizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    use_dirty_ancilla: bool = attrs.field(init=False, default=False, repr=False)

    @cached_property
    def _target_reg_side(self) -> Side:
        return Side.RIGHT

    @log_block_sizes.default
    def _default_log_block_sizes(self) -> Tuple[SymbolicInt, ...]:
        target_bitsize = sum(self.target_bitsizes) * sum(
            prod(shape) for shape in self.target_shapes
        )
        return tuple(
            get_optimal_log_block_size_clean_ancilla(ilen, target_bitsize)
            for ilen in self.data_shape
        )

    @classmethod
    def build_from_data(
        cls: Type['QROAMClean'],
        *data: ArrayLike,
        target_bitsizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
    ) -> 'QROAMClean':
        qroam: 'QROAMClean' = cls._build_from_data(
            *data, target_bitsizes=target_bitsizes, num_controls=num_controls
        )
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    @classmethod
    def build_from_bitsize(
        cls: Type['QROAMClean'],
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
        log_block_sizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
    ) -> 'QROAMClean':
        qroam: 'QROAMClean' = cls._build_from_bitsize(
            data_len_or_shape,
            target_bitsizes,
            selection_bitsizes=selection_bitsizes,
            num_controls=num_controls,
        )
        return qroam.with_log_block_sizes(log_block_sizes=log_block_sizes)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.target_registers,
                *self.junk_registers,
            ]
        )

    @cached_property
    def batched_data_permuted(self) -> List[np.ndarray]:
        if is_symbolic(*self.block_sizes):
            raise ValueError(
                f"Cannot decompose SelectSwapQROM bloq with symbolic block sizes. Found {self.block_sizes=}"
            )
        block_sizes = cast(Tuple[int, ...], self.block_sizes)
        ret = []
        for data, swz in zip(self.batched_data, self.swap_with_zero_bloqs):
            permuted_batched_data = np.zeros(data.shape + block_sizes, dtype=data.dtype)
            for sel_l in np.ndindex(cast(Tuple[int, ...], self.batched_qrom_shape)):
                for sel_k in np.ndindex(block_sizes):
                    sel_kwargs = {reg.name: sel for reg, sel in zip(swz.selection_registers, sel_k)}
                    curr_data = swz.call_classically(**sel_kwargs, targets=np.copy(data[sel_l]))[-1]
                    idx = (*sel_l, *sel_k)
                    permuted_batched_data[idx][:] = curr_data

            n_blocks = len(self.block_sizes)
            transpose_axes = [x for i in range(n_blocks) for x in [i, i + n_blocks]]
            transpose_axes += [i + 2 * n_blocks for i in range(n_blocks)]
            desired_shape = tuple(
                NbyK * K for NbyK, K in zip(self.batched_qrom_shape, self.block_sizes)
            )
            ret.append(
                permuted_batched_data.transpose(transpose_axes).reshape(
                    desired_shape + self.block_sizes
                )
            )
        return ret

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        # The newly allocated registers should be kept around for measurement based uncomputation.
        junk_regs = []
        block_size = prod(self.block_sizes)
        for reg in self.target_registers:
            assert reg.shape == ()
            if is_symbolic(block_size) or block_size > 1:
                junk_regs += [attrs.evolve(reg, name='junk_' + reg.name, shape=(block_size - 1,))]
        return tuple(junk_regs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        ret: Dict[Bloq, SymbolicInt] = defaultdict(lambda: 0)
        ret[self.qrom_bloq] += 1
        for swz in self.swap_with_zero_bloqs:
            if any(is_symbolic(s) or s > 0 for s in swz.selection_bitsizes):
                ret[swz] += 1
        return set(ret.items())

    def _build_composite_bloq_with_swz_clean(
        self,
        bb: 'BloqBuilder',
        ctrl: List['SoquetT'],
        selection: List['SoquetT'],
        qrom_targets: List['SoquetT'],
    ) -> Tuple[List['SoquetT'], List['SoquetT'], List['SoquetT']]:
        sel_l, sel_k = self._partition_sel_register(bb, selection)
        ctrl, sel_l, qrom_targets = self._add_qrom_bloq(bb, ctrl, sel_l, qrom_targets)
        sel_k, qrom_targets = self._add_swap_with_zero_bloq(bb, sel_k, qrom_targets)
        selection = self._unpartition_sel_register(bb, sel_l, sel_k)
        return ctrl, selection, qrom_targets

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        # Get the ctrl and target register for the SelectSwapQROM.
        ctrl = [soqs.pop(reg.name) for reg in self.control_registers]
        selection = [soqs.pop(reg.name) for reg in self.selection_registers]
        if is_symbolic(*self.block_sizes):
            raise ValueError(
                f"Cannot decompose QROAM bloq with symbolic block sizes. Found {self.block_sizes=}"
            )
        block_sizes = cast(Tuple[int, ...], self.block_sizes)
        # Allocate intermediate clean/dirty ancilla for the underlying QROM call.
        qrom_targets = []
        for reg in self.target_registers:
            qrom_target = _alloc_anc_for_reg_except_first(
                bb, reg.dtype, block_sizes, self.use_dirty_ancilla
            )
            qrom_target[np.unravel_index(0, block_sizes)] = _alloc_anc_for_reg(  # type: ignore[index]
                bb, reg.dtype, reg.shape, dirty=False
            )
            qrom_targets.append(qrom_target)
        # Assert that all registers have been used by now.
        assert not soqs, f"All registers must have been used by now. Found: {soqs}"
        # Add the bloq decomposition
        if any(b > 1 for b in block_sizes):
            ctrl, selection, qrom_targets = self._build_composite_bloq_with_swz_clean(
                bb, ctrl, selection, qrom_targets
            )
        else:
            ctrl, selection, qrom_targets = self._add_qrom_bloq(bb, ctrl, selection, qrom_targets)
        # Construct and return dictionary of final soquets.
        soqs |= {reg.name: soq for reg, soq in zip(self.control_registers, ctrl)}
        soqs |= {reg.name: soq for reg, soq in zip(self.selection_registers, selection)}
        soqs |= {reg.name: soq.flat[1:] for reg, soq in zip(self.junk_registers, qrom_targets)}  # type: ignore[union-attr]
        soqs |= {reg.name: soq.flat[0] for reg, soq in zip(self.target_registers, qrom_targets)}  # type: ignore[union-attr]
        return soqs

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Dict[str, 'ClassicalValT']:
        vals_without_junk = super().on_classical_vals(**vals)
        selection = cast(Tuple[int, ...], tuple(vals[reg.name] for reg in self.selection_registers))
        for d, junk_reg in zip(self.batched_data_permuted, self.junk_registers):
            vals_without_junk[junk_reg.name] = d[selection].flat[1:]
        return vals_without_junk

    def adjoint(self) -> 'QROAMCleanAdjoint':
        if self.has_data():
            return QROAMCleanAdjoint.build_from_data(
                *self.batched_data_permuted,
                target_shapes=(self.block_sizes,) * len(self.batched_data_permuted),
            )
        else:
            return QROAMCleanAdjoint.build_from_bitsize(
                self.data_shape,
                target_bitsizes=self.target_bitsizes,
                target_shapes=(self.block_sizes,) * len(self.target_bitsizes),
            )


@bloq_example
def _qroam_clean_multi_data() -> QROAMClean:
    data1 = np.arange(5, dtype=int)
    data2 = np.arange(5, dtype=int) + 1
    qroam_multi_data = QROAMClean.build_from_data(data1, data2, log_block_sizes=(2,))
    return qroam_multi_data


@bloq_example
def _qroam_clean_multi_dim() -> QROAMClean:
    data1 = np.arange(25, dtype=int).reshape((5, 5))
    data2 = (np.arange(25, dtype=int) + 1).reshape((5, 5))
    qroam_multi_dim = QROAMClean.build_from_data(data1, data2, log_block_sizes=(2, 1))
    return qroam_multi_dim


@bloq_example
def _qroam_clean_symb_1d() -> QROAMClean:
    N, b, k, c = sympy.symbols('N b k c')
    qroam_symb_clean_1d = QROAMClean.build_from_bitsize(
        (N,), (b,), log_block_sizes=(k,), num_controls=c
    )
    return qroam_symb_clean_1d


@bloq_example
def _qroam_clean_symb_2d() -> QROAMClean:
    N, M, b1, b2, k1, k2, c = sympy.symbols('N M b1 b2 k1 k2 c')
    log_block_sizes = (k1, k2)
    qroam_symb_clean_2d = QROAMClean.build_from_bitsize(
        (N, M), (b1, b2), log_block_sizes=log_block_sizes, num_controls=c
    )
    return qroam_symb_clean_2d


_QROAM_CLEAN_DOC = BloqDocSpec(
    bloq_cls=QROAMClean,
    import_line='from qualtran.bloqs.data_loading.select_swap_qrom import QROAMClean',
    examples=[
        _qroam_clean_multi_data,
        _qroam_clean_multi_dim,
        _qroam_clean_symb_1d,
        _qroam_clean_symb_2d,
    ],
)
