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
"""Advanced Quantum Read Only Memory."""
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np
from attrs import frozen
from numpy.typing import ArrayLike, NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    BoundedQUInt,
    QAny,
    Register,
    Signature,
    Soquet,
)
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.data_loading.qrom import _to_tuple
from qualtran.drawing import TextBox, WireSymbol
from qualtran.simulation.classical_sim import ClassicalValT

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def find_optimal_log_block_size(
    iteration_length: int, target_bitsize: int, adjoint: bool = False
) -> int:
    """Find optimal block size, which is a power of 2, for QROAM and the corresponding Toffoli cost.

    This functions returns the optimal `k` s.t.
        * k is in an integer and k >= 0.
        * iteration_length/2^k + target_bitsize*(2^k - 1) is minimized.
    The corresponding block size for SelectSwapQROM would be 2^k.

    Args:
        iteration_length: The amount of data to load for each data set (the array length).
        target_bitsize: The total bitsize of the target register(s).
        adjoint: Whether we are doing inverse qrom or not.

    Returns:
        k_opt: The optimal log block size.
        val_opt: The optimal toffoli cost with the block size.
    """
    if adjoint:
        k = 0.5 * np.log2(iteration_length)

        def value(kk: List[int]):
            return iteration_length / np.power(2, kk) + target_bitsize

    else:
        k = 0.5 * np.log2(iteration_length / target_bitsize)

        def value(kk: List[int]):
            return iteration_length / np.power(2, kk) + target_bitsize * (np.power(2, kk) - 1)

    if k < 0:
        return 1, np.ceil(value(2))
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = int(k_int[np.argmin(value(k_int))])  # obtain optimal k
    val_opt = int(np.ceil(value(2**k_opt)))
    return k_opt, val_opt


@cirq.value_equality()
@frozen
class QROAM(Bloq):
    r"""Advanced qroam i.e. QRO(A)M for loading data into a target register.

    Args:
        data: Sequence of integers to load in the target register. If more than one sequence
            is provided, each sequence must be of the same length. Each array must be one
            dimensional.
        target_bitsizes: Sequence of integers describing the size of target register for each
            data sequence to load.
        block_size(B): Load batches of `B` data elements in each iteration of traditional QROM
            (N/B iterations required). Complexity of QROAM scales as
            `O(B * b + N / B)`, where `B` is the block_size. Defaults to optimal value of
            `\sim sqrt(N / b)`.
        is_adjoint: Whether this bloq is daggered or not.

    Registers:
        control: Optional control registers
        selection: The selection registers which are iterated over when loading the data.
        target: The target registers for each data set.

    Raises:
        ValueError: If all target data sequences to load do not have the same length.
    """

    data: Sequence[NDArray] = attrs.field(converter=_to_tuple)
    target_bitsizes: Tuple[int, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    block_size: int = 1
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        assert self.block_size == 1, "Use QROM for block_size == 1"
        assert len(set(len(d) for d in self.data)) == 1
        assert len(self.target_bitsizes) == len(self.data)
        assert all(t >= int(max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data))
        assert 0 < self.block_size <= len(self.data[0])

    @classmethod
    def build(
        cls,
        *data: ArrayLike,
        target_bitsizes: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> 'QROAM':
        """Factory method to build a QROAM block from numpy arrays of input data.

        Args:
            data: Sequence of integers to load in the target register. If more than one sequence
                is provided, each sequence must be of the same length.
            target_bitsizes: Sequence of integers describing the size of target register for each
                data sequence to load. Defaults to `max(data[i]).bit_length()` for each i.
        """
        _data = [np.array(d, dtype=int) for d in data]
        if target_bitsizes is None:
            target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        if target_bitsizes is None:
            target_bitsizes = [int(max(d)).bit_length() for d in data]
        if block_size is None:
            # Figure out optimal value of block_size
            block_size = 2 ** find_optimal_log_block_size(len(_data[0]), sum(target_bitsizes))[0]
        return QROAM(data=_data, target_bitsizes=target_bitsizes, block_size=block_size)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        idx = vals['selection']
        selections = {'selection': idx}
        # Retrieve the data; bitwise add them in to the input target values
        targets = {f'target{d_i}_': d[idx] for d_i, d in enumerate(self.data)}
        targets = {k: v ^ vals[k] for k, v in targets.items()}
        return selections | targets

    def adjoint(self) -> 'Bloq':
        k_opt, _ = find_optimal_log_block_size(
            len(self.data[0]), sum(self.target_bitsizes), adjoint=not self.is_adjoint
        )
        return attrs.evolve(self, is_adjoint=not self.is_adjoint, block_size=2**k_opt)

    def pretty_name(self) -> str:
        dag = '†' if self.is_adjoint else ''
        return f"QROAM{dag}"

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        data_len = len(self.data[0])
        sel_bitsize = (data_len - 1).bit_length()
        return (Register('selection', BoundedQUInt(sel_bitsize, data_len)),)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        # See https://github.com/quantumlib/Qualtran/issues/556 for unusual placement of underscore.
        return tuple(
            Register(f'target{sequence_id}_', QAny(self.target_bitsizes[sequence_id]))
            for sequence_id in range(len(self.data))
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cost = find_optimal_log_block_size(
            len(self.data[0]), sum(self.target_bitsizes), adjoint=self.is_adjoint
        )[1]
        return {(Toffoli(), cost)}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        name = soq.reg.name
        if name == 'selection':
            return TextBox('In')
        elif 'target' in name:
            trg_indx = int(name.replace('target', '').replace('_', ''))
            # match the sel index
            subscript = chr(ord('a') + trg_indx)
            return TextBox(f'data_{subscript}')
        raise ValueError(f'Unknown register name {name}')

    def _value_equality_values_(self):
        data_tuple = tuple(tuple(d.flatten()) for d in self.data)
        return (self.selection_registers, self.target_registers, data_tuple)


@bloq_example
def _qroam_small() -> QROAM:
    data = np.arange(5)
    qrom_small = QROAM.build(data)
    return qrom_small


_QROAM_DOC = BloqDocSpec(
    bloq_cls=QROAM,
    import_line='from qualtran.bloqs.data_loading.qroam import QROAM',
    examples=[_qroam_small],
)
