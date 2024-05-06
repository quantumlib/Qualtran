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
"""Common Chemistry bloqs which have costs that differ from those currently implemented in qualtran.

These are for temporary convenience to lock-in the quoted literature costs.
"""
from functools import cached_property
from typing import Optional, Set, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np
from attrs import field, frozen

from qualtran import Bloq, BloqBuilder, QAny, QBit, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.drawing import Circle, TextBox, WireSymbol

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def get_qroam_cost(
    data_size: int, bitsize: int, adjoint: bool = False, qroam_block_size: Optional[int] = None
) -> int:
    """This gives the optimal k and minimum cost for a QROM over L values of size M.

    Adapted from openfermion and accounts for quoted inverse cost.

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.
        adjoint: whether to get costs from inverse qrom (true) or not (false).
        qroam_block_size: The block size for QROAM. Default find the optimal
            value given the data size.

    Returns:
       val_opt: minimal (optimal) cost of QROM
    """
    if qroam_block_size == 1:
        return data_size - 1
    if adjoint:
        if qroam_block_size is None:
            log_blk = 0.5 * np.log2(data_size)
            qroam_block_size = 2**log_blk
        value = lambda x: data_size / x + x
    else:
        if qroam_block_size is None:
            log_blk = 0.5 * np.log2(data_size / bitsize)
            assert log_blk >= 0
            qroam_block_size = 2**log_blk
        value = lambda x: data_size / x + bitsize * (x - 1)
    k = np.log2(qroam_block_size)
    k_int = np.array([np.floor(k), np.ceil(k)])
    k_opt = k_int[np.argmin(value(2**k_int))]
    val_opt = np.ceil(value(2**k_opt))
    return int(val_opt)


@frozen
class QROAM(Bloq):
    """Advanced qroam i.e. QRO(A)M for loading data into a target register.

    Args:
        data: Sequence of integers to load in the target register. If more than one sequence
            is provided, each sequence must be of the same length.
        target_bitsizes: Sequence of integers describing the size of target register for each
            data sequence to load. Defaults to `max(data[i]).bit_length()` for each i.
        block_size(B): Load batches of `B` data elements in each iteration of traditional QROM
            (N/B iterations required). Complexity of QROAM scales as
            `O(B * b + N / B)`, where `B` is the block_size. Defaults to optimal value of
                `O(sqrt(N / b))`.

    Registers:
        control: Optional control registers
        selection: The selection registers which are iterated over when loading the data.
        target: The target registers for each data set.

    Raises:
        ValueError: If all target data sequences to load do not have the same length.
    """

    data_size: int
    target_bitsize: int
    is_adjoint: bool = False
    qroam_block_size: Optional[int] = None

    def pretty_name(self) -> str:
        dag = '†' if self.is_adjoint else ''
        return f"QROAM{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(sel=(self.data_size - 1).bit_length(), trg=self.target_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cost = get_qroam_cost(
            self.data_size,
            self.target_bitsize,
            adjoint=self.is_adjoint,
            qroam_block_size=self.qroam_block_size,
        )
        return {(Toffoli(), cost)}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)
