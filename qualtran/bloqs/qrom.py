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
from typing import Dict, Optional, Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen
from cirq_ft import QROM as CirqQROM
from cirq_ft import SelectSwapQROM as CirqSelectSwapQROM
from cirq_ft.algos.select_swap_qrom import find_optimal_log_block_size
from numpy.typing import NDArray

from qualtran import Bloq, CompositeBloq, Register, Signature
from qualtran.cirq_interop import CirqQuregT, decompose_from_cirq_op


@frozen
class QROM(Bloq):
    """Gate to load data[l] in the target register when the selection stores an index l.

    In the case of multi-dimensional data[p,q,r,...] we use multiple named
    selection registers [selection0, selection1, selection2, ...] to index and
    load the data.

    Args:
        data: List of numpy ndarrays specifying the data to load. If the length
            of this list is greater than one then we use the same selection indices
            to load each dataset (for example, to load alt and keep data for
            state preparation). Each data set is required to have the same
            shape and to be of integer type.
        selection_bitsizes: The number of bits used to represent each selection register
            corresponding to the size of each dimension of the array. Should be
            the same length as the shape of each of the datasets.
        data_bitsizes: The number of bits used to represent the data
            registers. This can be deduced from the maximum element of each of the
            datasets. Should be of length len(data), i.e. the number of datasets.
        num_controls: The number of controls registers.
    """

    data: Sequence[NDArray]
    selection_bitsizes: Tuple[int, ...]
    data_bitsizes: Tuple[int, ...]
    num_controls: int = 0

    @cached_property
    def signature(self) -> Signature:
        regs = [
            Register(f"selection{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)
        ]
        regs += [Register(f"target{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        if self.num_controls > 0:
            regs += [Register("control", bitsize=self.num_controls)]
        return Signature(regs)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        qrom = CirqQROM(
            data=self.data,
            selection_bitsizes=self.selection_bitsizes,
            target_bitsizes=self.data_bitsizes,
            num_controls=self.num_controls,
        )
        return (qrom.on_registers(**cirq_quregs), cirq_quregs)

    def __hash__(self):
        # This is not a great hash. No guarantees.
        # See: https://github.com/quantumlib/Qualtran/issues/339
        return hash(self.signature)

    def __eq__(self, other) -> bool:
        return self.signature == other.signature

    def __ne__(self, other) -> bool:
        return self.signature != other.signature


@frozen
class SelectSwapQROM(Bloq):
    """Gate to load data[l] in the target register when the selection stores an index l.

    In contrast to the QROM bloq, SelectSwapQROM only accepts one-dimensional
    data. See cirq_ft.select_swap_qrom for futher implmentation details.

    The final T-complexity of `SelectSwapQROM` is `O(B * b + N / B)`
    T-gates; where `B` is the block-size with an optimal value of `O(sqrt(N /
    b))`.

    This improvement in T-complexity is achieved at the cost of using an additional `O(B * b)`
    ancilla qubits, with a nice property that these additional ancillas can be `dirty`; i.e.
    they don't need to start in the |0> state and thus can be borrowed from other parts of the
    algorithm. The state of these dirty ancillas would be unaffected after the operation has
    finished.

    For more details, see the reference below:

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).  Low, Kliuchnikov, Schaeffer. 2018.
    """

    data: Tuple[Tuple[int, ...], ...]
    target_bitsizes: Sequence[int]
    selection_bitsize: int
    block_size: int
    num_blocks: int
    num_sequences: int

    @classmethod
    def build(
        cls,
        *data: Sequence[int],
        target_bitsizes: Optional[Sequence[int]] = None,
        block_size: Optional[int] = None,
    ) -> 'SelectSwapQROM':
        """Factory method to build SelectSwapQROM instance.

        For a single data sequence of length `N`, maximum target bitsize `b` and block size `B`;
        SelectSwapQROM requires:
            - Selection register & ancilla of size `logN` for QROM data load.
            - 1 clean target register of size `b`.
            - `B` dirty target registers, each of size `b`.

        Similarly, to load `M` such data sequences, `SelectSwapQROM` requires:
            - Selection register & ancilla of size `logN` for QROM data load.
            - 1 clean target register of size `sum(target_bitsizes)`.
            - `B` dirty target registers, each of size `sum(target_bitsizes)`.

        Args:
            data: Sequence of integers to load in the target register. If more than one sequence
                is provided, each sequence must be of the same length.
            target_bitsizes: Sequence of integers describing the size of target register for each
                data sequence to load. Defaults to `max(data[i]).bit_length()` for each i.
            block_size(B): Load batches of `B` data elements in each iteration of traditional QROM
                (N/B iterations required). Complexity of SelectSwap QROAM scales as
                `O(B * b + N / B)`, where `B` is the block_size. Defaults to optimal value of
                 `O(sqrt(N / b))`.

        Raises:
            ValueError: If all target data sequences to load do not have the same length.
        """
        # Validate input.
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        if target_bitsizes is None:
            target_bitsizes = [max(d).bit_length() for d in data]
        assert len(target_bitsizes) == len(data)
        assert all(t >= max(d).bit_length() for t, d in zip(target_bitsizes, data))
        num_sequences = len(data)
        target_bitsizes = target_bitsizes
        iteration_length = len(data[0])
        if block_size is None:
            # Figure out optimal value of block_size
            block_size = 2 ** find_optimal_log_block_size(iteration_length, sum(target_bitsizes))
        assert 0 < block_size <= iteration_length
        block_size = block_size
        num_blocks = int(np.ceil(iteration_length / block_size))
        selection_q, selection_r = tuple((L - 1).bit_length() for L in [num_blocks, block_size])
        data = tuple(tuple(d) for d in data)
        return SelectSwapQROM(
            data, target_bitsizes, selection_q + selection_r, block_size, num_blocks, num_sequences
        )

    @cached_property
    def signature(self) -> Signature:
        regs = [Register("selection", bitsize=self.selection_bitsize)]
        regs += [Register(f"target{i}", bitsize=bs) for i, bs in enumerate(self.target_bitsizes)]
        return Signature(regs)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        qrom = CirqSelectSwapQROM(
            *self.data, target_bitsizes=tuple(self.target_bitsizes), block_size=self.block_size
        )
        return (qrom.on_registers(**cirq_quregs), cirq_quregs)
