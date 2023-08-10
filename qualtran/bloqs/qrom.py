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
from typing import Dict, Sequence, Tuple

import cirq
from attrs import frozen
from cirq_ft import QROM as CirqQROM
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
