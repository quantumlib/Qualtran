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

import abc
import math
from typing import Optional

from attrs import field, frozen

import qualtran.surface_code.quantum_error_correction_scheme_summary as qec
from qualtran.surface_code.reference import Reference


class DataBlock(metaclass=abc.ABCMeta):
    """A cost model for the data block of a surface code compilation.

    A surface code layout is segregated into qubits dedicated to magic state distillation
    and others dedicated to storing the actual data being processed. The latter area is
    called the data block, and we provide its costs here.
    """

    @abc.abstractmethod
    def footprint(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block.

        Attributes:
            n_algo_qubits: The number of algorithm qubits whose data must be stored and
                accessed.
        """

    @abc.abstractmethod
    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""

    @abc.abstractmethod
    def n_cycles_to_consume_a_magic_state(self) -> int:
        """The worst case number of cycles needed to consume a magic state."""


@frozen
class SimpleDataBlock(DataBlock):
    """A simple data block that uses a fixed code distance and routing overhead.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.
        routing_overhead: As an approximation, assume some routing or auxiliary
            qubits proportional to the number of algorithm qubits.
        qec_scheme: Underlying quantum error correction scheme.
        reference: A description of the source of the model.
    """

    data_d: int
    routing_overhead: float = 0.5
    qec_scheme: qec.QuantumErrorCorrectionSchemeSummary = qec.FowlerSuperconductingQubits
    reference: Optional[Reference] = None

    def n_logical_qubits(self, n_algo_qubits: int) -> int:
        """Number of logical qubits including overhead.

        Note: the spreadsheet from the reference had a 50% overhead hardcoded for
        some of the cells using this quantity and variable (but set to 50% as default)
        for others.
        """
        return math.ceil((1 + self.routing_overhead) * n_algo_qubits)

    def footprint(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block."""
        n_phys_per_logical = self.qec_scheme.physical_qubits(self.data_d)
        return self.n_logical_qubits(n_algo_qubits) * n_phys_per_logical

    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""
        # spacetime_volue = number of data cells x number of cycles they will live for.
        spacetime_volume = self.n_logical_qubits(n_algo_qubits) * n_cycles
        return spacetime_volume * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.data_d
        )

    def n_cycles_to_consume_a_magic_state(self) -> int:
        return self.data_d


@frozen
class CompactDataBlock(SimpleDataBlock):
    r"""The compact data block uses a fixed code distance and routing overhead.

    The compact data block lays $n$ qubit batches in grid of shape (3, $n/2$) where
    the data batches are lined in the first and last row with the middle row being
    an ancilla region. This lowers the memory footprint of the block at the cost of an
    increased number of cycles to consume a magic state.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.
        qec_scheme: Underlying quantum error correction scheme.
        reference: A description of the source of the model.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892)
        page 7, figure 9
    """

    routing_overhead: float = field(default=0.5, init=False)
    reference: Reference = field(
        default=Reference(url='https://arxiv.org/abs/1808.02892', page=7), init=False
    )

    def n_cycles_to_consume_a_magic_state(self) -> int:
        return 9 * self.data_d


@frozen
class IntermediateDataBlock(SimpleDataBlock):
    r"""The intermediate data block uses a fixed code distance and routing overhead.

    The intermediate data block lays $n$ qubit batches in grid of shape (2, $2n+2$) where
    the data batches are lined in the first row with the second row being an ancilla region.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.
        qec_scheme: Underlying quantum error correction scheme.
        reference: A description of the source of the model.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892)
        page 9, figure 13a
    """

    routing_overhead: float = field(default=1.0, init=False)
    reference: Reference = field(
        default=Reference(url='https://arxiv.org/abs/1808.02892', page=8), init=False
    )

    def n_cycles_to_consume_a_magic_state(self) -> int:
        return 5 * self.data_d


@frozen
class FastDataBlock(DataBlock):
    r"""The fast data block uses a fixed code distance and a square layout.

    The fast data block lays $n$ qubit batches in a square grid of side length $1 + \sqrt{2n}$
    where the bottom row is an ancilla region and the top $\sqrt{2n}x\sqrt{2n}$ region is divided
    into alternating data and ancilla columns.
    The increased footprint is to be able to consume magic states in a single timestep.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.
        qec_scheme: Underlying quantum error correction scheme.
        reference: A description of the source of the model.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892)
        page 9, figure 13b
    """

    data_d: int
    qec_scheme: qec.QuantumErrorCorrectionSchemeSummary = qec.FowlerSuperconductingQubits
    reference: Reference = field(
        default=Reference(url='https://arxiv.org/abs/1808.02892', page=9), init=False
    )

    @staticmethod
    def grid_size(n_algo_qubits: int) -> int:
        return math.ceil(2 * n_algo_qubits + math.sqrt(8 * n_algo_qubits) + 1)

    def footprint(self, n_algo_qubits: int) -> int:
        return FastDataBlock.grid_size(n_algo_qubits)

    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""
        # spacetime_volue = number of data cells x number of cycles they will live for.
        spacetime_volume = self.n_logical_qubits(n_algo_qubits) * n_cycles
        return spacetime_volume * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.data_d
        )

    def n_logical_qubits(self, n_algo_qubits: int) -> int:
        return FastDataBlock.grid_size(n_algo_qubits)

    def n_cycles_to_consume_a_magic_state(self) -> int:
        return self.data_d

    @staticmethod
    def from_error_budget(
        error_budget: float,
        n_algo_qubits: int,
        qec_scheme: qec.QuantumErrorCorrectionSchemeSummary,
        physical_error_rate: float,
    ) -> 'FastDataBlock':
        q = FastDataBlock.grid_size(n_algo_qubits)
        d = qec_scheme.code_distance_from_budget(physical_error_rate, error_budget / q)
        return FastDataBlock(data_d=d, qec_scheme=qec_scheme)
