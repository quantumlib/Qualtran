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

from attrs import frozen

import qualtran.surface_code.quantum_error_correction_scheme_summary as qec


class DataBlock(metaclass=abc.ABCMeta):
    """A cost model for the data block of a surface code compilation.

    A surface code layout is segregated into qubits dedicated to magic state distillation
    and others dedicated to storing the actual data being processed. The latter area is
    called the data block, and we provide its costs here.
    """

    @abc.abstractmethod
    def footprint(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block.

        Args:
            n_algo_qubits: The number of algorithm qubits whose data must be stored and
                accessed.
        """

    @abc.abstractmethod
    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""


@frozen
class SimpleDataBlock(DataBlock):
    """A simple data block that uses a fixed code distance and routing overhead.

    Args:
        data_d: The code distance `d` for protecting the qubits in the data block.
        routing_overhead: As an approximation, assume a number of routing or auxiliary
            qubits proportional to the number of algorithm qubits.
    """

    data_d: int
    routing_overhead: float = 0.5
    qec_scheme: qec.QuantumErrorCorrectionSchemeSummary = qec.FowlerSuperconductingQubits

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
        data_cells = self.n_logical_qubits(n_algo_qubits) * n_cycles
        return data_cells * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.data_d
        )
