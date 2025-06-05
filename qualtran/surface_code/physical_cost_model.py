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
from functools import cached_property, lru_cache
from typing import Tuple, TYPE_CHECKING

from attrs import frozen

from .qec_scheme import LogicalErrorModel

if TYPE_CHECKING:
    from qualtran.surface_code import (
        AlgorithmSummary,
        DataBlock,
        MagicStateFactory,
        PhysicalParameters,
        QECScheme,
    )


@frozen
class PhysicalCostModel:
    """A model for estimating physical costs from algorithm counts.

    The model is parameterized by 1) properties of the target hardware architecture encapsulated
    in the data class `PhysicalParameters`, and 2) Execution protocol design choices.

    We further factor the design choices into a) the data block design for storing
    algorithm qubits, b) the magic state factory construction, and c) the error suppression
    ability of the code.

    Each method for computing physical costs take `AlgorithmSummary` inputs: the number of
    algorithm qubits and the number of algorithm gates. Output quantities
    include the wall-clock time, the number of physical qubits, and the probability of failure
    due to the physical realization of the algorithm.

    ### Time costs

    The amount of time to run an algorithm is modeled as the greater of two quantities:
    The number of cycles required to generate enough magic states (via the `factory`), and
    the number of cycles required to consume the magic states (via the `data_block`). The model
    assumes that the rate of magic state generation is slower than the reaction limit. Each
    cycle takes a fixed amount of wall-clock time, given by `architecture`.

    ### Space costs

    The number of physical qubits is the sum of the number of factory qubits and data block qubits.

    ### Error

    We assume the constituent error probabilities are sufficiently low to permit a first-order
    approximation for combining sources of error. The total error is the sum of error probabilities
    due to magic state production (via `factory`) and data errors (via `data_block`). Note that
    the total error in data storage depends on the number of cycles, which depends on the
    factory design.

    Args:
        physical_params: The physical parameters of the target hardware
        data_block: The design of the data block
        factory: The construction of the magic state factory/ies
        qec_scheme: The scheme used to suppress errors.
    """

    physical_params: 'PhysicalParameters'
    data_block: 'DataBlock'
    factory: 'MagicStateFactory'
    qec_scheme: 'QECScheme'

    @cached_property
    def logical_error_model(self):
        """The QEC scheme with a physical error rate defines the logical error model.

        A logical error model is a callable that returns the logical error rate expected (per cycle)
        for a surface code of distance $d$.
        """
        return LogicalErrorModel(
            physical_error=self.physical_params.physical_error, qec_scheme=self.qec_scheme
        )

    @lru_cache
    def _get_physical_cost_base_quantities(self, algo_summary: 'AlgorithmSummary'):
        # Time
        n_generation_cycles = self.factory.n_cycles(
            n_logical_gates=algo_summary.n_logical_gates,
            logical_error_model=self.logical_error_model,
        )
        n_consumption_cycles = self.data_block.n_cycles(
            n_logical_gates=algo_summary.n_logical_gates,
            logical_error_model=self.logical_error_model,
        )
        n_cycles = int(max(n_generation_cycles, n_consumption_cycles))

        # Space
        n_factory_phys_q = self.factory.n_physical_qubits()
        n_data_phys_q = self.data_block.n_physical_qubits(n_algo_qubits=algo_summary.n_algo_qubits)
        n_phys_q = n_factory_phys_q + n_data_phys_q

        # Error (depends on time)
        factory_error = self.factory.factory_error(
            n_logical_gates=algo_summary.n_logical_gates,
            logical_error_model=self.logical_error_model,
        )
        data_error = self.data_block.data_error(
            n_algo_qubits=algo_summary.n_algo_qubits,
            n_cycles=int(n_cycles),
            logical_error_model=self.logical_error_model,
        )
        error = factory_error + data_error

        return n_cycles, n_phys_q, error

    def n_cycles(self, algo_summary: 'AlgorithmSummary') -> int:
        """The number of error correction cycles required to execute the algorithm."""
        n_cycles, n_phys_q, error = self._get_physical_cost_base_quantities(algo_summary)
        return n_cycles

    def duration_hr(self, algo_summary: 'AlgorithmSummary'):
        """The duration in hours required to execute the algorithm."""
        n_cycles = self.n_cycles(algo_summary)
        cycle_time_us = self.physical_params.cycle_time_us
        duration_hr = (cycle_time_us * n_cycles) / (1_000_000 * 60 * 60)
        return duration_hr

    def n_phys_qubits(self, algo_summary: 'AlgorithmSummary') -> int:
        """The number of physical qubits required to execute the algorithm"""
        n_cycles, n_phys_q, error = self._get_physical_cost_base_quantities(algo_summary)
        return n_phys_q

    def error(self, algo_summary: 'AlgorithmSummary') -> float:
        """The total error rate of executing the algorithm."""
        n_cycles, n_phys_q, error = self._get_physical_cost_base_quantities(algo_summary)
        return error

    @classmethod
    def make_gidney_fowler(cls, data_d: int):
        from qualtran.surface_code import (
            CCZ2TFactory,
            PhysicalParameters,
            QECScheme,
            SimpleDataBlock,
        )

        return cls(
            physical_params=PhysicalParameters.make_gidney_fowler(),
            data_block=SimpleDataBlock(data_d=data_d),
            factory=CCZ2TFactory(),
            qec_scheme=QECScheme.make_gidney_fowler(),
        )

    @classmethod
    def make_beverland_et_al(
        cls, data_d: int, data_block_name: str = 'compact', factory_ds: Tuple = (9, 3, 3)
    ):
        from qualtran.surface_code import (
            CompactDataBlock,
            FastDataBlock,
            FifteenToOne,
            IntermediateDataBlock,
            PhysicalParameters,
            QECScheme,
        )

        data_block: DataBlock
        if data_block_name == 'fast':
            data_block = FastDataBlock(data_d=data_d)
        elif data_block_name == 'compact':
            data_block = CompactDataBlock(data_d=data_d)
        elif data_block_name == 'intermediate':
            data_block = IntermediateDataBlock(data_d=data_d)
        else:
            raise ValueError(f"Unknown data block '{data_block_name}'")

        return cls(
            physical_params=PhysicalParameters.make_beverland_et_al(),
            data_block=data_block,
            factory=FifteenToOne(*factory_ds),
            qec_scheme=QECScheme.make_beverland_et_al(),
        )
