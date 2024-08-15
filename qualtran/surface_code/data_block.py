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
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import LogicalErrorModel, QECScheme


class DataBlock(metaclass=abc.ABCMeta):
    """Methods for modeling the costs of the data block of a surface code compilation.

    The number of algorithm qubits is reported by Qualtran as a logical cost of a bloq. The
    surface code is a rate-1 code, so each bit of data needs at least one surface code tile. Due
    to locality constraints imposed by the 2D surface code combined with the need to interact
    qubits that aren’t necessarily local, additional tiles are needed to actually execute a program.

    Each data block is responsible for reporting the number of tiles required to store a certain
    number of algorithm qubits; as well as the number of time steps required to consume a magic
    state. Different data blocks exist in the literature, and data block provides a different
    space-time tradeoff.

    The space occupied by the data block is to be contrasted with the space used for magic
    state distillation.
    """

    @property
    @abc.abstractmethod
    def data_d(self):
        """The code distance used to store the data in the data block."""

    @abc.abstractmethod
    def n_tiles(self, n_algo_qubits: int) -> int:
        """The number of surface code tiles used to store a given number of algorithm qubits.

         We define an “algorithm qubit” to be a qubit used in the routing of algorithm-relevant
         quantum data in a bloq. A physical qubit is a physical system that can encode one qubit,
         albeit noisily. Specific to the surface code, we define a “tile” to be the minimal area
         of physical qubits necessary to encode one logical qubit to a particular code distance.
         A tile can store an algorithm qubit, can be used for ancillary purposes like routing,
         or can be left idle. A tile is usually a square grid of $2d^2$ physical qubits.

         DataBlock implementations must override this method. This method is used by
         `self.n_phys_qubits` to report the total number of physical qubits.

        Args:
            n_algo_qubits: The number of algorithm qubits to compute the number of tiles for.

        Returns:
            The number of tiles used by this data block to store the given number of algorithm
            qubits.
        """

    @property
    @abc.abstractmethod
    def n_steps_to_consume_a_magic_state(self):
        """The number of surface code steps to consume a magic state.

        We must teleport in "magic states" to do non-Clifford operations on our algorithmic
        data qubits. The layout of the data block can limit the number magic states consumed
        per unit time.

        One surface code step is `data_d` cycles of error correction.

        DataBlock imlpementation must override this method. This method is used by
        `self.n_cycles` to report the total number of cycles required.
        """

    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        """The number of surface code cycles to apply the number of gates to the data block.

        Note that only the Litinski (2019) derived data blocks model a limit on the number of
        magic states consumed per step. Other data blocks return "zero" for the number of cycles
        due to the data block. When using those data block designs, it is assumed that the
        number of cycles taken by the magic state factories is the limiting factor in the
        computation.
        """
        counts = n_logical_gates.total_t_and_ccz_count()
        n_steps = (counts['n_t'] + counts['n_ccz']) * self.n_steps_to_consume_a_magic_state
        n_cycles = self.data_d * n_steps
        return n_cycles

    def n_physical_qubits(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block."""
        n_phys_per_tile = 2 * self.data_d**2
        return n_phys_per_tile * self.n_tiles(n_algo_qubits)

    def data_error(
        self, n_algo_qubits: int, n_cycles: int, logical_error_model: 'LogicalErrorModel'
    ) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""
        # spacetime_volue = number of data cells x number of cycles they will live for.
        spacetime_volume = self.n_tiles(n_algo_qubits) * n_cycles
        return spacetime_volume * logical_error_model(self.data_d)


@frozen
class SimpleDataBlock(DataBlock):
    """A simple data block that uses a fixed code distance and routing overhead.

    The simple data block approximates the total tile usage by considering one tile
    per algorithm qubit plus a constant factor overhead presumed to be used for routing.

    Note: the spreadsheet from the reference had a 50% overhead hardcoded for
    some of the cells using this quantity and variable (but set to 50% as default)
    for others.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.
        routing_overhead: As an approximation, assume some routing or auxiliary
            qubits proportional to the number of algorithm qubits.

    References:
        Efficient magic state factories with a catalyzed |CCZ> to 2|T> transformation.
        https://arxiv.org/abs/1812.01238
    """

    data_d: int
    routing_overhead: float = 0.5

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil((1 + self.routing_overhead) * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        # The simple data block assume that an unbounded number of magic states can be
        # processed simultaneously and that the computation is bounded by the time to
        # produce the magic states.
        return 0


@frozen
class CompactDataBlock(DataBlock):
    r"""The compact data block uses a fixed code distance and one, long access hallway.

    The compact data block lays $n$ qubit batches in grid of shape (3, $n/2$) where
    the data batches are lined in the first and last row with the middle row being
    an ancilla region. This lowers the space footprint of the block at the cost of an
    increased number of cycles to consume a magic state.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892).
        Litinski (2019). Page 7, figure 9
    """

    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(1.5 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 9


@frozen
class IntermediateDataBlock(DataBlock):
    r"""The intermediate data block uses a fixed code distance and routing overhead.

    The intermediate data block lays $n$ qubit batches in grid of shape (2, $2n+2$) where
    the data batches are lined in the first row with the second row being an ancilla region.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892).
        Litinski (2019). Page 9, figure 13a
    """

    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(2 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 5


@frozen
class FastDataBlock(DataBlock):
    r"""The fast data block uses a fixed code distance and a square layout.

    The fast data block lays $n$ qubit batches in a square grid of side length $1 + \sqrt{2n}$
    where the bottom row is an ancilla region and the top $\sqrt{2n}x\sqrt{2n}$ region is divided
    into alternating data and ancilla columns.

    The increased footprint is to be able to consume magic states in a single timestep.

    Attributes:
        data_d: The code distance `d` for protecting the qubits in the data block.

    References:
        [A Game of Surface Codes](https://arxiv.org/abs/1808.02892).
        Litinski (2019). Page 9, figure 13b
    """

    data_d: int

    @staticmethod
    def get_n_tiles(n_algo_qubits: int):
        # This static method can be used in contexts where we want to know the number
        # of tiles independent of `self.data_d`.
        return math.ceil(2 * n_algo_qubits + math.sqrt(8 * n_algo_qubits) + 1)

    def n_tiles(self, n_algo_qubits: int) -> int:
        return self.get_n_tiles(n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 1

    @classmethod
    def from_error_budget(
        cls, error_budget: float, n_algo_qubits: int, qec_scheme: 'QECScheme', phys_err_rate: float
    ) -> 'FastDataBlock':
        q = FastDataBlock.get_n_tiles(n_algo_qubits)
        d = qec_scheme.code_distance_from_budget(phys_err_rate, error_budget / q)
        return cls(data_d=d)
