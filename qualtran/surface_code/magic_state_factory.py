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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import LogicalErrorModel


class MagicStateFactory(metaclass=abc.ABCMeta):
    """Methods for modeling the costs of the magic state factories of a surface code compilation.

    An important consideration for a surface code compilation is how to execute arbitrary gates
    to run the desired algorithm. The surface code can execute Clifford gates in a fault-tolerant
    manner. Non-Clifford gates like the T gate, Toffoli or CCZ gate, or non-Clifford rotation
    gates require more expensive gadgets to implement. Executing a T or CCZ gate requires first
    using the technique of state distillation in an area of the computation called a "magic state
    factory" to distill a noisy T or CCZ state into a "magic state" of sufficiently low error.
    Such quantum states can be used to enact the non-Clifford quantum gate through gate
    teleportation.

    Magic state production is thought to be an important runtime and qubit-count bottleneck in
    foreseeable fault-tolerant quantum computers.

    This abstract interface specifies that each magic state factory must report its required
    number of physical qubits, the number of error correction cycles to produce enough magic
    states to enact a given number of logical gates and an error model, and the expected error
    associated with generating those magic states.
    """

    @abc.abstractmethod
    def n_physical_qubits(self) -> int:
        """The number of physical qubits used by the magic state factory."""

    @abc.abstractmethod
    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        """The number of cycles (time) required to produce the requested number of magic states."""

    @abc.abstractmethod
    def factory_error(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> float:
        """The total error expected from distilling magic states with a given physical error rate.

        This includes the cumulative effects of data-processing errors and distillation failures.
        """
