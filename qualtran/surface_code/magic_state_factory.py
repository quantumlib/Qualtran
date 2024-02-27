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

from qualtran.surface_code.magic_count import MagicCount


class MagicStateFactory(metaclass=abc.ABCMeta):
    """A cost model for the magic state distillation factory of a surface code compilation.

    A surface code layout is segregated into qubits dedicated to magic state distillation
    and storing the data being processed. The former area is called the magic state distillation
    factory, and we provide its costs here.
    """

    @abc.abstractmethod
    def footprint(self) -> int:
        """The number of physical qubits used by the magic state factory."""

    @abc.abstractmethod
    def n_cycles(self, n_magic: MagicCount, phys_err: float) -> int:
        """The number of cycles (time) required to produce the requested number of magic states."""

    @abc.abstractmethod
    def distillation_error(self, n_magic: MagicCount, phys_err: float) -> float:
        """The total error expected from distilling magic states with a given physical error rate."""
