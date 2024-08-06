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

from attrs import frozen


@frozen
class PhysicalCostsSummary:
    failure_prob: float
    """Approximate probability of an error occurring during execution of the algorithm.

    This can be a bad CCZ being produced, a bad T state being produced,
    or a topological error occurring during the algorithm.
    """

    footprint: int
    """Total physical qubits required to run algorithm."""

    duration_hr: float
    """Total time in hours to run algorithm."""

    @property
    def qubit_hours(self) -> float:
        return self.footprint * self.duration_hr
