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
from typing import Optional

from attrs import field, frozen


@frozen
class PhysicalParameters:
    """The physical properties of a quantum computer.

    Attributes:
        t_gate_ns: Clifford gate physical time.
        t_meas_ns: Measurement physical time.
        physical_error_rate: Physical error rate.
        reference: Source of these estimates.
    """

    t_gate_ns: float = field(repr=lambda x: f'{x:g}')
    t_meas_ns: float = field(repr=lambda x: f'{x:g}')

    physical_error_rate: float = field(default=1e-3, repr=lambda x: f'{x:g}')

    reference: Optional[str] = None


BEVERLAND_PARAMS = PhysicalParameters(
    t_gate_ns=50,  # 50ns
    t_meas_ns=100,  # 100ns
    physical_error_rate=1e-4,
    reference='https://arxiv.org/abs/2211.07629',
)
