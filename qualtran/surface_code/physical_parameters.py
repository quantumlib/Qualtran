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

import attr
from attrs import frozen


@frozen
class PhysicalParameters:
    """PhysicalParameters contains physical properties of a quantum computer.

    Attributes:
        t_gate: Clifford gate time.
        t_meas: Measurement time.
        physical_error_rate: Physical error rate.
        reference: Source of these estimates.
    """

    t_gate = attr.ib(type=float, default=1e-6, repr=lambda x: f'{x:g}')  # 1us
    t_meas = attr.ib(type=float, default=1e-6, repr=lambda x: f'{x:g}')  # 1us

    physical_error_rate = attr.ib(type=float, default=1e-3, repr=lambda x: f'{x:g}')

    reference = attr.ib(type=str, default='')


FowlerGidney = PhysicalParameters(
    t_gate=1e-6, t_meas=1e-6, physical_error_rate=1e-3, reference='https://arxiv.org/abs/1808.06709'
)


BeverlandEtAl = PhysicalParameters(
    t_gate=50 * 1e-9,
    t_meas=100 * 1e-9,
    physical_error_rate=1e-4,
    reference='https://arxiv.org/abs/2211.07629',
)
