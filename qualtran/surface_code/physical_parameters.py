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

from attrs import field, frozen


@frozen
class PhysicalParameters:
    """The physical properties of a quantum computer.

    Attributes:
        physical_error: The error rate of the underlying physical qubits.
        cycle_time_us: The number of microseconds it takes to do one cycle of error correction.

    """

    physical_error: float = field(default=1e-3, repr=lambda x: f'{x:g}')
    cycle_time_us: float = 1.0

    @classmethod
    def make_beverland_et_al(
        cls, qubit_modality: str = 'superconducting', optimistic_err_rate: bool = False
    ):
        """The physical parameters considered in the Beverland et. al. reference.

        Args:
            qubit_modality: One of "superconducting", "ion", or "majorana". This sets the
                cycle time, with ions being considerably slower.
            optimistic_err_rate: In the reference, the authors consider two error rates, which
                they term "realistic" and "optimistic". Set this to `True` to use optimistic
                error rates.

        References:
            [Assessing requirements to scale to practical quantum advantage](https://arxiv.org/abs/2211.07629).
            Beverland et. al. (2022).
        """
        if optimistic_err_rate:
            phys_err_rate = 1e-4
        else:
            phys_err_rate = 1e-3

        if qubit_modality == 'ion':
            t_gate_ns = 100_000
            t_meas_ns = 100_000
        elif qubit_modality == 'superconducting':
            t_gate_ns = 50
            t_meas_ns = 100
        elif qubit_modality == 'majorana':
            if optimistic_err_rate:
                phys_err_rate = 1e-6
            else:
                phys_err_rate = 1e-4

            t_gate_ns = 100
            t_meas_ns = 100
        else:
            raise ValueError(
                f"Unknown qubit modality {qubit_modality}. Must be one "
                f"of 'ion', 'superconducting', or 'majorana'."
            )

        cycle_time_ns = 4 * t_gate_ns + 2 * t_meas_ns
        return PhysicalParameters(
            physical_error=phys_err_rate, cycle_time_us=cycle_time_ns / 1000.0
        )

    @classmethod
    def make_gidney_fowler(cls, optimistic_err_rate: bool = False):
        """The physical parameters considered in the Gidney and Fowler reference.

        References:
            [Efficient magic state factories with a catalyzed |CCZ> to 2|T> transformation](https://arxiv.org/abs/1812.01238).
            Gidney and Fowler (2018).
        """
        if optimistic_err_rate:
            phys_err_rate = 1e-4
        else:
            phys_err_rate = 1e-3
        return PhysicalParameters(physical_error=phys_err_rate, cycle_time_us=1.0)
