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

_PRETTY_FLOAT = attr.ib(type=float, default=0.0, repr=lambda x: f'{x:g}')


@frozen
class AlgorithmSpecs:
    """Properties of a quantum algorithm that impact its physical cost

    Attributes:
        algorithm_qubits: Number of qubits used by the algorithm.
        measurements: Number of Measurements.
        t_gates: Number of T gates.
        toffoli_gates: Number of Toffoli gates.
        rotation_gates: Number of Rotations.
        rotation_circuit_depth: Depth of rotation circuit.
    """

    algorithm_qubits = _PRETTY_FLOAT  # Number of algorithm qubits $Q_{alg}$
    measurements = _PRETTY_FLOAT  # Number of measurements $M_R$.
    t_gates = _PRETTY_FLOAT  # Number of T gates $M_T$.
    toffoli_gates = _PRETTY_FLOAT  # Number of Toffoli gates $M_{Tof}$.
    rotation_gates = _PRETTY_FLOAT  # Number of Rotations $M_R$.
    rotation_circuit_depth = _PRETTY_FLOAT  # Depth of rotation circuit $D_R$.
