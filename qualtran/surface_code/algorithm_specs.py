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

_PRETTY_FLOAT = field(default=0.0, repr=lambda x: f'{x:g}')


@frozen
class AlgorithmSpecs:
    """Properties of a quantum algorithm that impact its physical cost

    Attributes:
        algorithm_qubits: Number of qubits used by the algorithm $Q_{alg}$.
        measurements: Number of Measurements $M_R$.
        t_gates: Number of T gates $M_T$.
        toffoli_gates: Number of Toffoli gates $M_{Tof}$.
        rotation_gates: Number of Rotations $M_R$.
        rotation_circuit_depth: Depth of rotation circuit $D_R$.
    """

    algorithm_qubits = _PRETTY_FLOAT
    measurements = _PRETTY_FLOAT
    t_gates = _PRETTY_FLOAT
    toffoli_gates = _PRETTY_FLOAT
    rotation_gates = _PRETTY_FLOAT
    rotation_circuit_depth = _PRETTY_FLOAT
