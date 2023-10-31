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

_PRETTY_FLOAT = field(default=0.0, converter=float, repr=lambda x: f'{x:g}')


@frozen
class AlgorithmSummary:
    """Properties of a quantum algorithm that impact its physical cost

    Counts of different properties that affect the physical cost of
    running an algorithm (e.g. number of T gates).
    All counts default to zero.

    Attributes:
        algorithm_qubits: Number of qubits used by the algorithm $Q_{alg}$.
        measurements: Number of Measurements $M_R$.
        t_gates: Number of T gates $M_T$.
        toffoli_gates: Number of Toffoli gates $M_{Tof}$.
        rotation_gates: Number of Rotations $M_R$.
        rotation_circuit_depth: Depth of rotation circuit $D_R$.
    """

    algorithm_qubits: float = _PRETTY_FLOAT
    measurements: float = _PRETTY_FLOAT
    t_gates: float = _PRETTY_FLOAT
    toffoli_gates: float = _PRETTY_FLOAT
    rotation_gates: float = _PRETTY_FLOAT
    rotation_circuit_depth: float = _PRETTY_FLOAT

    def __mul__(self, other: int) -> 'AlgorithmSummary':
        if not isinstance(other, int):
            raise TypeError(
                f"Multiplication isn't supported between AlgorithmSummary and non integer type {type(other)}"
            )

        return AlgorithmSummary(
            algorithm_qubits=self.algorithm_qubits * other,
            measurements=self.measurements * other,
            t_gates=self.t_gates * other,
            toffoli_gates=self.toffoli_gates * other,
            rotation_gates=self.rotation_gates * other,
            rotation_circuit_depth=self.rotation_circuit_depth * other,
        )

    def __rmul__(self, other: int) -> 'AlgorithmSummary':
        return self.__mul__(other)

    def __add__(self, other: 'AlgorithmSummary') -> 'AlgorithmSummary':
        if not isinstance(other, AlgorithmSummary):
            raise TypeError(
                f"Addition isn't supported between AlgorithmSummary and type {type(other)}"
            )
        return AlgorithmSummary(
            algorithm_qubits=self.algorithm_qubits + other.algorithm_qubits,
            measurements=self.measurements + other.measurements,
            t_gates=self.t_gates + other.t_gates,
            toffoli_gates=self.toffoli_gates + other.toffoli_gates,
            rotation_gates=self.rotation_gates + other.rotation_gates,
            rotation_circuit_depth=self.rotation_circuit_depth + other.rotation_circuit_depth,
        )

    def __sub__(self, other: 'AlgorithmSummary') -> 'AlgorithmSummary':
        if not isinstance(other, AlgorithmSummary):
            raise TypeError(
                f"Subtraction isn't supported between AlgorithmSummary and type {type(other)}"
            )
        return AlgorithmSummary(
            algorithm_qubits=self.algorithm_qubits - other.algorithm_qubits,
            measurements=self.measurements - other.measurements,
            t_gates=self.t_gates - other.t_gates,
            toffoli_gates=self.toffoli_gates - other.toffoli_gates,
            rotation_gates=self.rotation_gates - other.rotation_gates,
            rotation_circuit_depth=self.rotation_circuit_depth - other.rotation_circuit_depth,
        )
