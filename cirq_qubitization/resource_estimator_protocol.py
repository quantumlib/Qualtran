
from typing import Callable, Union
from dataclasses import dataclass
from typing_extensions import Protocol

import cirq
from cirq.type_workarounds import NotImplementedType
from cirq.protocols import SupportsDecompose, decompose_once


@dataclass(frozen=True)
class Resources:
    T: float = 0
    Clifford: float = 0
    Rotations: float = 0

    def __add__(self, other: 'Resources') -> 'Resources':
        return Resources(
                self.T + other.T,
                self.Clifford + other.Clifford,
                self.Rotations + other.Rotations)

ResourcesFunction = Union[None, NotImplementedType, Callable[[int, float], Resources]]

class SupportsResourceEstimation(SupportsDecompose):

    @classmethod
    def _clifford_and_t(cls) -> ResourcesFunction:
        pass

    def _num_qubits_(self) -> int:
        pass

def estimate(sre: SupportsResourceEstimation, eps: float) -> Resources:
    estimator = type(sre)._clifford_and_t()
    if estimator is not None and estimator is not NotImplemented:
        return estimator(sre._num_qubits_(), eps)
    
    r = Resources()
    for sub_sre in decompose_once(sre):
        r = r + estimate(sub_sre, eps)
    return r

def estimate_circuit(circuit: cirq.Circuit, eps: float) -> Resources:
    r = Resources()
    for sre in circuit.all_operations():
        r = r + estimate(sre, eps)
    return r