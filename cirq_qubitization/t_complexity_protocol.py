from typing import Any
from dataclasses import dataclass
from typing_extensions import Protocol

import cirq
from cirq.protocols import SupportsDecompose, decompose_once


@dataclass(frozen=True)
class TComplexity:
    t: int = 0
    clifford: int = 0
    rotations: int = 0

    def __add__(self, other: 'TComplexity') -> 'TComplexity':
        return TComplexity(
                self.t + other.t,
                self.clifford + other.clifford,
                self.rotations + other.rotations)


class SupportsTComplexity(SupportsDecompose):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the _t_complexity function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""
        pass


def t_complexity(stc: Any) -> TComplexity:
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        #if _t_complexity is defined return its results.
        return estimator()        

    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()

    try:
        if cirq.has_stabilizer_effect(stc):
            # Clifford operation.
            return TComplexity(clifford=1)
    except:
        pass

    if isinstance(stc, cirq.GateOperation):
        if stc.gate is cirq.T or (isinstance(stc.gate, cirq.ZPowGate) and stc.gate**-1 == cirq.T):
            return TComplexity(t=1) # T gate
        if isinstance(stc.gate, (cirq.XPowGate, cirq.CXPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate)):
            # Rotation Operation
            return TComplexity(rotations=1)

    # Decompose the object and recursively compute the complexity.
    t = TComplexity()
    for sub_stc in decompose_once(stc):
        t = t + t_complexity(sub_stc)
    return t