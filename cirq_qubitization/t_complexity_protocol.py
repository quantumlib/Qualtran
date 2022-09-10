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

    def _t_complexity_(self) -> TComplexity:
        pass


def t_complexity(stc: Any) -> TComplexity:
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        return estimator()
    
    t = TComplexity()
    for sub_stc in decompose_once(stc):
        t = t + t_complexity(sub_stc)
    return t