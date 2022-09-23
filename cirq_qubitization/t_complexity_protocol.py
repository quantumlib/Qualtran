from typing import Any, Iterable, Optional
from dataclasses import dataclass
from typing_extensions import Protocol

import cirq
from cirq.protocols import SupportsDecompose
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq import flatten_op_tree

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


class SupportsTComplexity(Protocol):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the `_t_complexity_` function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""
        pass


def _has_t_complexity(stc: Any) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling its _t_complexity_ if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        return estimator()        
    return None

def _is_cliffort_or_t(stc: Any) -> Optional[TComplexity]:
    """Attempts to infer the type of an operation as one of clifford, T or Rotation."""
    if isinstance(stc, cirq.AbstractCircuit):
        return None

    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()

    if cirq.has_stabilizer_effect(stc):
        # Clifford operation.
        return TComplexity(clifford=1)

    if isinstance(stc, (cirq.Gate, cirq.Operation)):
        # Gateset in operator assumes operand is a Gate or Operation
        if stc in cirq.Gateset(cirq.T, cirq.T ** -1):
            return TComplexity(t=1) # T gate

        if stc in cirq.Gateset(cirq.XPowGate, cirq.CXPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate):
            # Rotation Operation
            return TComplexity(rotations=1)
    return None

def _has_decomposition(stc: Any) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    t = TComplexity()
    decomposition, _, _ = _try_decompose_into_operations_and_qubits(stc)
    if decomposition is None:
        return None
    for sub_stc in decomposition:
        st = t_complexity(sub_stc)
        if st is None:
            return None
        t = t + st
    return t

def _is_iterable(it: Any) -> Optional[TComplexity]:
    if isinstance(it, Iterable) and not isinstance(it, str):
        t = TComplexity()
        for v in it:
            r = t_complexity(v)
            if r is None:
                return None
            t = t + r
        return t
    return None

def _is_op_tree(tree: Any) -> Optional[TComplexity]:
    try:
        t = TComplexity()
        for v in flatten_op_tree(tree):
            r = t_complexity(v)
            if r is None:
                return None
            t = t + r
        return t
    except:
        return None

def t_complexity(stc: Any) -> Optional[TComplexity]:
    """Returns the TComplexity or None on failure."""
    strategies = [
        _has_t_complexity,
        _is_cliffort_or_t,
        _has_decomposition,
        _is_iterable,
        _is_op_tree,
    ]
    for strategy in strategies:
        ret = strategy(stc)
        if ret is not None:
            return ret
    return None

