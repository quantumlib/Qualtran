from typing import Any, Iterable, Optional
from dataclasses import dataclass
from typing_extensions import Protocol

import cirq
from cirq_qubitization.decompose_protocol import decompose_once_into_operations

_CACHE = {}
_T_GATESET = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)


@dataclass(frozen=True)
class TComplexity:
    t: int = 0
    clifford: int = 0
    rotations: int = 0

    def __add__(self, other: 'TComplexity') -> 'TComplexity':
        return TComplexity(
            self.t + other.t, self.clifford + other.clifford, self.rotations + other.rotations
        )


class SupportsTComplexity(Protocol):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the `_t_complexity_` function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""


def _has_t_complexity(stc: Any, **kwargs) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling its _t_complexity_ if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        result = estimator()
        return None if result is NotImplemented else result
    return None


def _is_clifford_or_t(stc: Any, **kwargs) -> Optional[TComplexity]:
    """Attempts to infer the type of a gate/operation as one of clifford, T or Rotation."""
    if not isinstance(stc, (cirq.Gate, cirq.Operation)):
        return None

    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()

    if cirq.has_stabilizer_effect(stc):
        # Clifford operation.
        return TComplexity(clifford=1)

    if stc in _T_GATESET:
        # T-gate.
        return TComplexity(t=1)  # T gate

    if cirq.num_qubits(stc) == 1 and cirq.has_unitary(stc):
        # Single qubit rotation operation.
        return TComplexity(rotations=1)
    return None


def _is_iterable(it: Any, fail_quietly: bool = False) -> Optional[TComplexity]:
    if not isinstance(it, Iterable):
        return None
    t = TComplexity()
    for v in it:
        r = _t_complexity(v, fail_quietly=fail_quietly)
        if r is None:
            return None
        t = t + r
    return t


def _from_decomposition(stc: Any, fail_quietly: bool = False) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    decomposition = decompose_once_into_operations(stc)
    if decomposition is None:
        return None
    return _is_iterable(decomposition, fail_quietly=fail_quietly)


def get_hash(val: Any) -> Optional[int]:
    """Computes a qubit invariant hash Operations and Gates.

        The hash of a cirq.Operation changes depending on its
        qubits, tags, classical controls, and other properties it might have.
        None of these properties affect the TComplexity.
        For gates and gate backed operations we compute the hash
        of the gate which is a property of the Gate.
        For other operations we default to the hash of the operation.
    Args:
        val: object to comptue its hash.

    Returns:
        hash value or None.
    """
    if not isinstance(val, (cirq.Operation, cirq.Gate)):
        return None
    if not isinstance(val, cirq.Gate):
        if val.gate is not None:
            val = val.gate
    return hash(val)


def _t_complexity(stc: Any, fail_quietly: bool = False) -> Optional[TComplexity]:
    h = get_hash(stc)
    if h in _CACHE:
        return _CACHE[h]
    strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition, _is_iterable]
    ret = None
    for strategy in strategies:
        ret = strategy(stc)
        if ret is not None:
            break
    if ret is None and not fail_quietly:
        raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    if h is not None:
        _CACHE[h] = ret
    return ret


def t_complexity(stc: Any, fail_quietly: bool = False) -> Optional[TComplexity]:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its TComplexity.
        fail_quietly: bool whether to return None on failure or raise an error.

    Returns:
        The TComplexity of the given object or None on failure (and fail_quietly=True).

    Raises:
        TypeError: if fail_quietly=False and the methods fails to compute TComplexity.
    """
    _CACHE.clear()
    return _t_complexity(stc, fail_quietly=fail_quietly)
