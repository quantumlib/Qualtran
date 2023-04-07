from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import cirq
from typing_extensions import Protocol

from cirq_qubitization.cirq_infra.decompose_protocol import decompose_once_into_operations

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

    def __mul__(self, other: int) -> 'TComplexity':
        return TComplexity(self.t * other, self.clifford * other, self.rotations * other)

    def __rmul__(self, other: int) -> 'TComplexity':
        return self.__mul__(other)


class SupportsTComplexity(Protocol):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the `_t_complexity_` function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""


def _has_t_complexity(
    stc: Any, cache: Dict[Any, TComplexity], fail_quietly: bool = False
) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling its _t_complexity_ if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        result = estimator()
        return None if result is NotImplemented else result
    return None


def _is_clifford_or_t(
    stc: Any, cache: Dict[Any, TComplexity], fail_quietly: bool = False
) -> Optional[TComplexity]:
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


def _is_iterable(
    it: Any, cache: Dict[Any, TComplexity], fail_quietly: bool = False
) -> Optional[TComplexity]:
    if not isinstance(it, Iterable):
        return None
    t = TComplexity()
    for v in it:
        r = _t_complexity(v, cache=cache, fail_quietly=fail_quietly)
        if r is None:
            return None
        t = t + r
    return t


def _from_decomposition(
    stc: Any, cache: Dict[Any, TComplexity], fail_quietly: bool = False
) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    decomposition = decompose_once_into_operations(stc)
    if decomposition is None:
        return None
    return _is_iterable(decomposition, cache=cache, fail_quietly=fail_quietly)


def _get_hash(val: Any) -> Optional[int]:
    """Returns a hash of cirq.Operation and cirq.Gate.

        The hash of a cirq.Operation changes depending on its qubits, tags,
        classical controls, and other properties it has, none of these properties
        affect the TComplexity.
        For gates and gate backed operations we compute the hash of the gate which
        is a property of the Gate.
    Args:
        val: object to comptue its hash.

    Returns:
        hash value for gates and gate backed operations or None otherwise.
    """
    if not isinstance(val, (cirq.Operation, cirq.Gate)):
        return None
    if isinstance(val, cirq.Operation):
        val = val.gate
        if val is None:
            return None
    return hash(val)


def _t_complexity(
    stc: Any, cache: Dict[Any, TComplexity], fail_quietly: bool = False
) -> Optional[TComplexity]:
    h = _get_hash(stc)
    if h is not None and h in cache:
        return cache[h]
    strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition, _is_iterable]
    ret = None
    for strategy in strategies:
        ret = strategy(stc, cache=cache, fail_quietly=fail_quietly)
        if ret is not None:
            break
    if ret is None and not fail_quietly:
        raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    if h is not None:
        cache[h] = ret
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
    cache = {}
    return _t_complexity(stc, cache=cache, fail_quietly=fail_quietly)
