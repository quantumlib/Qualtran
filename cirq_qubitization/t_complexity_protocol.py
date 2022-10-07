from typing import Any, Callable, Iterable, Optional
from dataclasses import dataclass
from typing_extensions import Protocol

import cirq
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits

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


def _is_iterable(it: Any, **kwargs) -> Optional[TComplexity]:
    if not isinstance(it, Iterable):
        return None
    custom_strategies = kwargs['custom_strategies'] if 'custom_strategies' in kwargs else ()
    t = TComplexity()
    for v in it:
        r = t_complexity(v, custom_strategies=custom_strategies)
        if r is None:
            return None
        t = t + r
    return t


def _has_decomposition(stc: Any, **kwargs) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    decomposition, _, _ = _try_decompose_into_operations_and_qubits(stc)
    if decomposition is None:
        return None
    custom_strategies = kwargs['custom_strategies'] if 'custom_strategies' in kwargs else ()
    return _is_iterable(decomposition, custom_strategies=custom_strategies)


def t_complexity(
    stc: Any,
    fail_quietly: bool = False,
    custom_strategies: Iterable[Callable[[Any], TComplexity]] = (),
) -> Optional[TComplexity]:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its TComplexity.
        fail_quietly: bool whether to return None on failure or raise an error.
        custom_strategies: Callables that apply custom strategies
                            and take precendence on cirq.decompose_once.

    Returns:
        The TComplexity of the given object or None on failure (and fail_quietly=True).

    Raises:
        TypeError: if fail_quietly=False and the methods fails to compute TComplexity.
    """
    strategies = [_has_t_complexity, _is_clifford_or_t]
    strategies.extend(custom_strategies)
    strategies.extend([_has_decomposition, _is_iterable])

    for strategy in strategies:
        ret = strategy(stc, custom_strategies=custom_strategies)
        if ret is not None:
            return ret
    if fail_quietly:
        return None
    raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")


def fredkin_t_complexity(val: Any, **kwargs) -> Optional[TComplexity]:
    """Optimal TComplexity of FREDKIN gate from figure 5.2 of https://arxiv.org/pdf/1308.4134.pdf"""
    if isinstance(val, (cirq.Operation, cirq.Gate)) and val in cirq.Gateset(
        cirq.FREDKIN, unroll_circuit_op=False
    ):
        return TComplexity(t=7, clifford=10)
    return None
