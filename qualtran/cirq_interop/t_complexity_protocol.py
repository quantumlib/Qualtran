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
from typing import Any, Callable, Hashable, Iterable, Optional, Protocol, Union

import attrs
import cachetools
import cirq

from qualtran import Bloq, Controlled
from qualtran.cirq_interop.decompose_protocol import _decompose_once_considering_known_decomposition
from qualtran.resource_counting.symbolic_counting_utils import (
    ceil,
    log2,
    SymbolicFloat,
    SymbolicInt,
)

_T_GATESET = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)
_ROTS_GATESET = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate)


@attrs.frozen
class TComplexity:
    """Dataclass storing counts of logical T-gates, Clifford gates and single qubit rotations."""

    t: int = 0
    clifford: int = 0
    rotations: int = 0

    @staticmethod
    def rotation_cost(eps: SymbolicFloat) -> SymbolicFloat:
        return ceil(1.149 * log2(1.0 / eps) + 9.2)

    def t_incl_rotations(self, eps: float = 1e-11) -> SymbolicInt:
        """Return the total number of T gates after compiling rotations"""

        # TODO Determine precise clifford count and/or ignore.
        # This is an improvement over Ref. 2 from the docstring which provides
        # a bound of 3 log(1/eps).
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        return ceil(self.t + self.rotation_cost(eps) * self.rotations)

    def __add__(self, other: 'TComplexity') -> 'TComplexity':
        return TComplexity(
            self.t + other.t, self.clifford + other.clifford, self.rotations + other.rotations
        )

    def __mul__(self, other: int) -> 'TComplexity':
        return TComplexity(self.t * other, self.clifford * other, self.rotations * other)

    def __rmul__(self, other: int) -> 'TComplexity':
        return self.__mul__(other)

    def __str__(self) -> str:
        return (
            f'T-count:   {self.t:g}\n'
            f'Rotations: {self.rotations:g}\n'
            f'Cliffords: {self.clifford:g}\n'
        )


class SupportsTComplexity(Protocol):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the `_t_complexity_` function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""


def _from_explicit_annotation(stc: Any) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling `stc._t_complexity_()` method, if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        result = estimator()
        if result is not NotImplemented:
            return result
    if isinstance(stc, cirq.Operation) and stc.gate is not None:
        return _from_explicit_annotation(stc.gate)
    return None


def _from_directly_countable(stc: Any) -> Optional[TComplexity]:
    """Directly count a clifford, T or Rotation (if it is one)."""
    from qualtran.bloqs.basic_gates import TGate

    if isinstance(stc, TGate):
        return TComplexity(t=1)

    if not isinstance(stc, (cirq.Gate, cirq.Operation)):
        return None

    if isinstance(stc, cirq.GlobalPhaseGate) or (
        isinstance(stc, cirq.Operation) and isinstance(stc.gate, cirq.GlobalPhaseGate)
    ):
        return TComplexity()

    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()

    if cirq.num_qubits(stc) <= 2 and cirq.has_stabilizer_effect(stc):
        # Clifford operation.
        return TComplexity(clifford=1)

    if stc in _T_GATESET:
        # T-gate.
        return TComplexity(t=1)  # T gate

    if stc in _ROTS_GATESET:
        return TComplexity(rotations=1)

    if isinstance(stc, Controlled) and cirq.num_qubits(stc) <= 2:
        # We need this hack temporarily because we assume access to decomposition
        # of a C-U gate where $U$ is a single qubit rotation. Cirq has this decomposition
        # but the right thing to do in Qualtran is to add explicit bloqs and annotate
        # them with costs. See https://github.com/quantumlib/Qualtran/issues/878
        from qualtran._infra.gate_with_registers import get_named_qubits

        quregs = get_named_qubits(stc.signature)
        qm = cirq.SimpleQubitManager()
        op, _ = stc.as_cirq_op(qubit_manager=qm, **quregs)
        return t_complexity(cirq.decompose_once(op))

    if cirq.num_qubits(stc) == 1 and cirq.has_unitary(stc):
        # Single qubit rotation operation.
        return TComplexity(rotations=1)
    return None


def _from_iterable(it: Any) -> Optional[TComplexity]:
    if not isinstance(it, Iterable):
        return None
    t = TComplexity()
    for v in it:
        r = t_complexity(v)
        if r is None:
            return None
        t = t + r
    return t


def _from_bloq_build_call_graph(stc: Any) -> Optional[TComplexity]:
    # Uses the depth 1 call graph of Bloq `stc` to recursively compute the complexity.
    from qualtran.resource_counting import get_bloq_callee_counts
    from qualtran.resource_counting.generalizers import cirq_to_bloqs

    if not isinstance(stc, Bloq):
        return None
    callee_counts = get_bloq_callee_counts(bloq=stc, generalizer=cirq_to_bloqs)
    if len(callee_counts) == 0:
        return None
    ret = TComplexity()
    for bloq, n in callee_counts:
        r = t_complexity(bloq)
        if r is None:
            return None
        ret += n * r
    return ret


def _from_cirq_decomposition(stc: Any) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    decomposition = _decompose_once_considering_known_decomposition(stc)
    if decomposition is None:
        return None
    return _from_iterable(decomposition)


def _get_hash(val: Any):
    """Returns hash keys for caching a cirq.Operation and cirq.Gate.

    The hash of a cirq.Operation changes depending on its qubits, tags,
    classical controls, and other properties it has, none of these properties
    affect the TComplexity.
    For gates and gate backed operations we intend to compute the hash of the
    gate which is a property of the Gate.

    Args:
        val: object to compute its hash.

    Returns:
        - `val.gate` if `val` is a `cirq.Operation` which has an underlying `val.gate`.
        - `val` otherwise
    """
    if isinstance(val, cirq.Operation) and val.gate is not None:
        val = val.gate
    return val


def _t_complexity_from_strategies(
    stc: Any, strategies: Iterable[Callable[[Any], Optional[TComplexity]]]
):
    ret = None
    for strategy in strategies:
        ret = strategy(stc)
        if ret is not None:
            break
    return ret


@cachetools.cached(cachetools.LRUCache(128), key=_get_hash, info=True)
def _t_complexity_for_gate_or_op(
    gate_or_op: Union[cirq.Gate, cirq.Operation, Bloq]
) -> Optional[TComplexity]:

    if isinstance(gate_or_op, cirq.Operation) and gate_or_op.gate is not None:
        gate_or_op = gate_or_op.gate

    strategies = [
        _from_explicit_annotation,
        _from_bloq_build_call_graph,
        _from_directly_countable,
        _from_cirq_decomposition,
    ]
    return _t_complexity_from_strategies(gate_or_op, strategies)


def t_complexity(stc: Any) -> TComplexity:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its TComplexity.

    Returns:
        The TComplexity of the given object.

    Raises:
        TypeError: if the methods fails to compute TComplexity.
    """
    if isinstance(stc, (cirq.Gate, cirq.Operation, Bloq)) and isinstance(stc, Hashable):
        ret = _t_complexity_for_gate_or_op(stc)
    else:
        strategies = [
            _from_explicit_annotation,
            _from_directly_countable,
            _from_bloq_build_call_graph,
            _from_cirq_decomposition,
            _from_iterable,
        ]
        ret = _t_complexity_from_strategies(stc, strategies)

    if ret is None:
        raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    return ret


t_complexity.cache_clear = _t_complexity_for_gate_or_op.cache_clear  # type: ignore[attr-defined]
t_complexity.cache_info = _t_complexity_for_gate_or_op.cache_info  # type: ignore[attr-defined]
t_complexity.cache = _t_complexity_for_gate_or_op.cache  # type: ignore[attr-defined]
