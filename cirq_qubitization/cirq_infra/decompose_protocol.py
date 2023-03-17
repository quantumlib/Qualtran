from typing import Any, Callable, Optional, Tuple

import cirq
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits


DecomposeResult = Optional[Tuple[cirq.Operation, ...]]
OpDecomposer = Callable[[Any], DecomposeResult]

_FREDKIN_GATESET = cirq.Gateset(cirq.FREDKIN, unroll_circuit_op=False)


def _fredkin(qubits: cirq.Qid) -> cirq.OP_TREE:
    """Decomposition with 7 T and 10 clifford operations from https://arxiv.org/abs/1308.4134"""
    c, t1, t2 = qubits
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t1), cirq.H(t2)]
    yield [cirq.T(c), cirq.T(t1) ** -1, cirq.T(t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t2), cirq.T(t1)]
    yield [cirq.CNOT(c, t1), cirq.T(t2) ** -1]
    yield [cirq.T(t1) ** -1, cirq.CNOT(c, t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.T(t1), cirq.H(t2)]
    yield [cirq.CNOT(t2, t1)]


def _try_decompose_from_known_decompositions(val: Any) -> DecomposeResult:
    """Returns a flattened decomposition of the object into operations, if possible.

    Args:
        val: The object to decompose.

    Returns:
        A flattened decomposition of `val` if it's a gate or operation with a known decomposition.
    """
    known_decompositions = [(_FREDKIN_GATESET, _fredkin)]
    if not isinstance(val, (cirq.Gate, cirq.Operation)):
        return None

    classical_controls = None
    if isinstance(val, cirq.ClassicallyControlledOperation):
        classical_controls = val.classical_controls
        val = val.without_classical_controls()

    if isinstance(val, cirq.Operation):
        qubits = val.qubits
    else:
        qubits = cirq.LineQid.for_gate(val)

    for gateset, decomposer in known_decompositions:
        if val in gateset:
            decomposition = cirq.flatten_op_tree(decomposer(qubits))
            if classical_controls is not None:
                return tuple(op.with_classical_controls(classical_controls) for op in decomposition)
            else:
                return tuple(decomposition)
    return None


def decompose_once_into_operations(
    val: Any,
    intercepting_decomposer: Optional[OpDecomposer] = _try_decompose_from_known_decompositions,
    fallback_decomposer: Optional[OpDecomposer] = None,
) -> DecomposeResult:
    """Decomposes a value into operations, if possible.

    Args:
        val: The value to decompose into operations.
        intercepting_decomposer: An optional method that is called before the
            default decomposer (the value's `_decompose_` method). If
            `intercepting_decomposer` is specified and returns a result that
            isn't `NotImplemented` or `None`, that result is used. Otherwise the
            decomposition falls back to the default decomposer.
        fallback_decomposer: An optional decomposition that used after the
            `intercepting_decomposer` and the default decomposer (the value's
            `_decompose_` method) both fail.
    Returns:
        A tuple of operations if decomposition succeeds.
    """
    decomposers = (
        intercepting_decomposer,
        lambda x: _try_decompose_into_operations_and_qubits(x)[0],
        fallback_decomposer,
    )
    for decomposer in decomposers:
        if decomposer is None:
            continue
        res = decomposer(val)
        if res is not None:
            return res
    return None
