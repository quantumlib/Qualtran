from typing import Any, Callable, Iterable, Optional

import cirq
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits

_FREDKIN_GATESET = cirq.Gateset(cirq.FREDKIN, unroll_circuit_op=False)
_T_GATESET = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)

def _is_t_cnot_or_single_qubit_clifford(op: cirq.Operation) -> bool:
    """checks if `op` is T, T^-1, CNOT or a single qubit clifford."""
    if not isinstance(op, cirq.Operation):
        return False
    if isinstance(op, cirq.ClassicallyControlledOperation):
        op = op.without_classical_controls()
    if op in _T_GATESET:
        return True
    if op.gate is cirq.CNOT:
        return True
    if len(op.qubits) == 1 and cirq.has_stabilizer_effect(op):
        return True
    return False

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


def _decompose_from_known_decompositions(val: Any) -> Optional[Iterable[cirq.Operation]]:
    """Returns a flattened decomposition of the object or None.

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
                return tuple(
                    map(lambda op: op.with_classical_controls(classical_controls), decomposition)
                )
            else:
                return tuple(decomposition)
    return None


def decompose_once(val: Any) -> Optional[Iterable[cirq.Operation]]:
    """Returns a decomposition of `val` with optimal T complexity."""
    res = _decompose_from_known_decompositions(val)
    if res is not None:
        return res
    decomposition, _, _ = _try_decompose_into_operations_and_qubits(val)
    return decomposition


def decompose(
    val: Any, keep: Optional[Callable[['cirq.Operation'], bool]] = None
) -> Iterable[cirq.Operation]:
    """Recursively decomposes `val` into `cirq.Operation`s with optimial T-complexity.

        Extends `cirq.decompose` to work with `cirq.Gate`s.
        The method uses an interceptor to produce a decomposition with optimal T count
        and to avoid decomposing T, CNOT or single qubit clifford operations.
    Args:
        val: object to decompose.
        keep: A callable that indicates whether to keep an operation or decompose it.

    Returns:
        A decomposition that has optimal T complexity.
    """
    if isinstance(val, cirq.Gate):
        val = val.on(*cirq.LineQid.for_gate(val))

    def is_leaf(op: cirq.Operation) -> bool:
        if _is_t_cnot_or_single_qubit_clifford(op):
            return True
        if keep is not None and keep(op):
            return True
        return False

    return cirq.decompose(
        val, intercepting_decomposer=_decompose_from_known_decompositions,
        keep=is_leaf,
        on_stuck_raise=None
    )