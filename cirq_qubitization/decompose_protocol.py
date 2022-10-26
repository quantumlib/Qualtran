from typing import Any, Iterable, List, Optional, Tuple

import cirq
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits

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


def _try_decompose_from_known_decompositions(val: Any) -> Optional[Tuple[cirq.Operation, ...]]:
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


def decompose_once_into_operations(val: Any) -> Optional[Iterable[cirq.Operation]]:
    """Decomposes a value into operations, if possible.
    
        This method decomposes the value exactly once, repeated application
        of the function results in a decomposition with optimal T complexity.
    """
    res = _try_decompose_from_known_decompositions(val)
    if res is not None:
        return res
    decomposition, _, _ = _try_decompose_into_operations_and_qubits(val)
    return decomposition
