from typing import List
import cirq


def And(c1: cirq.Qid, c2: cirq.Qid, target: cirq.Qid) -> cirq.Operation:
    return cirq.X(target).controlled_by(c1, c2, control_values=[1, 0])


def AndAdjoint(c1: cirq.Qid, c2: cirq.Qid, target: cirq.Qid) -> cirq.Operation:
    return cirq.CCNOT(c1, c2, target)


def _unary_iteration_impl(
    control: cirq.Qid,
    selection: List[cirq.Qid],
    ancilla: List[cirq.Qid],
    operations: List[cirq.Operation],
    sl: int,
    l: int,
    r: int,
) -> cirq.OP_TREE:
    iteration_length = len(operations)
    if l >= min(r, iteration_length):
        yield []
    if l == (r - 1):
        yield operations[l].controlled_by(control)
    else:
        assert sl < len(selection)
        m = (l + r) >> 1
        if m >= iteration_length:
            yield from _unary_iteration_impl(
                control, selection, ancilla, operations, sl + 1, l, m
            )
        else:
            anc, sq = ancilla[sl], selection[sl]
            yield And(control, sq, anc)
            yield from _unary_iteration_impl(
                anc, selection, ancilla, operations, sl + 1, l, m
            )
            yield cirq.CNOT(control, anc)
            yield from _unary_iteration_impl(
                anc, selection, ancilla, operations, sl + 1, m, r
            )
            yield AndAdjoint(control, sq, anc)


def unary_iteration(
    control: cirq.Qid,
    selection: List[cirq.Qid],
    ancilla: List[cirq.Qid],
    operations: List[cirq.Operation],
) -> cirq.CircuitOperation:
    """Implements OP[l] when selection register stores integer `l`."""
    assert len(ancilla) == len(selection)
    return cirq.CircuitOperation(
        cirq.FrozenCircuit(
            _unary_iteration_impl(
                control, selection, ancilla, operations, 0, 0, 2 ** len(selection)
            )
        )
    )
