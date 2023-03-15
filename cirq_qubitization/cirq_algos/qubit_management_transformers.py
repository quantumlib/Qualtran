from typing import Dict, Tuple

import cirq

from cirq_qubitization.cirq_algos import qid_types, qubit_manager


def _get_qubit_mapping_first_and_last_moment(
    circuit: cirq.AbstractCircuit,
) -> Dict[cirq.Qid, Tuple[int, int]]:
    ret = {q: (len(circuit), 0) for q in circuit.all_qubits()}
    for i, moment in enumerate(circuit):
        for q in moment.qubits:
            ret[q] = (min(ret[q][0], i), max(ret[q][1], i))
    return ret


def _is_temp(q: cirq.Qid) -> bool:
    return isinstance(q, (qid_types.CleanQubit, qid_types.BorrowableQubit))


def decompose_and_allocate_qubits(
    circuit: cirq.AbstractCircuit,
    *,
    prefix: str = "ancilla",
    decompose: lambda op: cirq.decompose_once(op, default=op),
) -> cirq.Circuit:
    """Replaces all `CleanQubit`/`Borrowable` qubits by greedily allocating named qubits."""
    allocated_qubits = {q for q in circuit.all_qubits() if _is_temp(q)}
    qubits_lifespan = _get_qubit_mapping_first_and_last_moment(circuit)
    qm = qubit_manager.GreedyQubitManager(prefix=prefix)
    all_qubits = frozenset(circuit.all_qubits() - allocated_qubits)
    trivial_map = {q: q for q in all_qubits}
    # Allocated map maintains the mapping of all temporary qubits seen so far, mapping each of
    # them to either a newly allocated managed ancilla or an existing borrowed system qubit.
    allocated_map = {}
    to_free = set()
    last_op_idx = -1

    def map_func(op: cirq.Operation, idx: int) -> cirq.OP_TREE:
        nonlocal last_op_idx, to_free

        if idx > last_op_idx:
            # New moment, free up all clean / borrowed qubits which ended in the previous moment.
            for q in sorted(to_free):
                if qm.is_used(allocated_map[q]):
                    qm.qfree([allocated_map[q]])
                allocated_map.pop(q)
            to_free = set()

        last_op_idx = idx

        # Find the decomposition of this operation and the set of qubits that it acts upon.
        decomposed = cirq.Circuit(decompose(op))
        op, op_qubits = decomposed, decomposed.all_qubits()
        assert all(q in all_qubits for q in op_qubits if not _is_temp(q))

        # To check borrowable qubits, we manually manage only the original system qubits
        # that are not managed by the qubit manager. If any of the system qubits cannot be
        # borrowed, we defer to the qubit manager to allocate a new clean qubit for us.
        # This is a heuristic and can be improved by also checking if any allocated but not
        # yet freed managed qubit can be borrowed for the shorter scope, but we ignore the
        # optimization for the sake of simplicity here.
        borrowable_qubits = set(all_qubits - set(allocated_map.values()) - op_qubits)
        for q in sorted(op_qubits):
            if not _is_temp(q):
                continue

            # At this point, the qubit `q` is a temporary allocated ancilla which either already
            # existed in the original circuit, in which case it's qubit lifespan is known, or it
            # appeared in the circuit due to decomposition of `op`, in which case we assign it's
            # lifespan to be (idx, idx) since we preserve moment structure during decomposition.
            st, en = qubits_lifespan[q] if q in qubits_lifespan else (idx, idx)
            assert st <= idx <= en
            if st < idx:
                # If we have seen this temporary qubit before, it should already have a mapping.
                assert q in allocated_map
                continue

            if isinstance(q, qid_types.BorrowableQubit):
                # For each of the system qubits that can be borrowed, check whether they have a
                # conflicting operation in the range [st, en]; which is the scope for which the
                # borrower needs the borrowed qubit for.
                start_frontier = {q: st for q in borrowable_qubits}
                end_frontier = {q: en + 1 for q in borrowable_qubits}
                ops_in_between = circuit.findall_operations_between(start_frontier, end_frontier)
                # Filter the set of borrowable qubits which do not have any conflicting operations.
                filtered_borrowable_qubits = borrowable_qubits - set(
                    q for _, op in ops_in_between for q in op.qubits
                )
                if filtered_borrowable_qubits:
                    # Allocate a borrowable qubit and remove it from the pool of available qubits.
                    allocated_map[q] = min(filtered_borrowable_qubits)
                    borrowable_qubits.remove(allocated_map[q])
                    continue

            # Allocate a new clean qubit if `q` was a CleanQubit type, or we couldn't find a
            # suitable borrowable qubit.
            allocated_map[q] = qm.qalloc(1)[0]
            # The qm-managed qubit should not be part of our manually managed borrowable qubit set.
            assert allocated_map[q] not in borrowable_qubits

        # Mark all temporarily allocated qubits which can be freed after this moment.
        for q in op_qubits:
            if not _is_temp(q):
                continue
            _, en = qubits_lifespan[q] if q in qubits_lifespan else (idx, idx)
            if en == idx:
                to_free.add(q)

        # Return the transformed operation / decomposed op-tree.
        return op.transform_qubits({**allocated_map, **trivial_map})

    return cirq.map_operations_and_unroll(circuit, map_func, raise_if_add_qubits=False).unfreeze()
