from typing import Dict, Iterable, List, Set, Tuple

import cirq

from cirq_qubitization.qid_types import BorrowableQubit, CleanQubit


def qalloc_clean(n: int) -> List[CleanQubit]:
    return [CleanQubit() for _ in range(n)]


def qalloc_borrow(n: int) -> List[BorrowableQubit]:
    return [BorrowableQubit() for _ in range(n)]


def qalloc_reset() -> None:
    CleanQubit.reset_count()
    BorrowableQubit.reset_count()


class GreedyQubitManager:
    def __init__(self, prefix: str, *, size: int = 0, parallelize: bool = True):
        self._prefix = prefix
        self._used_qubits: Set[cirq.NamedQubit] = set()
        self._free_qubits: List[cirq.NamedQubit] = []
        self._size = 0
        self.parallelize = parallelize
        self.resize(size)

    def resize(self, new_size: int) -> None:
        if new_size <= self._size:
            return
        new_qubits = [cirq.q(f'{self._prefix}_{s}') for s in range(self._size, new_size)]
        self._free_qubits = new_qubits + self._free_qubits
        self._size = new_size

    def qalloc(self, n: int) -> List[cirq.Qid]:
        self.resize(self._size + n - len(self._free_qubits))
        ret_qubits = self._free_qubits[:n] if self.parallelize else self._free_qubits[-n:]
        self._free_qubits = self._free_qubits[n:] if self.parallelize else self._free_qubits[:-n]
        self._used_qubits.update(ret_qubits)
        return ret_qubits

    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        qs = set(qubits)
        assert self._used_qubits.issuperset(qs), "Only managed qubits currently in-use can be freed"
        self._used_qubits -= qs
        self._free_qubits.extend(qs)

    def is_used(self, qubit: cirq.Qid) -> bool:
        return qubit in self._used_qubits

    def __str__(self) -> str:
        return (
            f'GreedyQubitManager managing:\n'
            f'Used Qubits: {self._used_qubits}\n'
            f'Free Qubits: {self._free_qubits}'
        )


def _get_qubit_mapping_first_and_last_moment(
    circuit: cirq.AbstractCircuit,
) -> Dict[cirq.Qid, Tuple[int, int]]:
    ret = {q: (len(circuit), 0) for q in circuit.all_qubits()}
    for i, moment in enumerate(circuit):
        for q in moment.qubits:
            ret[q] = (min(ret[q][0], i), max(ret[q][1], i))
    return ret


def _is_temp(q: cirq.Qid) -> bool:
    return isinstance(q, (CleanQubit, BorrowableQubit))


def expand_composite_and_allocate_qubits(
    circuit: cirq.AbstractCircuit,
    *,
    prefix: str = "ancilla",
    decompose: lambda op: cirq.decompose_once(op, default=op),
) -> cirq.Circuit:
    allocated_qubits = {q for q in circuit.all_qubits() if _is_temp(q)}
    qubits_lifespan = _get_qubit_mapping_first_and_last_moment(circuit)
    qm = GreedyQubitManager(prefix=prefix)
    all_qubits = set(circuit.all_qubits() - allocated_qubits)
    qubit_map = {q: q for q in all_qubits}
    to_free = set()
    last_op_idx = -1

    def map_func(op: cirq.Operation, idx: int) -> cirq.OP_TREE:
        nonlocal last_op_idx, to_free
        if idx > last_op_idx:
            for q in sorted(to_free):
                qm.qfree(q)
                qubit_map.pop(q)
            to_free = set()
        last_op_idx = idx

        op_qubits = frozenset(op.qubits)
        decomposed = cirq.Circuit(decompose(op))
        if op_qubits != decomposed.all_qubits():
            op, op_qubits = decomposed, decomposed.all_qubits()
        assert all(q in all_qubits for q in op_qubits if not _is_temp(q))
        borrowable_qubits = all_qubits - set(
            qubit_map[q] if q in qubit_map else q for q in op_qubits
        )
        op_qubits = sorted(op_qubits)
        for q in op_qubits:
            if not _is_temp(q):
                continue
            st, en = qubits_lifespan[q] if q in qubits_lifespan else (idx, idx)
            assert st <= idx <= en
            if st < idx:
                assert q in qubit_map
                continue
            if isinstance(q, BorrowableQubit):
                start_frontier = {q: st for q in borrowable_qubits}
                end_frontier = {q: en + 1 for q in borrowable_qubits}
                ops_in_between = circuit.findall_operations_between(start_frontier, end_frontier)
                filtered_borrowable_qubits = borrowable_qubits - set(
                    q for _, op in ops_in_between for q in op.qubits
                )
                if filtered_borrowable_qubits:
                    qubit_map[q] = min(filtered_borrowable_qubits)
                    borrowable_qubits.remove(qubit_map[q])
                    continue
            qubit_map[q] = qm.qalloc(1)[0]
        for q in op_qubits:
            if not _is_temp(q):
                continue
            _, en = qubits_lifespan[q] if q in qubits_lifespan else (idx, idx)
            if en == idx and qm.is_used(qubit_map[q]):
                to_free.add(qubit_map[q])
        return op.transform_qubits(qubit_map)

    return cirq.map_operations_and_unroll(circuit, map_func, raise_if_add_qubits=False)
