import abc
import contextlib
from typing import Iterable, List, Set, TypeVar

import cirq

from cirq_qubitization.cirq_algos import qid_types


class QubitManager:
    @abc.abstractmethod
    def qalloc(self, n: int) -> List[cirq.Qid]:
        pass

    @abc.abstractmethod
    def qborrow(self, n: int) -> List[cirq.Qid]:
        pass

    @abc.abstractmethod
    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        pass


QubitManagerT = TypeVar('QubitManagerT', bound=QubitManager)


class SimpleQubitManager:
    """Always allocates a new qubit."""

    def __init__(self):
        self._clean_id = 0
        self._borrow_id = 0

    def qalloc(self, n: int) -> List[cirq.Qid]:
        self._clean_id = self._clean_id + n
        return [qid_types.CleanQubit(i) for i in range(self._clean_id - n, self._clean_id)]

    def qborrow(self, n: int) -> List[cirq.Qid]:
        self._borrow_id = self._borrow_id + n
        return [qid_types.BorrowableQubit(i) for i in range(self._borrow_id - n, self._borrow_id)]

    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        for q in qubits:
            good = isinstance(q, qid_types.CleanQubit) and q.id < self._clean_id
            good |= isinstance(q, qid_types.BorrowableQubit) and q.id < self._borrow_id
            if not good:
                raise ValueError(f"{q} was not allocated by {self}")


_global_qubit_manager = SimpleQubitManager()


def qalloc(n: int) -> List[cirq.Qid]:
    return _global_qubit_manager.qalloc(n)


def qborrow(n: int) -> List[cirq.Qid]:
    return _global_qubit_manager.qborrow(n)


def qfree(qubits: Iterable[cirq.Qid]) -> None:
    return _global_qubit_manager.qfree(qubits)


@contextlib.contextmanager
def memory_management_context(qubit_manager: QubitManagerT = None) -> None:
    if qubit_manager is None:
        qubit_manager = SimpleQubitManager()
    global _global_qubit_manager
    _global_qubit_manager = qubit_manager
    try:
        yield
    finally:
        _global_qubit_manager = SimpleQubitManager()


class GreedyQubitManager(QubitManager):
    def __init__(self, prefix: str, *, size: int = 0, parallelize: bool = True):
        self._prefix = prefix
        self._used_qubits: Set[cirq.Qid] = set()
        self._free_qubits: List[cirq.Qid] = []
        self._size = 0
        self.parallelize = parallelize
        self.resize(size)

    def resize(self, new_size: int) -> None:
        if new_size <= self._size:
            return
        new_qubits: List[cirq.Qid] = [
            cirq.q(f'{self._prefix}_{s}') for s in range(self._size, new_size)
        ]
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

    def qborrow(self, n: int) -> List[cirq.Qid]:
        return self.qalloc(n)

    def is_used(self, qubit: cirq.Qid) -> bool:
        return qubit in self._used_qubits
