import abc
import contextlib
from typing import Iterable, List, Set, TypeVar

import cirq

from cirq_qubitization.cirq_infra import qid_types


class QubitManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qalloc(self, n: int) -> List[cirq.Qid]:
        """Allocate `n` clean qubits, i.e. qubits guaranteed to be in state |0>."""

    @abc.abstractmethod
    def qborrow(self, n: int) -> List[cirq.Qid]:
        """Allocate `n` dirty qubits, i.e. the returned qubits can be in any state."""

    @abc.abstractmethod
    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        """Free pre-allocated clean or dirty qubits managed by this qubit manager."""


QubitManagerT = TypeVar('QubitManagerT', bound=QubitManager)


class SimpleQubitManager:
    """Always allocates a new `CleanQubit`/`BorrowableQubit` for `qalloc`/`qborrow` requests."""

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
    """Greedily allocator that maximizes/minimizes qubit reuse based on a configurable parameter.

    Greedy qubit manager can be configured, using `parallelize` flag, to work in one of two modes:
    - Minimize qubit reuse (parallelize=True): For a fixed width, this mode uses a FIFO (First in
            First out) strategy s.t. the next allocated qubit is one which was freed the earliest.
    - Maximize qubit reuse (parallelize=False): For a fixed width, this mode uses a LIFO (Last in
            First out) strategy s.t. the next allocated qubit is one which was freed the latest.

    If the requested qubits are more than the set of free qubits, the qubit manager automatically
    resizes the size of the managed qubit pool and adds new free qubits, that have their last
    freed time to be -infinity.

    For borrowing qubits, the qubit manager simply delegates borrow requests to `self.qalloc`, thus
    always allocating new clean qubits.
    """

    def __init__(self, prefix: str, *, size: int = 0, parallelize: bool = True):
        """Initializes `GreedyQubitManager`

        Args:
            prefix: The prefix to use for naming new clean ancilla's allocated by the qubit manager.
                    The i'th allocated qubit is of the type `cirq.NamedQubit(f'{prefix}_{i}')`.
            size: The initial size of the pool of ancilla qubits managed by the qubit manager. The
                    qubit manager can automatically resize itself when the allocation request
                    exceeds the number of available qubits.
            parallelize: Flag to control a FIFO vs LIFO strategy, defaults to True (FIFO).
        """
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
