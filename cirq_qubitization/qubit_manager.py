import abc
from typing import Iterable, List, Set, Sequence, Union

import cirq


class QubitManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qalloc(self, num_qubits: int) -> List[cirq.Qid]:
        ...

    @abc.abstractmethod
    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        ...


class GreedyQubitManager(QubitManager):
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

    def __str__(self) -> str:
        return (
            f'GreedyQubitManager managing:\n'
            f'Used Qubits: {self._used_qubits}\n'
            f'Free Qubits: {self._free_qubits}'
        )
