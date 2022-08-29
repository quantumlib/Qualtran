import cirq
from typing import List

class QubitManager:
    _free_qubits = []
    _total_allocated = 0

    @classmethod
    def _allocate(cls):
        if not len(cls._free_qubits):
            cls._total_allocated += 1
            cls._free_qubits.append(cirq.NamedQubit(''))
        return cls._free_qubits.pop()
    
    @classmethod
    def _free(cls, q: cirq.NamedQubit):
        cls._free_qubits.append(q)

def qalloc(n: int = 1, prefix: str = "") -> List[cirq.NamedQubit]:
    name = lambda i: prefix + str(i)
    if n == 1: name = lambda i: prefix
    qubits = [QubitManager._allocate() for i in range(n)]
    for i in range(n):
        qubits[i]._name = name(i)
    return qubits

def qfree(q: cirq.NamedQubit) -> None:
    QubitManager._free(q)


def qtot() -> int:
    return QubitManager._total_allocated

if __name__ == "__main__":
    qs = qalloc(2, "test")
    print(qs)
    qfree(qs.pop(0))
    qs.extend(qalloc(2, "x"))
    print(qs)
    print(qtot())