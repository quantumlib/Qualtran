import cirq
from typing import List, Optional

_free_qubits = []
_total_allocated = 0

def _allocate(name: Optional[str] = None, reuse: bool = True):
    global _free_qubits, _total_allocated
    if not (reuse and len(_free_qubits)):
        name = name or f'q{_total_allocated}'
        _free_qubits.append(cirq.NamedQubit(name))
        _total_allocated += 1
    return _free_qubits.pop()

def _free(q: cirq.NamedQubit):
    global _free_qubits
    _free_qubits.append(q)

def qalloc(n: int = 1, prefix: str = "", reuse: bool = True) -> List[cirq.NamedQubit]:
    name = lambda i: prefix + str(i)
    if n == 1: name = lambda i: prefix
    qubits = [_allocate(name(i), reuse=reuse) for i in range(n)]
    if reuse:
        for i in range(n):
            qubits[i]._name = name(i)
    return qubits

def qfree(q: cirq.NamedQubit) -> None:
    _free(q)


def qtot() -> int:
    global _total_allocated
    return _total_allocated

if __name__ == "__main__":
    qs = qalloc(2, "test")
    print(qs)
    qfree(qs.pop(0))
    qs.extend(qalloc(2, "x"))
    print(qs)
    print(qtot()) 
    qs.extend(qalloc(2, prefix='noreuse', reuse=False))
    print(qs)
    print(qtot()) 
