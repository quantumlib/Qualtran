import cirq
from typing import List, Optional

_free_qubits = {}
_used_qubits = {}

def _allocate(name: Optional[str] = None, reuse: bool = True):
    global _free_qubits, _used_qubits

    name = name or f'ancilla{len(_free_qubits) + len(_used_qubits)}'
    if name in _used_qubits:
        raise ValueError(f'a qubit with name {name} already exists')

    if reuse and len(_free_qubits) and (name not in _free_qubits):
        # if reuse is enabled then reuse any of the free qubits.
        name = next(iter(_free_qubits))

    if name in _free_qubits:
        q = _free_qubits[name]
        del _free_qubits[name]
        _used_qubits[name] = q
        return q

    q = cirq.NamedQubit(name)
    _used_qubits[name] = q
    return q

def _free(q: cirq.NamedQubit):
    global _free_qubits, _used_qubits
    del _used_qubits[q.name]
    _free_qubits[q.name] = q

def qalloc(n: int = 1, prefix: Optional[str] = None, reuse: bool = True) -> List[cirq.NamedQubit]:
    name = lambda i: prefix + str(i) if prefix else None
    if n == 1: name = lambda i: prefix if prefix else None

    qubits = [_allocate(name(i), reuse=reuse) for i in range(n)]
    return qubits

def qfree(q: cirq.NamedQubit) -> None:
    _free(q)

def qtot() -> int:
    global _free_qubits, _used_qubits
    return len(_free_qubits) + len(_used_qubits)

if __name__ == "__main__":
    qs = qalloc(2)
    print(qs)
    qfree(qs.pop(0))
    qs.extend(qalloc(2, "x"))
    print(qs)
    print(qtot()) 
    qfree(qs.pop(0))
    qs.extend(qalloc(2, prefix='noreuse', reuse=False))
    print(qs)
    print(qtot()) 
