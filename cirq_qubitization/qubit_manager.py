from collections import defaultdict
from glob import glob
import cirq
from typing import List

_free_qubits = {}
_used_qubits = {}

_qubit_count = defaultdict(lambda: 0)

def _next_name(prefix: str, need_index: bool) -> str:
    global _qubit_count
    # Don't add an index if we don't need to.
    # This makes the call qalloc(1, name, False) equivalent to cirq.NamedQubit(name).
    if _qubit_count[prefix] == 0 and not need_index: 
        return prefix
    return f'{prefix}{_qubit_count[prefix]}'

def _allocate(prefix: str, need_index: bool, reuse: bool = True):
    global _free_qubits, _used_qubits

    name = _next_name(prefix, need_index)
    if reuse and len(_free_qubits) and (name not in _free_qubits):
        # if reuse is enabled then reuse any of the free qubits.
        name = next(iter(_free_qubits))

    if name in _free_qubits:
        q = _free_qubits[name]
        del _free_qubits[name]
        _used_qubits[name] = q
        return q

    q = cirq.NamedQubit(name)
    _qubit_count[prefix] += 1
    _used_qubits[name] = q
    return q

def _free(q: cirq.NamedQubit):
    global _free_qubits, _used_qubits
    del _used_qubits[q.name]
    _free_qubits[q.name] = q

def qalloc(n: int = 1, prefix: str = 'ancilla', reuse: bool = True) -> List[cirq.NamedQubit]:
    qubits = [_allocate(prefix, n > 1, reuse=reuse) for i in range(n)]
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
    qs.extend(qalloc(1, prefix='y', reuse=False))
    print(qs)
    print(qtot()) 
