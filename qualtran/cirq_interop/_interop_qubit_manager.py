#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Qubit Manager to use when converting Cirq gates to/from Bloqs."""
from typing import Iterable, List, Optional, Set

import cirq


class InteropQubitManager(cirq.QubitManager):
    """Qubit Manager to use to facilitate interop of Cirq gates and Bloqs."""

    def __init__(self, qm: Optional[cirq.QubitManager] = None):
        if qm is None:
            qm = cirq.SimpleQubitManager()
        self._qm = qm
        self._managed_qubits: Set[cirq.Qid] = set()

    def qalloc(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        ret: List['cirq.Qid'] = []
        qubits_to_free: List['cirq.Qid'] = []
        while len(ret) < n:
            new_alloc = self._qm.qalloc(n - len(ret), dim)
            for q in new_alloc:
                if q in self._managed_qubits:
                    qubits_to_free.append(q)
                else:
                    ret.append(q)
        self._qm.qfree(qubits_to_free)
        return ret

    def qborrow(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        ret: List['cirq.Qid'] = []
        qubits_to_free: List['cirq.Qid'] = []
        while len(ret) < n:
            new_alloc = self._qm.qborrow(n - len(ret), dim)
            for q in new_alloc:
                if q in self._managed_qubits:
                    qubits_to_free.append(q)
                else:
                    ret.append(q)
        self._qm.qfree(qubits_to_free)
        return ret

    def manage_qubits(self, qubits: Iterable[cirq.Qid]):
        self._managed_qubits |= set(qubits)

    def qfree(self, qubits: Iterable[cirq.Qid]):
        managed_qs = set(qubits) & self._managed_qubits
        self._managed_qubits -= managed_qs
        self._qm.qfree([q for q in qubits if q not in managed_qs])
