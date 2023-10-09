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
from typing import Iterable

import cirq


class InteropQubitManager(cirq.SimpleQubitManager):
    """Qubit Manager to use to facilitate interop of Cirq gates and Bloqs."""

    def __init__(self):
        super().__init__()
        self._managed_qubits = set()

    def manage_qubits(self, qubits: Iterable[cirq.Qid]):
        self._managed_qubits |= set(qubits)

    def qfree(self, qubits: Iterable[cirq.Qid]):
        qs = set(qubits)
        managed_qs = qs & self._managed_qubits
        qs -= managed_qs
        self._managed_qubits -= managed_qs
        super().qfree(qs)
