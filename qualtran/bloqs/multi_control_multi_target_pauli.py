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
from functools import cached_property
from typing import Dict, Tuple

import cirq
from attrs import frozen
from cirq_ft import MultiControlPauli as CirqMultiControlPauli

from qualtran import Bloq, CompositeBloq, Register, Signature
from qualtran.cirq_interop import CirqQuregT, decompose_from_cirq_op


@frozen
class MultiControlPauli(Bloq):
    """Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-1$
    clean ancillas using a multi-controlled `AND` gate.

    References:
        [Constructing Large Controlled Nots]
        (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)

    Args:
        cvs: Control values.
        target_gate: Pauli gate to apply to target register.
    """

    cvs: Tuple[Tuple[int, ...], ...]
    target_gate: Bloq

    @cached_property
    def signature(self) -> Signature:
        regs = [Register(f"ctrl{i}", bitsize=len(cv)) for i, cv in enumerate(self.cvs)]
        regs += [Register('trgt', bitsize=1)]
        return Signature(regs)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        controls = [cirq_quregs[f'ctrl{i}'].tolist() for i, _ in enumerate(self.cvs)]
        target = cirq_quregs['trgt'].tolist()
        gate_map = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}
        bloq_gate = self.target_gate.short_name()[0]
        return (
            CirqMultiControlPauli(self.cvs, target_gate=gate_map[bloq_gate]).on_registers(
                controls=controls, target=target
            ),
            cirq_quregs,
        )
