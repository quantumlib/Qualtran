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
import numpy as np
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
        cvs: Control values. Expect a tuple of tuples of control values, where
            each tuple of control values should be the same length as the number
            of bits in the corresponding control register. E.g. cvs = ((1, 1),
            (1,), (1,1,1)) would mean we have 3 control registers of sizes (2,
            1, 3). Currently we assume only control values of 1, a zero would
            signify anti controls.
        target_pauli: The name of the Pauli gate ("X", "Y", or "Z")
    """

    cvs: Tuple[Tuple[int, ...], ...]
    pauli_name: str

    def __attrs_post_init__(self):
        assert isinstance(self.pauli_name, str)
        assert self.pauli_name in ("X", "Y", "Z")

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
        controls = np.concatenate(
            [cirq_quregs[f'ctrl{i}'].tolist() for i, _ in enumerate(self.cvs)]
        )
        target = cirq_quregs['trgt'].tolist()
        gate_map = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}
        return (
            CirqMultiControlPauli(
                tuple(np.concatenate(self.cvs)), target_gate=gate_map[self.pauli_name]
            ).on_registers(controls=controls, target=target),
            cirq_quregs,
        )
