#  Copyright 2024 Google LLC
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
from typing import Iterable

import numpy as np
import pyzx as zx
from attrs import define
from numpy.typing import NDArray

from qualtran import Bloq, CompositeBloq, LeftDangle, Register, RightDangle, Signature, Soquet

ZXQubitMap = dict[str, NDArray[np.integer]]
"""A mapping from register names to an NDArray of ZX qubits"""


@define
class ZXAncillaManager:
    n: int

    def allocate(self) -> int:
        idx = self.n
        self.n += 1
        return idx


def _empty_qubit_map_from_registers(registers: Iterable[Register]) -> ZXQubitMap:
    return {reg.name: np.empty(reg.shape + (reg.bitsize,), dtype=int) for reg in registers}


def _initalize_zx_circuit_from_signature(
    signature: Signature,
) -> tuple[zx.Circuit, ZXQubitMap, ZXAncillaManager]:
    n_qubits: int = 0
    qubit_d: ZXQubitMap = {}

    for reg in signature.lefts():
        n = reg.total_bits()
        idxs = np.arange(n) + n_qubits
        shape = reg.shape + (reg.bitsize,)
        qubit_d[reg.name] = idxs.reshape(shape)
        n_qubits += n

    circ = zx.Circuit(qubit_amount=n_qubits)
    return circ, qubit_d, ZXAncillaManager(n_qubits)


def _add_bloq_to_pyzx_circuit(
    circ: zx.Circuit, bloq: Bloq, ancilla_manager: ZXAncillaManager, in_qubits: ZXQubitMap
) -> ZXQubitMap:
    try:
        gates, out_qubits = bloq.as_zx_gates(ancilla_manager, **in_qubits)
        for gate in gates:
            circ.add_gate(gate)
        return out_qubits
    except NotImplementedError:
        pass

    cbloq = bloq.decompose_bloq()
    return _add_cbloq_to_pyzx_circuit(circ, cbloq, ancilla_manager, in_qubits)


def _add_cbloq_to_pyzx_circuit(
    circ: zx.Circuit, cbloq: CompositeBloq, ancilla_manager: ZXAncillaManager, in_qubits: ZXQubitMap
) -> ZXQubitMap:
    soq_map: dict[Soquet, NDArray[np.integer]] = {
        Soquet(binst=LeftDangle, reg=reg, idx=idx): in_qubits[reg.name][idx]
        for reg in cbloq.signature.lefts()
        for idx in reg.all_idxs()
    }

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq

        # compute the input qubits
        bloq_in_qubits: ZXQubitMap = _empty_qubit_map_from_registers(bloq.signature.lefts())
        for cxn in pred_cxns:
            bloq_soq = cxn.right
            bloq_in_qubits[bloq_soq.reg.name][bloq_soq.idx] = soq_map.pop(cxn.left)

        out_qubits: ZXQubitMap = _add_bloq_to_pyzx_circuit(
            circ, bloq, ancilla_manager, bloq_in_qubits
        )

        # forward the output qubits to their corresponding soqs
        for cxn in succ_cxns:
            bloq_soq = cxn.left
            soq_map[bloq_soq] = out_qubits[bloq_soq.reg.name][bloq_soq.idx]

    # forward the soqs to the cbloq output soqs
    for cxn in cbloq.connections:
        if cxn.right.binst is RightDangle:
            soq_map[cxn.right] = soq_map.pop(cxn.left)

    # get the output qubits
    out_qubits = _empty_qubit_map_from_registers(cbloq.signature.rights())
    for soq, qubits in soq_map.items():
        out_qubits[soq.reg.name][soq.idx] = qubits
    return out_qubits


def bloq_to_pyzx_circuit(bloq: Bloq) -> zx.Circuit:
    circ, in_qubits, ancilla_manager = _initalize_zx_circuit_from_signature(bloq.signature)
    _ = _add_bloq_to_pyzx_circuit(circ, bloq, ancilla_manager, in_qubits)
    return circ
