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

"""Qualtran Bloqs to Cirq gates/circuits conversion."""

from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cirq
import networkx as nx
import numpy as np

from qualtran import (
    Bloq,
    CompositeBloq,
    Connection,
    GateWithRegisters,
    LeftDangle,
    Register,
    RightDangle,
    Side,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns
from qualtran.cirq_interop._cirq_to_bloq import _QReg, CirqQuregInT, CirqQuregT


class BloqAsCirqGate(GateWithRegisters):
    """A shim for using bloqs in a Cirq circuit.

    Args:
        bloq: The bloq to wrap.
        reg_to_wires: an optional callable to produce a list of wire symbols for each register
            to match Cirq diagrams.
    """

    def __init__(self, bloq: Bloq, reg_to_wires: Optional[Callable[[Register], List[str]]] = None):
        for _, regs in bloq.signature.groups():
            if len(regs) > 1:
                raise ValueError(
                    f"Automated cirq conversion doesn't support multiple registers with same name."
                    f" Found {regs}\n. Please override `bloq.as_cirq_op` for `{bloq=}` instead."
                )
        self._bloq = bloq
        self._reg_to_wires = reg_to_wires

    @property
    def bloq(self) -> Bloq:
        """The bloq we're wrapping."""
        return self._bloq

    @cached_property
    def signature(self) -> Signature:
        """`GateWithRegisters` registers."""
        return self.bloq.signature

    @classmethod
    def bloq_on(
        cls, bloq: Bloq, cirq_quregs: Dict[str, 'CirqQuregT'], qubit_manager: cirq.QubitManager
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        """Shim `bloq` into a cirq gate and call it on `cirq_quregs`.

        This is used as a default implementation for `Bloq.as_cirq_op` if a native
        cirq conversion is not specified.

        Args:
            bloq: The bloq to be wrapped with `BloqAsCirqGate`
            cirq_quregs: The cirq qubit registers on which we call the gate. Should correspond to
                registers in `self.bloq.signature.lefts()`.
            qubit_manager: A `cirq.QubitManager` to allocate new qubits.

        Returns:
            op: A cirq operation whose gate is the `BloqAsCirqGate`-wrapped version of `bloq`.
            cirq_quregs: The output cirq qubit registers.
        """
        return BloqAsCirqGate(bloq=bloq).as_cirq_op(qubit_manager=qubit_manager, **cirq_quregs)

    def decompose_bloq(self) -> 'CompositeBloq':
        return self.bloq.decompose_bloq()

    def _t_complexity_(self):
        """Delegate to the bloq's t complexity."""
        return self._bloq.t_complexity()

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Draw cirq diagrams.

        By default, we label each qubit with its register name. If `reg_to_wires` was provided
        in the class constructor, we use that to get a list of wire symbols for each register.
        """

        if self._reg_to_wires is not None:
            reg_to_wires = self._reg_to_wires
        else:
            reg_to_wires = lambda reg: [reg.name] * reg.total_bits()

        wire_symbols = []
        for reg in self._bloq.signature:
            symbs = reg_to_wires(reg)
            assert len(symbs) == reg.total_bits()
            wire_symbols.extend(symbs)
        if self._reg_to_wires is None:
            wire_symbols[0] = self._bloq.pretty_name()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __eq__(self, other):
        if not isinstance(other, BloqAsCirqGate):
            return False
        return self.bloq == other.bloq

    def __hash__(self):
        return hash(self.bloq)

    def __str__(self) -> str:
        return f'bloq.{self.bloq}'

    def __repr__(self) -> str:
        return f'BloqAsCirqGate({self.bloq})'


def _track_soq_name_changes(cxns: Iterable[Connection], qvar_to_qreg: Dict[Soquet, _QReg]):
    """Track inter-Bloq name changes across the two ends of a connection."""
    for cxn in cxns:
        qvar_to_qreg[cxn.right] = qvar_to_qreg[cxn.left]
        del qvar_to_qreg[cxn.left]


def _bloq_to_cirq_op(
    bloq: Bloq,
    pred_cxns: Iterable[Connection],
    succ_cxns: Iterable[Connection],
    qvar_to_qreg: Dict[Soquet, _QReg],
    qubit_manager: cirq.QubitManager,
) -> cirq.Operation:
    _track_soq_name_changes(pred_cxns, qvar_to_qreg)
    in_quregs: Dict[str, CirqQuregT] = {
        reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object)
        for reg in bloq.signature.lefts()
    }
    # Construct the cirq qubit registers using input / output connections.
    # 1. All input Soquets should already have the correct mapping in `qvar_to_qreg`.
    for cxn in pred_cxns:
        soq = cxn.right
        assert soq in qvar_to_qreg, f"{soq=} should exist in {qvar_to_qreg=}."
        in_quregs[soq.reg.name][soq.idx] = qvar_to_qreg[soq].qubits
        if soq.reg.side == Side.LEFT:
            # Remove soquets for LEFT registers from qvar_to_qreg mapping.
            del qvar_to_qreg[soq]

    op, out_quregs = bloq.as_cirq_op(qubit_manager=qubit_manager, **in_quregs)

    # 2. Update the mappings based on output soquets and `out_quregs`.
    for cxn in succ_cxns:
        soq = cxn.left
        assert soq.reg.name in out_quregs, f"{soq=} should exist in {out_quregs=}."
        if soq.reg.side == Side.RIGHT:
            qvar_to_qreg[soq] = _QReg(out_quregs[soq.reg.name][soq.idx])
    return op


def _cbloq_to_cirq_circuit(
    signature: Signature,
    cirq_quregs: Dict[str, 'CirqQuregInT'],
    binst_graph: nx.DiGraph,
    qubit_manager: cirq.QubitManager,
) -> Tuple[cirq.FrozenCircuit, Dict[str, 'CirqQuregT']]:
    """Propagate `as_cirq_op` calls through a composite bloq's contents to export a `cirq.Circuit`.

    Args:
        signature: The cbloq's signature for validating inputs and outputs.
        cirq_quregs: Mapping from left register name to Cirq qubit arrays.
        binst_graph: The cbloq's binst graph. This is read only.
        qubit_manager: A `cirq.QubitManager` to allocate new qubits.

    Returns:
        circuit: The cirq.FrozenCircuit version of this composite bloq.
        cirq_quregs: The output mapping from right register names to Cirq qubit arrays.
    """
    cirq_quregs = {k: np.apply_along_axis(_QReg, -1, v) for k, v in cirq_quregs.items()}
    qvar_to_qreg: Dict[Soquet, _QReg] = {
        Soquet(LeftDangle, idx=idx, reg=reg): cirq_quregs[reg.name][idx]
        for reg in signature.lefts()
        for idx in reg.all_idxs()
    }
    moments: List[cirq.Moment] = []
    for binsts in nx.topological_generations(binst_graph):
        moment: List[cirq.Operation] = []

        for binst in binsts:
            if binst is LeftDangle:
                continue
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            if binst is RightDangle:
                _track_soq_name_changes(pred_cxns, qvar_to_qreg)
                continue

            op = _bloq_to_cirq_op(binst.bloq, pred_cxns, succ_cxns, qvar_to_qreg, qubit_manager)
            if op is not None:
                moment.append(op)
        if moment:
            moments.append(cirq.Moment(moment))

    # Find output Cirq quregs using `qvar_to_qreg` mapping for registers in `signature.rights()`.
    def _f_quregs(reg: Register) -> CirqQuregT:
        ret = np.empty(reg.shape + (reg.bitsize,), dtype=object)
        for idx in reg.all_idxs():
            soq = Soquet(RightDangle, idx=idx, reg=reg)
            ret[idx] = qvar_to_qreg[soq].qubits
        return ret

    out_quregs = {reg.name: _f_quregs(reg) for reg in signature.rights()}

    return cirq.FrozenCircuit(moments), out_quregs
