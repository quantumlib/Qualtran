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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cirq
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    Connection,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    LeftDangle,
    Register,
    RightDangle,
    Side,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns, _reg_to_soq
from qualtran._infra.gate_with_registers import (
    _get_all_and_output_quregs_from_input,
    merge_qubits,
    split_qubits,
    total_bits,
)
from qualtran.cirq_interop._cirq_to_bloq import _QReg, CirqQuregInT, CirqQuregT
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager
from qualtran.drawing import Circle, LarrowTextBox, ModPlus, RarrowTextBox, TextBox, WireSymbol


def _cirq_style_decompose_from_decompose_bloq(
    bloq: Bloq, quregs, context: cirq.DecompositionContext
) -> cirq.Circuit:
    """Helper function to implement cirq-style `_decompose_with_context_` that relies
    on `Bloq.decompose_bloq()`
    """
    cbloq = bloq.decompose_bloq()
    in_quregs = {reg.name: quregs[reg.name] for reg in bloq.signature.lefts()}
    # Input qubits can get de-allocated by cbloq.to_cirq_circuit, thus mark them as managed.
    qm = InteropQubitManager(context.qubit_manager)
    qm.manage_qubits(merge_qubits(bloq.signature.lefts(), **in_quregs))
    circuit, out_quregs = cbloq.to_cirq_circuit(qubit_manager=qm, **in_quregs)
    qubit_map = {q: q for q in circuit.all_qubits()}
    for reg in bloq.signature.rights():
        if reg.side == Side.RIGHT:
            # Right only registers can get mapped to newly allocated output qubits in `out_regs`.
            # Map them back to the original system qubits and deallocate newly allocated qubits.
            assert reg.name in quregs and reg.name in out_quregs
            assert quregs[reg.name].shape == out_quregs[reg.name].shape
            qm.qfree([q for q in out_quregs[reg.name].flatten()])
            qubit_map |= zip(out_quregs[reg.name].flatten(), quregs[reg.name].flatten())
    return circuit.unfreeze(copy=False).transform_qubits(qubit_map)


class BloqAsCirqGate(cirq.Gate):
    """A shim for using bloqs in a Cirq circuit.

    Args:
        bloq: The bloq to wrap.
    """

    def __init__(self, bloq: Bloq):
        for _, regs in bloq.signature.groups():
            if len(regs) > 1:
                raise ValueError(
                    f"Automated cirq conversion doesn't support multiple registers with same name."
                    f" Found {regs}\n. Please override `bloq.as_cirq_op` for `{bloq=}` instead."
                )
        self._bloq = bloq

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
        all_quregs, out_quregs = _get_all_and_output_quregs_from_input(
            bloq.signature, qubit_manager, in_quregs=cirq_quregs
        )
        return BloqAsCirqGate(bloq=bloq).on_registers(**all_quregs), out_quregs

    def _t_complexity_(self):
        """Delegate to the bloq's t complexity."""
        return self._bloq.t_complexity()

    def _num_qubits_(self) -> int:
        return total_bits(self.signature)

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        quregs = split_qubits(self.signature, qubits)
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        try:
            return _cirq_style_decompose_from_decompose_bloq(
                bloq=self.bloq, quregs=quregs, context=context
            )
        except (DecomposeNotImplementedError, DecomposeTypeError):
            pass
        return NotImplemented

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return self._decompose_with_context_(qubits)

    def _unitary_(self):
        if self._num_qubits_() > 3:
            # Prefer decomposition for large bloqs
            return NotImplemented
        tensor = self.bloq.tensor_contract()
        if tensor.ndim != 2:
            return NotImplemented
        return tensor

    def on_registers(
        self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]]
    ) -> cirq.Operation:
        return self.on(*merge_qubits(self.signature, **qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Draw cirq diagrams.

        By default, we label each qubit with its register name. If `reg_to_wires` was provided
        in the class constructor, we use that to get a list of wire symbols for each register.
        """
        return _wire_symbol_to_cirq_diagram_info(self._bloq, args)

    def __pow__(self, power, modulo=None):
        if power == 1:
            return self
        if power == -1:
            return BloqAsCirqGate(self.bloq.adjoint())
        raise ValueError(f"Bad power: {power}")

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


def _wire_symbol_to_cirq_diagram_info(
    bloq: Bloq, args: cirq.CircuitDiagramInfoArgs
) -> cirq.CircuitDiagramInfo:
    wire_symbols = []
    for reg in bloq.signature:
        # Note: all of our soqs lack a `binst`. The `bloq.wire_symbol` methods
        # should never use the binst field, but this isn't really enforced anywhere.
        # https://github.com/quantumlib/Qualtran/issues/608
        soqs = _reg_to_soq(None, reg)
        if isinstance(soqs, Soquet):
            assert soqs.idx == ()
            soqs = np.array([soqs] * reg.bitsize)
        else:
            soqs = np.broadcast_to(soqs, (reg.bitsize,) + reg.shape)

        for soq in soqs.reshape(-1):
            wire_symbols.append(bloq.wire_symbol(soq))

    def _qualtran_wire_symbols_to_cirq_text(ws: WireSymbol) -> str:
        if isinstance(ws, Circle):
            if ws.filled:
                return '@'
            else:
                return '@(0)'
        if isinstance(ws, (TextBox, RarrowTextBox, LarrowTextBox)):
            return ws.text
        if isinstance(ws, ModPlus):
            return 'X'
        raise NotImplementedError(f"Unknown cirq version of {ws}")

    wire_symbols = [_qualtran_wire_symbols_to_cirq_text(ws) for ws in wire_symbols]
    return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
