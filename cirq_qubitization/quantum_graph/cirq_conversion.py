from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import cirq
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

import cirq_qubitization.cirq_infra.qubit_manager as cqm
from cirq_qubitization import GateWithRegisters
from cirq_qubitization.cirq_infra import Register as LegacyRegister
from cirq_qubitization.cirq_infra import Registers as LegacyRegisters
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _binst_to_cxns,
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

CirqQuregT = NDArray[cirq.Qid]


@frozen
class CirqGateAsBloq(Bloq):
    """A Bloq wrapper around a `cirq.Gate`.

    This bloq has one thru-register named "qubits", which is a 1D array of soquets
    representing individual qubits.
    """

    gate: cirq.Gate

    def pretty_name(self) -> str:
        return f'cirq.{self.gate}'

    def short_name(self) -> str:
        g = min(self.gate.__class__.__name__, str(self.gate), key=len)
        return f'cirq.{g}'

    @cached_property
    def n_qubits(self):
        return cirq.num_qubits(self.gate)

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('qubits', 1, wireshape=(self.n_qubits,))])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        unitary = cirq.unitary(self.gate).reshape((2,) * 2 * self.n_qubits)

        tn.add(
            qtn.Tensor(
                data=unitary,
                inds=outgoing['qubits'].tolist() + incoming['qubits'].tolist(),
                tags=[self.short_name(), tag],
            )
        )

    def as_cirq_op(self, cirq_quregs: Dict[str, 'NDArray[cirq.Qid]']) -> 'cirq.Operation':
        qubits = cirq_quregs['qubits']
        assert qubits.shape == (self.n_qubits, 1)
        return self.gate.on(*cirq_quregs['qubits'][:, 0])


def cirq_circuit_to_cbloq(circuit: cirq.Circuit) -> CompositeBloq:
    """Convert a Cirq circuit into a `CompositeBloq`.

    Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper. The
    resultant composite bloq will represent a unitary with one thru-register
    named "qubits" of wireshape `(n_qubits,)`.
    """
    bb = CompositeBloqBuilder()

    # "qubits" means cirq qubits | "qvars" means bloq Soquets
    all_qubits = sorted(circuit.all_qubits())
    all_qvars = bb.add_register(FancyRegister('qubits', 1, wireshape=(len(all_qubits),)))
    qubit_to_qvar = dict(zip(all_qubits, all_qvars))

    for op in circuit.all_operations():
        if op.gate is None:
            raise ValueError(f"Only gate operations are supported, not {op}.")

        bloq = CirqGateAsBloq(op.gate)
        qvars = np.array([qubit_to_qvar[qubit] for qubit in op.qubits])
        (out_qvars,) = bb.add(bloq, qubits=qvars)
        qubit_to_qvar |= zip(op.qubits, out_qvars)

    qvars = np.array([qubit_to_qvar[qubit] for qubit in all_qubits])
    return bb.finalize(qubits=qvars)


def _get_in_cirq_quregs(
    binst: BloqInstance, reg: FancyRegister, soq_assign: Dict[Soquet, 'NDArray[cirq.Qid]']
) -> 'NDArray[cirq.Qid]':
    full_shape = reg.wireshape + (reg.bitsize,)
    arg = np.empty(full_shape, dtype=object)

    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_cirq_quregs(
    regs: Iterable[FancyRegister],
    binst: BloqInstance,
    cirq_quregs: Dict[str, CirqQuregT],
    soq_assign: Dict[Soquet, CirqQuregT],
):
    """Update `soq_assign` using `vals`.
    This helper function is responsible for error checking. We use `regs` to make sure all the
    keys are present in the vals dictionary. We check the classical value shapes, types, and
    ranges.
    """
    unprocessed_reg_names = set(cirq_quregs.keys())
    for reg in regs:
        try:
            arr = cirq_quregs[reg.name]
        except KeyError:
            raise ValueError(f"{binst} requires an input register named {reg.name}")
        unprocessed_reg_names.remove(reg.name)

        arr = np.asarray(arr)
        full_shape = reg.wireshape + (reg.bitsize,)
        if arr.shape != full_shape:
            raise ValueError(f"Incorrect shape {arr.shape} received for {binst}.{reg.name}")

        for idx in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=idx)
            soq_assign[soq] = arr[idx]

    if unprocessed_reg_names:
        raise ValueError(f"{binst} had extra cirq_quregs: {unprocessed_reg_names}")


def _binst_as_cirq_op(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, NDArray[cirq.Qid]],
) -> cirq.Operation:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soq_assign: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]
        del soq_assign[cxn.left]

    def _in_vals(reg: FancyRegister) -> CirqQuregT:
        # close over `binst` and `soq_assign`.
        return _get_in_cirq_quregs(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    cirq_quregs = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}

    # as_cirq_op mutates `vals`.
    op = bloq.as_cirq_op(cirq_quregs)
    _update_assign_from_cirq_quregs(bloq.registers.rights(), binst, cirq_quregs, soq_assign)
    return op


def _cbloq_to_cirq_circuit(
    registers: FancyRegisters, cirq_quregs: Dict[str, CirqQuregT], binst_graph: nx.DiGraph
) -> cirq.FrozenCircuit:
    """Transform CompositeBloq components into a `cirq.Circuit`.

    Args:
        cirq_quregs: Assignment from each register to an array of `cirq.Qid` for the conversion
            to a `cirq.Circuit`.
        binst_graph: A graph connecting bloq instances with edge attributes containing the
            full list of `Connection`s, as returned by `CompositeBloq._get_binst_graph()`.
            This function does not mutate `binst_graph`.

    Returns:
        A `cirq.FrozenCircuit` for the composite bloq.
    """
    soq_assign: Dict[Soquet, CirqQuregT] = {}
    _update_assign_from_cirq_quregs(registers.lefts(), LeftDangle, cirq_quregs, soq_assign)

    moments: List[cirq.Moment] = []
    for binsts in nx.topological_generations(binst_graph):
        moment: List[cirq.Operation] = []

        for binst in binsts:
            if isinstance(binst, DanglingT):
                continue

            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            op = _binst_as_cirq_op(binst, pred_cxns, soq_assign)
            if op:
                moment.append(op)
        if moment:
            moments.append(cirq.Moment(moment))

    # Track bloq-to-dangle name changes
    if len(list(registers.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_quregs(reg: FancyRegister):
        return _get_in_cirq_quregs(RightDangle, reg, soq_assign)

    cirq_quregs |= {reg.name: _f_quregs(reg) for reg in registers.rights()}

    return cirq.FrozenCircuit(moments)


class BloqAsCirqGate(GateWithRegisters):
    """A shim for using bloqs in a Cirq circuit.

    ...
    """

    def __init__(
        self, bloq: Bloq, reg_to_wires: Optional[Callable[[FancyRegister], List[str]]] = None
    ):
        self._bloq = bloq
        self._legacy_regs, self._compat_name_map = self._init_legacy_regs(bloq)
        self._reg_to_wires = reg_to_wires

    @property
    def bloq(self) -> Bloq:
        return self._bloq

    @property
    def registers(self) -> LegacyRegisters:
        return self._legacy_regs

    @staticmethod
    def _init_legacy_regs(bloq: Bloq):
        legacy_regs: List[LegacyRegister] = []
        side_suffixes = {Side.LEFT: '_l', Side.RIGHT: '_r', Side.THRU: ''}
        compat_name_map = {}
        for reg in bloq.registers:

            if not reg.wireshape:
                compat_name = f'{reg.name}{side_suffixes[reg.side]}'
                compat_name_map[compat_name] = (reg, ())
                legacy_regs.append(LegacyRegister(name=compat_name, bitsize=reg.bitsize))
                continue

            for idx in reg.wire_idxs():
                idx_str = '_'.join(str(i) for i in idx)
                compat_name = f'{reg.name}{side_suffixes[reg.side]}_{idx_str}'
                compat_name_map[compat_name] = (reg, idx)
                legacy_regs.append(LegacyRegister(name=compat_name, bitsize=reg.bitsize))

        return LegacyRegisters(legacy_regs), compat_name_map

    @classmethod
    def make_from_bloq_on_registers(
        cls, bloq: Bloq, cirq_quregs: Dict[str, 'NDArray[cirq.Qid]']
    ) -> 'cirq.Operation':

        qubits: List[cirq.Qid] = []
        for reg in bloq.registers:
            if reg.side is Side.THRU:
                for i, q in enumerate(cirq_quregs[reg.name].reshape(-1)):
                    qubits.append(q)
            elif reg.side is Side.LEFT:
                for i, q in enumerate(cirq_quregs[reg.name].reshape(-1)):
                    qubits.append(q)
                    cqm.qfree(q)
            elif reg.side is Side.RIGHT:
                new_qubits = cqm.qalloc(reg.total_bits())
                qubits.extend(new_qubits)
                cirq_quregs[reg.name] = np.array(new_qubits).reshape(reg.wireshape + (reg.bitsize,))

        return BloqAsCirqGate(bloq=bloq).on(*qubits)

    def decompose_from_registers(self, **cirq_quregs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        cbloq = self._bloq.decompose_bloq()

        # initialize wireshape regs
        qubit_regs = {}
        for reg in self._bloq.registers:
            if reg.wireshape:
                shape = reg.wireshape + (reg.bitsize,)
                qubit_regs[reg.name] = np.empty(shape, dtype=object)

        # unflatten
        for compat_name, qubits in cirq_quregs.items():
            reg, idx = self._compat_name_map[compat_name]
            if idx == ():
                qubit_regs[reg.name] = qubits
            else:
                qubit_regs[reg.name][idx] = qubits

        return cbloq.to_cirq_circuit(qubit_regs)

    def _t_complexity_(self):
        return self._bloq.t_complexity()

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Default diagram info that uses register names to name the boxes in multi-qubit gates.

        Descendants can override this method with more meaningful circuit diagram information.
        """

        if self._reg_to_wires is not None:
            reg_to_wires = self._reg_to_wires
        else:
            reg_to_wires = lambda reg: [reg.name] * reg.total_bits()

        wire_symbols = []
        for reg in self._bloq.registers:
            symbs = reg_to_wires(reg)
            assert len(symbs) == reg.total_bits()
            wire_symbols.extend(symbs)

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
