from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence

import cirq
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _binst_to_cxns,
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    DanglingT,
    LeftDangle,
    Soquet,
)


@frozen
class CirqGate(Bloq):
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
    def registers(self) -> 'FancyRegisters':
        n_qubits = cirq.num_qubits(self.gate)
        return FancyRegisters([FancyRegister('qubits', 1, wireshape=(n_qubits,))])

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        n_qubits = cirq.num_qubits(self.gate)
        unitary = cirq.unitary(self.gate).reshape((2,) * 2 * n_qubits)

        tn.add(
            qtn.Tensor(
                data=unitary,
                inds=outgoing['qubits'].tolist() + incoming['qubits'].tolist(),
                tags=[self.short_name(), tag],
            )
        )

    def on_registers(self, qubits: Sequence[cirq.Qid]) -> cirq.Operation:
        return self.gate.on(*qubits)


def cirq_circuit_to_cbloq(circuit: cirq.Circuit) -> CompositeBloq:
    """Convert a Cirq circuit into a `CompositeBloq`.

    Each `cirq.Operation` will be wrapped into a `CirqGate` wrapper bloq. The
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

        bloq = CirqGate(op.gate)
        qvars = np.array([qubit_to_qvar[qubit] for qubit in op.qubits])
        (out_qvars,) = bb.add(bloq, qubits=qvars)
        qubit_to_qvar |= zip(op.qubits, out_qvars)

    qvars = np.array([qubit_to_qvar[qubit] for qubit in all_qubits])
    return bb.finalize(qubits=qvars)


def _process_binst(
    binst: BloqInstance, soqmap: Dict[Soquet, Sequence[cirq.Qid]], binst_graph: nx.DiGraph
) -> Optional[cirq.Operation]:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soqmap: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    if isinstance(binst, DanglingT):
        return None

    pred_cxns, _ = _binst_to_cxns(binst, binst_graph)

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soqmap[cxn.right] = soqmap[cxn.left]
        del soqmap[cxn.left]

    bloq = binst.bloq

    # Pull out the qubits from soqmap into qumap which has string keys.
    # This implicitly joins things with the same name.
    quregs: Dict[str, List[cirq.Qid]] = defaultdict(list)
    for reg in bloq.registers.lefts():
        for li in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=li)
            quregs[reg.name].extend(soqmap[soq])
            del soqmap[soq]

    op = bloq.on_registers(**quregs)

    # We pluck things back out from their collapsed by-name qumap into soqmap
    # This does implicit splitting.
    for reg in bloq.registers.rights():
        qarr = np.asarray(quregs[reg.name])
        for ri in reg.wire_idxs():
            soq = Soquet(binst, reg, idx=ri)
            qs = qarr[ri]
            if isinstance(qs, np.ndarray):
                qs = qs.tolist()
            else:
                qs = [qs]
            soqmap[soq] = qs

    return op


def _cbloq_to_cirq_circuit(
    quregs: Dict[FancyRegister, NDArray[cirq.Qid]], binst_graph: nx.DiGraph
) -> cirq.Circuit:
    """Transform CompositeBloq components into a `cirq.Circuit`.

    Args:
        quregs: Assignment from each register to a sequence of `cirq.Qid` for the conversion
            to a `cirq.Circuit`.
        binst_graph: A graph connecting bloq instances with edge attributes containing the
            full list of `Connection`s, as returned by `CompositeBloq._get_binst_graph()`.
            This function does not mutate `binst_graph`.

    Returns:
        A `cirq.Circuit` for the quantum compute graph.
    """
    # A mapping of soquet to qubits that we update as operations are appended to the circuit.
    soqmap = {}
    for reg in quregs.keys():
        qarr = np.asarray(quregs[reg])
        for ii in reg.wire_idxs():
            soqmap[Soquet(LeftDangle, reg, idx=ii)] = qarr[ii]

    moments: List[cirq.Moment] = []
    for i, binsts in enumerate(nx.topological_generations(binst_graph)):
        mom: List[cirq.Operation] = []
        for binst in binsts:
            op = _process_binst(binst, soqmap, binst_graph)
            if op:
                mom.append(op)
        if mom:
            moments.append(cirq.Moment(mom))

    return cirq.Circuit(moments)
