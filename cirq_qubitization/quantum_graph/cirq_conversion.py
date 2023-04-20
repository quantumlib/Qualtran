from functools import cached_property
from typing import Any, Dict, Sequence

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters


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
