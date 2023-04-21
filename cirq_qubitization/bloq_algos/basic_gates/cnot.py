import itertools
from functools import cached_property
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters

if TYPE_CHECKING:
    import cirq

COPY = [1, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY, dtype=np.complex128).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2)).astype(np.complex128)


@frozen
class CNOT(Bloq):
    """Two-qubit controlled-NOT.

    Registers:
     - ctrl: One-bit control register.
     - target: One-bit target register.
    """

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, target=1)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        """Append tensors to `tn` that represent this operation.

        This bloq uses the factored form of CNOT composed of a COPY and XOR tensor joined
        by an internal index.

        References:
            [Lectures on Quantum Tensor Networks](https://arxiv.org/abs/1912.10049). Biamonte 2019.
        """
        internal = qtn.rand_uuid()
        tn.add(
            qtn.Tensor(
                data=COPY, inds=(incoming['ctrl'], outgoing['ctrl'], internal), tags=['COPY', tag]
            )
        )
        tn.add(
            qtn.Tensor(
                data=XOR, inds=(incoming['target'], outgoing['target'], internal), tags=['XOR']
            )
        )

    def on_classical_vals(self, ctrl: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'ctrl': ctrl, 'target': (ctrl + target) % 2}

    def on_registers(
        self, ctrl: 'NDArray[cirq.Qid]', target: 'NDArray[cirq.Qid]'
    ) -> 'cirq.OP_TREE':
        import cirq

        (ctrl,) = ctrl
        (target,) = target
        return cirq.CNOT(ctrl, target)
