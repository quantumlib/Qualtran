import itertools
from functools import cached_property
from typing import Any, Dict, Sequence, Tuple

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.algos.basic_gates.cnot import COPY, XOR
from cirq_qubitization.algos.basic_gates.zero_state import ZERO
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.util_bloqs import Partition, Unpartition


@frozen
class And(Bloq):
    """A two-bit and operation.

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:
     - control: A two-bit control register.
     - (right) target: The output bit.
    """

    cv1: int = 1
    cv2: int = 1
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('ctrl', 1, wireshape=(2,)),
                FancyRegister('target', 1, side=Side.RIGHT if not self.adjoint else Side.LEFT),
            ]
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):

        # Fill in our tensor using "and" logic.
        data = np.zeros((2, 2, 2, 2, 2))
        for c1, c2 in itertools.product((0, 1), repeat=2):
            if c1 == self.cv1 and c2 == self.cv2:
                data[c1, c2, c1, c2, 1] = 1
            else:
                data[c1, c2, c1, c2, 0] = 1

        # Here: adjoint just switches the direction of the target index.
        if self.adjoint:
            trg = incoming['target']
        else:
            trg = outgoing['target']

        tn.add(
            qtn.Tensor(
                data=data,
                inds=(
                    incoming['ctrl'][0],
                    incoming['ctrl'][1],
                    outgoing['ctrl'][0],
                    outgoing['ctrl'][1],
                    trg,
                ),
                tags=['And', tag],
            )
        )
