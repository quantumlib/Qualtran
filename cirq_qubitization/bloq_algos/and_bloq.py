import itertools
from functools import cached_property
from typing import Any, Dict, Tuple

import numpy as np
import quimb.tensor as qtn
from attrs import field, frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


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

    def apply_classical(self, ctrl):
        # (bitsize,)
        # (wx1, wx2, bitsize)
        assert ctrl.shape == (2, 1)
        c1, c2 = ctrl[:, 0]
        if c1 == self.cv1 and c2 == self.cv2:
            target = np.array([1])
        else:
            target = np.array([0])
        return ctrl, target

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):

        # Fill in our tensor using "and" logic.
        data = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
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


@frozen
class MultiAnd(Bloq):
    cvs: Tuple[int, ...] = field(validator=lambda i, f, v: len(v) >= 3)
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('ctrl', 1, wireshape=(len(self.cvs),)),
                FancyRegister('junk', 1, wireshape=(len(self.cvs) - 2,), side=Side.RIGHT),
                FancyRegister('target', 1, side=Side.RIGHT if not self.adjoint else Side.LEFT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, ctrl: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""
        c1c2, anc = bb.add(
            And(cv1=self.cvs[0], cv2=self.cvs[1], adjoint=self.adjoint), ctrl=ctrl[:2]
        )
        if len(self.cvs) == 3:
            # Base case: add a final `And`.
            (c3, junk), target = bb.add(
                And(cv1=1, cv2=self.cvs[2], adjoint=self.adjoint), ctrl=[anc, ctrl[2]]
            )
            return {
                'ctrl': np.asarray(c1c2.tolist() + [c3]),
                'junk': np.asarray([junk]),
                'target': target,
            }

        # Note: change from `add_from` to `add` for non-auto-decomposing.
        (anc, *c_rest), junk, target = bb.add_from(
            MultiAnd(cvs=(1, *self.cvs[2:]), adjoint=self.adjoint), ctrl=[anc] + ctrl[2:].tolist()
        )
        return {
            'ctrl': np.asarray(c1c2.tolist() + c_rest),
            'junk': np.asarray([anc] + junk.tolist()),
            'target': target,
        }

    def apply_classical(self, ctrl):
        target = True
        for cv, c in zip(self.cvs, ctrl[:, 0]):
            target = target and (c == cv)

        junk = np.zeros(len(self.cvs) - 2)
        target = [1] if target else [0]
        target = np.array(target)

        return ctrl, junk, target
