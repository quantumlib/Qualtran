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

import itertools
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    Register,
    Signature,
    SoquetT,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, ModPlus, Text, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT

COPY = [1, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY, dtype=np.complex128).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2)).astype(np.complex128)


@frozen
class CNOT(Bloq):
    """Two-qubit controlled-NOT.

    Registers:
        ctrl: One-bit control register.
        target: One-bit target register.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'Bloq':
        return self

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        # This bloq uses the factored form of CNOT composed of a COPY and XOR tensor joined
        # by an internal index.
        # [Lectures on Quantum Tensor Networks](https://arxiv.org/abs/1912.10049). Biamonte 2019.
        import quimb.tensor as qtn

        internal = qtn.rand_uuid()
        return [
            qtn.Tensor(
                data=COPY,
                inds=[(incoming['ctrl'], 0), (outgoing['ctrl'], 0), internal],
                tags=['COPY'],
            ),
            qtn.Tensor(
                data=XOR,
                inds=[(incoming['target'], 0), (outgoing['target'], 0), internal],
                tags=['XOR'],
            ),
        ]

    def on_classical_vals(self, ctrl: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'ctrl': ctrl, 'target': (ctrl + target) % 2}

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.basic_gates.toffoli import Toffoli

        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        bloq = Toffoli()

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (new_ctrl,) = ctrl_soqs
            (new_ctrl, existing_ctrl), target = bb.add(
                bloq, ctrl=np.array([new_ctrl, in_soqs['ctrl']]), target=in_soqs['target']
            )
            return (new_ctrl,), (existing_ctrl, target)

        return bloq, add_controlled

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', ctrl: 'CirqQuregT', target: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        import cirq

        (ctrl,) = ctrl
        (target,) = target
        return cirq.CNOT(ctrl, target), {'ctrl': np.array([ctrl]), 'target': np.array([target])}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle(filled=True)
        elif reg.name == 'target':
            return ModPlus()
        raise ValueError(f'Unknown wire symbol register name: {reg.name}')

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)


@bloq_example
def _cnot() -> CNOT:
    cnot = CNOT()
    return cnot


_CNOT_DOC = BloqDocSpec(bloq_cls=CNOT, examples=[_cnot])
