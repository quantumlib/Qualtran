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
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Identity(Bloq):
    r"""The identity gate on `n` qubits.

    Args:
        bitsize: number of qubits `n`, defaults to 1.

    Registers:
        q: register of `n` qubits
    """
    bitsize: SymbolicInt = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=self.bitsize)

    def adjoint(self) -> 'Bloq':
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=np.eye(2), inds=[(outgoing['q'], i), (incoming['q'], i)], tags=[str(self)]
            )
            for i in range(int(self.bitsize))
        ]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        import cirq

        if is_symbolic(self.bitsize):
            raise ValueError(f"cirq.IdentityGate does not support symbolic {self.bitsize=}")

        return cirq.IdentityGate(self.bitsize).on(*q), {'q': q}

    def _t_complexity_(self):
        return TComplexity()

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox('I')

    def on_classical_vals(self, q: int) -> Dict[str, 'ClassicalValT']:
        return {'q': q}

    def __str__(self) -> str:
        return 'I'

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        ctrl_I = Identity(ctrl_spec.num_qubits + self.bitsize)

        def ctrl_adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            parts = [
                (Register(f'ctrl_{i}', dtype=dtype, shape=shape), 'q')
                for i, (dtype, shape) in enumerate(ctrl_spec.activation_function_dtypes())
            ] + [(reg, 'q') for reg in self.signature]
            all_soqs = in_soqs | {f'ctrl_{i}': ctrl_soq for i, ctrl_soq in enumerate(ctrl_soqs)}
            out_soqs = bb.add_and_partition(ctrl_I, partitions=parts, left_only=False, **all_soqs)
            return out_soqs[:-1], out_soqs[-1:]

        return ctrl_I, ctrl_adder


@bloq_example
def _identity() -> Identity:
    identity = Identity()
    return identity


@bloq_example
def _identity_n() -> Identity:
    n = 4
    identity_n = Identity(n)
    return identity_n


@bloq_example
def _identity_symb() -> Identity:
    import sympy

    n = sympy.Symbol("n")
    identity_symb = Identity(n)
    return identity_symb


_IDENTITY_DOC = BloqDocSpec(bloq_cls=Identity, examples=[_identity_symb, _identity, _identity_n])
