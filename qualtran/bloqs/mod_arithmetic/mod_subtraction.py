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
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt import And, MultiControlX, MultiTargetCNOT
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import HasLength

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


@frozen
class ModNeg(Bloq):
    r"""Performs modular negation.

    Applies the operation $\ket{x} \rightarrow \ket{-x \% p}$

    Note: This implements the decomposition from Fig 6 in but doesn't match table 8
    since we don't use measurement based uncompution because that introduces random phase flips.

    Args:
        dtype: Datatype of the register.
        p: The modulus for the negation.

    Registers:
        x: The register contraining the integer we negate.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6b
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', self.dtype)])

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet) -> Dict[str, 'SoquetT']:
        if not isinstance(self.dtype.bitsize, int):
            raise ValueError(f'symbolic decomposition is not supported for {self}')

        ancilla = bb.allocate(1)
        ancilla = bb.add(XGate(), q=ancilla)

        x_arr = bb.split(x)
        x_arr, ancilla = bb.add(
            MultiControlX(cvs=[0] * self.dtype.bitsize), controls=x_arr, target=ancilla
        )
        x = bb.join(x_arr)

        ancilla, x = bb.add(MultiTargetCNOT(self.dtype.bitsize), control=ancilla, targets=x)
        (ancilla,), x = bb.add(
            AddK(self.dtype.bitsize, self.mod + 1, cvs=(1,), signed=False), ctrls=(ancilla,), x=x
        )

        x_arr = bb.split(x)
        x_arr, ancilla = bb.add(
            MultiControlX(cvs=[0] * self.dtype.bitsize).adjoint(), controls=x_arr, target=ancilla
        )
        x = bb.join(x_arr)

        ancilla = bb.add(XGate(), q=ancilla)
        bb.free(ancilla)
        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cvs: Union[list[int], HasLength]
        if isinstance(self.dtype.bitsize, int):
            cvs = [0] * self.dtype.bitsize
        else:
            cvs = HasLength(self.dtype.bitsize)
        return {
            (MultiControlX(cvs), 1),
            (MultiControlX(cvs).adjoint(), 1),
            (MultiTargetCNOT(self.dtype.bitsize), 1),
            (AddK(self.dtype.bitsize, k=self.mod + 1, cvs=(1), signed=False), 1),
            (XGate(), 2),
        }

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox('$-x$')
        raise ValueError(f'Unrecognized register name {reg.name}')

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if 0 < x < self.mod:
            x = self.mod - x
        return {'x': x}


@frozen
class CModNeg(Bloq):
    r"""Performs controlled modular negation.

    Applies the operation $\ket{c}\ket{x} \rightarrow \ket{c}\ket{(-1)^c x\%p}$

    Note: while this matches the count from Fig 8, it's a different decomposition that controls
    only the Add operation instead of turning the CNOTs into toffolis.

    Args:
        dtype: Datatype of the register.
        p: The modulus for the negation.
        cv: value at which the gate is active.

    Registers:
        ctrl: Control bit.
        x: The register contraining the integer we negate.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6b and 8
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    mod: 'SymbolicInt'
    cv: int = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', self.dtype)])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: Soquet, x: Soquet
    ) -> Dict[str, 'SoquetT']:
        if not isinstance(self.dtype.bitsize, int):
            raise ValueError(f'symbolic decomposition is not supported for {self}')

        ancilla = bb.allocate(1)
        ancilla = bb.add(XGate(), q=ancilla)

        x_arr = bb.split(x)
        x_arr, ancilla = bb.add(
            MultiControlX(cvs=[0] * self.dtype.bitsize), controls=x_arr, target=ancilla
        )
        x = bb.join(x_arr)

        (ctrl, ancilla), apply_op = bb.add(And(self.cv, 1), ctrl=(ctrl, ancilla))

        apply_op, x = bb.add(MultiTargetCNOT(self.dtype.bitsize), control=apply_op, targets=x)
        (apply_op,), x = bb.add(
            AddK(self.dtype.bitsize, self.mod + 1, cvs=(1,), signed=False), ctrls=(apply_op,), x=x
        )

        ctrl, ancilla = bb.add(And(self.cv, 1).adjoint(), ctrl=(ctrl, ancilla), target=apply_op)

        x_arr = bb.split(x)
        x_arr, ancilla = bb.add(
            MultiControlX(cvs=[0] * self.dtype.bitsize).adjoint(), controls=x_arr, target=ancilla
        )
        x = bb.join(x_arr)

        ancilla = bb.add(XGate(), q=ancilla)
        bb.free(ancilla)
        return {'ctrl': ctrl, 'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cvs: Union[list[int], HasLength]
        if isinstance(self.dtype.bitsize, int):
            cvs = [0] * self.dtype.bitsize
        else:
            cvs = HasLength(self.dtype.bitsize)
        return {
            (MultiControlX(cvs), 1),
            (MultiControlX(cvs).adjoint(), 1),
            (And(self.cv, 1), 1),
            (And(self.cv, 1).adjoint(), 1),
            (MultiTargetCNOT(self.dtype.bitsize), 1),
            (AddK(self.dtype.bitsize, k=self.mod + 1, cvs=(1,), signed=False), 1),
            (XGate(), 2),
        }

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'ctrl':
            return Circle(filled=self.cv == 1)
        elif reg.name == 'x':
            return TextBox('$-x$')
        raise ValueError(f'Unrecognized register name {reg.name}')

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == self.cv and 0 < x < self.mod:
            x = self.mod - x
        return {'ctrl': ctrl, 'x': x}


@bloq_example
def _mod_neg() -> ModNeg:
    n = 32
    prime = sympy.Symbol('p')
    mod_neg = ModNeg(QUInt(n), mod=prime)
    return mod_neg


_MOD_NEG_DOC = BloqDocSpec(bloq_cls=ModNeg, examples=[_mod_neg])


@bloq_example
def _cmod_neg() -> CModNeg:
    n = 32
    prime = sympy.Symbol('p')
    cmod_neg = CModNeg(QUInt(n), mod=prime)
    return cmod_neg


_CMOD_NEG_DOC = BloqDocSpec(bloq_cls=CModNeg, examples=[_cmod_neg])
