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
from typing import cast, Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QBit,
    QDType,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.gate_with_registers import SpecializedSingleQubitControlledGate
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _cvs_converter(vv):
    if isinstance(vv, (int, np.integer)):
        return (int(vv),)
    return tuple(int(v) for v in vv)


@frozen
class XorK(SpecializedSingleQubitControlledGate):
    r"""Maps |x> to |x \oplus k> for a constant k.

    Args:
        dtype: Data type of the input register `x`.
        k: The classical integer value to be XOR-ed to x.
        control_val: an optional single bit control, apply the operation when
                    the control qubit equals the `control_val`.

    Registers:
        x: A quantum register of type `self.dtype` (see above).
        ctrl: A sequence of control qubits (only when `control_val` is not None).
    """
    dtype: QDType
    k: SymbolicInt
    control_val: Optional[int] = None

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([*self.control_registers, Register('x', self.dtype)])

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        if self.control_val is not None:
            return (Register('ctrl', QBit()),)
        return ()

    @cached_property
    def bitsize(self) -> SymbolicInt:
        return self.dtype.num_qubits

    def is_symbolic(self):
        return is_symbolic(self.k, self.dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        # TODO clean this up once https://github.com/quantumlib/Qualtran/pull/1137 is merged
        ctrl = soqs.pop('ctrl', None)

        xs = bb.split(cast(Soquet, soqs.pop('x')))

        for i, bit in enumerate(self.dtype.to_bits(self.k)):
            if bit == 1:
                if ctrl is not None:
                    ctrl, xs[i] = bb.add(CNOT(), ctrl=ctrl, target=xs[i])
                else:
                    xs[i] = bb.add(XGate(), q=xs[i])

        soqs['x'] = bb.join(xs)

        if ctrl is not None:
            soqs['ctrl'] = ctrl
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        bit_flip_bloq = CNOT() if self.control_val is not None else XGate()
        num_flips = self.bitsize if self.is_symbolic() else sum(self.dtype.to_bits(self.k))
        return {(bit_flip_bloq, num_flips)}


@bloq_example(generalizer=ignore_split_join)
def _xork() -> XorK:
    xork = XorK(QUInt(8), 0b01010111)
    return xork


@bloq_example(generalizer=ignore_split_join)
def _cxork() -> XorK:
    cxork = XorK(QUInt(8), 0b01010111).controlled()
    assert isinstance(cxork, XorK)
    return cxork


@frozen
class Xor(Bloq):
    """Xor the value of one register into another via CNOTs.

    When both registers are in computational basis and the destination is 0,
    effectively copies the value of the source into the destination.

    Args:
        bitsize: The size of the register.

    Registers:
        ctrl: The source register.
        x: The target register.
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=self.bitsize, x=self.bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, ctrl: Soquet, x: Soquet) -> Dict[str, SoquetT]:
        if not isinstance(self.bitsize, int):
            raise DecomposeTypeError("`bitsize` must be a concrete value.")

        ctrls = bb.split(ctrl)
        xs = bb.split(x)

        for i in range(self.bitsize):
            ctrls[i], xs[i] = bb.add_t(CNOT(), ctrl=ctrls[i], target=xs[i])

        return {'ctrl': bb.join(ctrls), 'x': bb.join(xs)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(CNOT(), self.bitsize)}

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'ctrl': ctrl, 'x': ctrl}


@bloq_example
def _xor() -> Xor:
    xor = Xor(4)
    return xor


@bloq_example
def _xor_symb() -> Xor:
    xor_symb = Xor(sympy.Symbol("n"))
    return xor_symb


_XOR_DOC = BloqDocSpec(
    bloq_cls=Xor,
    import_line='from qualtran.bloqs.arithmetic import Xor',
    examples=(_xor, _xor_symb),
)
