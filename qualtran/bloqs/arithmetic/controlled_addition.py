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

from typing import Dict, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.data_types import QMontgomeryUInt
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import add_ints

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CAdd(Bloq):
    r"""An n-bit controlled-addition gate.

    Args:
        a_dtype: Quantum datatype used to represent the integer a.
        b_dtype: Quantum datatype used to represent the integer b. Must be large
            enough to hold the result in the output register of a + b, or else it simply
            drops the most significant bits. If not specified, b_dtype is set to a_dtype.
        cv: When controlled=0, this bloq is active when the ctrl register is 0. When
            controlled=1, this bloq is active when the ctrl register is 1.

    Registers:
        ctrl: the control bit for the addition
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    cv: int = field(default=1)

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @a_dtype.validator
    def _a_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")
        if isinstance(val.num_qubits, sympy.Expr):
            return
        if val.bitsize > self.b_dtype.bitsize:
            raise ValueError("a_dtype bitsize must be less than or equal to b_dtype bitsize")

    @b_dtype.validator
    def _b_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")

    @cv.validator
    def _controlled_validate(self, field, val):
        if val not in (0, 1):
            raise ValueError("controlled must be either 0 or 1")

    @property
    def signature(self):
        return Signature(
            [Register("ctrl", QBit()), Register("a", self.a_dtype), Register("b", self.b_dtype)]
        )

    def on_classical_vals(self, **kwargs) -> Dict[str, 'ClassicalValT']:
        a, b = kwargs['a'], kwargs['b']
        ctrl = kwargs['ctrl']
        if ctrl != self.cv:
            return {'ctrl': ctrl, 'a': a, 'b': b}
        else:
            if not isinstance(self.b_dtype.bitsize, int):
                raise ValueError(f'classical simulation is not supported for symbolic bloq {self}')
            return {
                'ctrl': ctrl,
                'a': a,
                'b': add_ints(
                    a, b, num_bits=self.b_dtype.bitsize, is_signed=isinstance(self.b_dtype, QInt)
                ),
            }

    def short_name(self) -> str:
        return "a+b"

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        if soq.reg.name == 'ctrl':
            return directional_text_box('ctrl', side=soq.reg.side)
        if soq.reg.name == 'a':
            return directional_text_box('a', side=soq.reg.side)
        elif soq.reg.name == 'b':
            return directional_text_box('a+b', side=soq.reg.side)
        else:
            raise ValueError()

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', a: 'Soquet', b: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        a_arr = bb.split(a)
        ctrl_q = bb.split(ctrl)[0]
        ancilla_arr = []
        for i in range(len(a_arr)):
            [ctrl_q, a_arr[i]], target = bb.add(And(self.cv, 1), ctrl=np.array([ctrl_q, a_arr[i]]))
            ancilla_arr.append(target)
        ancilla = bb.join(np.array(ancilla_arr), QUInt(len(ancilla_arr)))
        ancilla = bb.add(Cast(QUInt(len(ancilla_arr)), self.a_dtype), reg=ancilla)

        ancilla, b = bb.add(Add(self.a_dtype, self.b_dtype), a=ancilla, b=b)
        ancilla_arr = bb.split(ancilla).tolist()

        for i in reversed(range(len(a_arr))):
            ctrl_q, a_arr[i] = bb.add(
                And(self.cv, 1).adjoint(), ctrl=np.array([ctrl_q, a_arr[i]]), target=ancilla_arr[i]
            )

        a = bb.join(a_arr, self.a_dtype)
        ctrl = bb.join(np.array([ctrl_q]))
        return {'ctrl': ctrl, 'a': a, 'b': b}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            And(self.cv, 1): self.a_dtype.bitsize,
            Add(self.a_dtype, self.b_dtype): 1,
            And(self.cv, 1).adjoint(): self.a_dtype.bitsize,
        }


@bloq_example(generalizer=ignore_split_join)
def _cadd_small() -> CAdd:
    cadd_small = CAdd(QUInt(3))
    return cadd_small


@bloq_example(generalizer=ignore_split_join)
def _cadd_large() -> CAdd:
    cadd_large = CAdd(QUInt(1000), QUInt(1000))
    return cadd_large


_CADD_DOC = BloqDocSpec(bloq_cls=CAdd, examples=[_cadd_small, _cadd_large])
