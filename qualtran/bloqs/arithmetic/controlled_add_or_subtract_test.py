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
from typing import Union

import numpy as np
import pytest
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QInt,
    QMontgomeryUInt,
    QUInt,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import Add, Negate, Subtract
from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
    _ctrl_add_or_sub_signed,
    _ctrl_add_or_sub_signed_symb,
    _ctrl_add_or_sub_unsigned,
    ControlledAddOrSubtract,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_examples(bloq_autotester):
    bloq_autotester(_ctrl_add_or_sub_unsigned)
    bloq_autotester(_ctrl_add_or_sub_signed)
    bloq_autotester(_ctrl_add_or_sub_signed_symb)


@pytest.mark.parametrize("bitsize", [2, 4, pytest.param(5, marks=pytest.mark.slow)])
def test_controlled_add_or_subtract_classical_sim(bitsize: int):
    # TODO use QInt once classical simulation is fixed
    dtype = QUInt(bitsize)
    bloq = ControlledAddOrSubtract(dtype, dtype)
    for a_unsigned in dtype.get_classical_domain():
        for b_unsigned in dtype.get_classical_domain():
            for ctrl in [0, 1]:
                ctrl_out, a_out_unsigned, b_out_unsigned = bloq.call_classically(
                    ctrl=ctrl, a=a_unsigned, b=b_unsigned
                )
                assert ctrl_out == ctrl
                assert a_out_unsigned == a_unsigned

                if ctrl == 1:
                    # ctrl = 1 => add
                    assert b_out_unsigned == (b_unsigned + a_unsigned) % 2**bitsize
                else:
                    # ctrl = 0 => subtract
                    assert b_out_unsigned == (b_unsigned - a_unsigned + 2**bitsize) % 2**bitsize


@frozen
class TestNaiveControlledAddOrSubtract(Bloq):
    """A naive implementation of controlled add or subtract using two controlled bloqs.

    This should have the same action as `ControlledAddOrSubtract`, but twice the T-cost.
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()

    @property
    def signature(self) -> 'Signature':
        return ControlledAddOrSubtract(self.a_dtype, self.b_dtype).signature

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', a: 'Soquet', b: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        ctrl, a, b = bb.add(Add(self.a_dtype, self.b_dtype).controlled(), ctrl=ctrl, a=a, b=b)
        ctrl, a, b = bb.add(
            Subtract(self.a_dtype, self.b_dtype).controlled(CtrlSpec(cvs=0)), ctrl=ctrl, a=a, b=b
        )
        ctrl, b = bb.add(Negate(self.b_dtype).controlled(CtrlSpec(cvs=0)), ctrl=ctrl, x=b)
        return {'ctrl': ctrl, 'a': a, 'b': b}


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [QUInt(3), QUInt(4)])
def test_tensor(dtype):
    bloq = ControlledAddOrSubtract(dtype, dtype)
    naive_bloq = TestNaiveControlledAddOrSubtract(dtype, dtype)
    np.testing.assert_allclose(bloq.tensor_contract(), naive_bloq.tensor_contract())


def test_t_complexity():
    n = sympy.Symbol("n")
    dtype = QUInt(n)
    bloq = ControlledAddOrSubtract(dtype, dtype)

    counts = get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()
    assert counts['n_t'] == 0, 'toffoli only'
    assert counts['n_ccz'] == n - 1
