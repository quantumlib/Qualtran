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


@pytest.mark.parametrize("dtype_cls", [QUInt, QInt])
@pytest.mark.parametrize("bitsize", [2, 4, pytest.param(5, marks=pytest.mark.slow)])
@pytest.mark.parametrize("add_when_ctrl_is_on", [True, False])
def test_controlled_add_or_subtract_classical_sim(
    dtype_cls, bitsize: int, add_when_ctrl_is_on: bool
):
    dtype = dtype_cls(bitsize)
    bloq = ControlledAddOrSubtract(dtype, dtype, add_when_ctrl_is_on=add_when_ctrl_is_on)
    for a_val in dtype.get_classical_domain():
        for b_val in dtype.get_classical_domain():
            for ctrl in [0, 1]:
                ctrl_out, a_out, b_out = bloq.call_classically(ctrl=ctrl, a=a_val, b=b_val)
                assert ctrl_out == ctrl
                assert a_out == a_val

                is_add = (ctrl == 1) if add_when_ctrl_is_on else (ctrl == 0)
                if dtype_cls == QUInt:
                    expected_b = (b_val + a_val if is_add else b_val - a_val) % 2**bitsize
                else:
                    expected_b = (
                        (b_val + a_val if is_add else b_val - a_val) + 2 ** (bitsize - 1)
                    ) % 2**bitsize - 2 ** (bitsize - 1)
                assert b_out == expected_b


def test_controlled_add_or_subtract_symbolic_classical_sim():
    bloq = _ctrl_add_or_sub_signed_symb()
    with pytest.raises(ValueError, match="Cannot simulate symbolic bloq"):
        bloq.on_classical_vals(ctrl=1, a=1, b=1)


def test_caddsub_mixed_dtype():
    from qualtran.bloqs.arithmetic import Add

    bloq = Add(QUInt(4), QInt(4))
    a, b_res = bloq.call_classically(a=0, b=-1)
    assert a == 0
    assert b_res == -1

    bloq = ControlledAddOrSubtract(QUInt(4), QInt(4))
    ctrl, a, b_result = bloq.call_classically(a=0, b=-1, ctrl=1)
    assert ctrl == 1
    assert a == 0
    assert b_result == -1


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


def test_controlled_t_complexity():
    dtype = QUInt(10)
    bloq = ControlledAddOrSubtract(dtype, dtype)

    _ = bloq.controlled().adjoint().t_complexity()
    _ = bloq.adjoint().controlled().t_complexity()
