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
import itertools
from typing import Optional, Sequence, Tuple
from unittest.mock import ANY

import attrs
import pytest

import qualtran.testing as qlt_testing
from qualtran import (
    AddControlledT,
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.specialized_ctrl import (
    AdjointWithSpecializedCtrl,
    get_ctrl_system_1bit_cv,
    get_ctrl_system_1bit_cv_from_bloqs,
    SpecializeOnCtrlBit,
)
from qualtran.resource_counting import CostKey, GateCounts, get_cost_value, QECGatesCost


@attrs.frozen
class AtomWithSpecializedControl(Bloq):
    cv: Optional[int] = None
    ctrl_reg_name: str = 'ctrl'
    target_reg_name: str = 'q'

    @property
    def signature(self) -> 'Signature':
        n_ctrl = 1 if self.cv is not None else 0
        reg_name_map = {self.ctrl_reg_name: n_ctrl, self.target_reg_name: 2}
        return Signature.build(**reg_name_map)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_1bit_cv(
            self,
            ctrl_spec,
            current_ctrl_bit=self.cv,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (
                attrs.evolve(self, cv=cv),
                self.ctrl_reg_name,
            ),
        )

    @staticmethod
    def cost_expr_for_cv(cv: Optional[int]):
        import sympy

        c_unctrl = sympy.Symbol("_c_target_")
        c_ctrl = sympy.Symbol("_c_ctrl_")

        if cv is None:
            return c_unctrl
        return c_unctrl + c_ctrl

    def my_static_costs(self, cost_key: 'CostKey'):
        if cost_key == QECGatesCost():
            r = self.cost_expr_for_cv(self.cv)
            return GateCounts(rotation=r)

        return NotImplemented

    def adjoint(self) -> 'AdjointWithSpecializedCtrl':
        return AdjointWithSpecializedCtrl(self, specialize_on_ctrl=SpecializeOnCtrlBit.BOTH)


def ON(n: int = 1) -> CtrlSpec:
    return CtrlSpec(cvs=[1] * n)


def OFF(n: int = 1) -> CtrlSpec:
    return CtrlSpec(cvs=[0] * n)


@pytest.mark.parametrize(
    'ctrl_specs',
    [
        [ON()],
        [OFF()],
        [OFF(), OFF()],
        [OFF(4)],
        [OFF(2), OFF(2)],
        [ON(), OFF(5)],
        [ON(), ON(), ON()],
        [OFF(4), ON(3), OFF(5)],
    ],
)
@pytest.mark.parametrize('ctrl_reg_name', ['ctrl', 'control'])
def test_custom_controlled(ctrl_specs: Sequence[CtrlSpec], ctrl_reg_name: str):
    bloq: Bloq = AtomWithSpecializedControl(ctrl_reg_name=ctrl_reg_name)
    for ctrl_spec in ctrl_specs:
        bloq = bloq.controlled(ctrl_spec)
    n_ctrls = sum(ctrl_spec.num_bits for ctrl_spec in ctrl_specs)

    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(
        and_bloq=n_ctrls - 1,
        rotation=AtomWithSpecializedControl.cost_expr_for_cv(1),
        clifford=ANY,
        measurement=ANY,
    )


@attrs.frozen
class TestAtom(Bloq):
    tag: str

    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=2)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=CTestAtom(self.tag),
            ctrl_reg_name='ctrl',
        )

    def adjoint(self) -> 'AdjointWithSpecializedCtrl':
        return AdjointWithSpecializedCtrl(self, specialize_on_ctrl=SpecializeOnCtrlBit.ONE)


@attrs.frozen
class CTestAtom(Bloq):
    tag: str

    @property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, q=2)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=1, bloq_with_ctrl=self, ctrl_reg_name='ctrl'
        )

    def adjoint(self) -> 'AdjointWithSpecializedCtrl':
        return AdjointWithSpecializedCtrl(self, specialize_on_ctrl=SpecializeOnCtrlBit.ONE)


def test_bloq_with_controlled_bloq():
    assert TestAtom('g').controlled() == CTestAtom('g')

    ctrl_bloq = CTestAtom('g').controlled()
    _, sigma = ctrl_bloq.call_graph()
    assert sigma == {And(): 1, CTestAtom('g'): 1, And().adjoint(): 1}

    ctrl_bloq = CTestAtom('n').controlled(CtrlSpec(cvs=0))
    _, sigma = ctrl_bloq.call_graph()
    assert sigma == {And(0, 1): 1, CTestAtom('n'): 1, And(0, 1).adjoint(): 1}

    ctrl_bloq = TestAtom('nn').controlled(CtrlSpec(cvs=[0, 0]))
    _, sigma = ctrl_bloq.call_graph()
    assert sigma == {And(0, 0): 1, CTestAtom('nn'): 1, And(0, 0).adjoint(): 1}


def test_ctrl_adjoint():
    assert TestAtom('a').adjoint().controlled() == CTestAtom('a').adjoint()

    _, sigma = TestAtom('g').adjoint().controlled(ctrl_spec=CtrlSpec(cvs=[1, 1])).call_graph()
    assert sigma == {And(): 1, And().adjoint(): 1, CTestAtom('g').adjoint(): 1}

    _, sigma = CTestAtom('c').adjoint().controlled().call_graph()
    assert sigma == {And(): 1, And().adjoint(): 1, CTestAtom('c').adjoint(): 1}

    for cv in [0, 1]:
        assert (
            AtomWithSpecializedControl().adjoint().controlled(ctrl_spec=CtrlSpec(cvs=cv))
            == AtomWithSpecializedControl(cv=cv).adjoint()
        )


@attrs.frozen
class TestBloqWithDecompose(Bloq):
    ctrl_reg_name: str
    target_reg_name: str

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [Register(self.ctrl_reg_name, QBit()), Register(self.target_reg_name, QAny(2))]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        for _ in range(2):
            soqs = bb.add_d(
                AtomWithSpecializedControl(
                    cv=1, ctrl_reg_name=self.ctrl_reg_name, target_reg_name=self.target_reg_name
                ),
                **soqs,
            )
        return soqs


@pytest.mark.parametrize(
    ('ctrl_reg_name', 'target_reg_name'),
    [
        (ctrl, targ)
        for (ctrl, targ) in itertools.product(['ctrl', 'control', 'a', 'b'], repeat=2)
        if ctrl != targ
    ],
)
def test_get_ctrl_system(ctrl_reg_name: str, target_reg_name: str):
    bloq = TestBloqWithDecompose(ctrl_reg_name, target_reg_name).controlled()
    _ = bloq.decompose_bloq().flatten()


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('specialized_ctrl')
