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
from typing import Optional, Sequence, Tuple
from unittest.mock import ANY

import attrs
import pytest

from qualtran import AddControlledT, Bloq, CtrlSpec, Signature
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.bloq_with_specialized_single_qubit_control import (
    get_ctrl_system_for_bloq_with_specialized_single_qubit_control,
)
from qualtran.resource_counting import CostKey, GateCounts, get_cost_value, QECGatesCost


@attrs.frozen
class AtomWithSpecializedControl(Bloq):
    cv: Optional[int] = None

    @property
    def signature(self) -> 'Signature':
        n_ctrl = 1 if self.cv is not None else 0
        return Signature.build(ctrl=n_ctrl, q=2)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=self.cv,
            bloq_without_ctrl=attrs.evolve(self, cv=None),
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (attrs.evolve(self, cv=cv), 'ctrl'),
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
def test_custom_controlled(ctrl_specs: Sequence[CtrlSpec]):
    bloq: Bloq = AtomWithSpecializedControl()
    for ctrl_spec in ctrl_specs:
        bloq = bloq.controlled(ctrl_spec)
    n_ctrls = sum(ctrl_spec.num_qubits for ctrl_spec in ctrl_specs)

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
        def get_ctrl_bloq_and_ctrl_reg_name(cv):
            if cv == 0:
                return None
            return CTestAtom(self.tag), 'ctrl'

        return get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=None,
            bloq_without_ctrl=self,
            get_ctrl_bloq_and_ctrl_reg_name=get_ctrl_bloq_and_ctrl_reg_name,
        )


@attrs.frozen
class CTestAtom(Bloq):
    tag: str

    @property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, q=2)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        def get_ctrl_bloq_and_ctrl_reg_name(cv):
            if cv == 0:
                return None
            return self, 'ctrl'

        return get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=1,
            bloq_without_ctrl=TestAtom(self.tag),
            get_ctrl_bloq_and_ctrl_reg_name=get_ctrl_bloq_and_ctrl_reg_name,
        )


def test_bloq_with_controlled_bloq():
    assert TestAtom('g').controlled() == CTestAtom('g')

    def _keep_and(b):
        # TODO remove this after https://github.com/quantumlib/Qualtran/issues/1346 is resolved.
        return isinstance(b, And)

    ctrl_bloq = CTestAtom('g').controlled()
    _, sigma = ctrl_bloq.call_graph(keep=_keep_and)
    assert sigma == {And(): 1, CTestAtom('g'): 1, And().adjoint(): 1}

    ctrl_bloq = CTestAtom('n').controlled(CtrlSpec(cvs=0))
    _, sigma = ctrl_bloq.call_graph(keep=_keep_and)
    assert sigma == {And(0, 1): 1, CTestAtom('n'): 1, And(0, 1).adjoint(): 1}

    ctrl_bloq = TestAtom('nn').controlled(CtrlSpec(cvs=[0, 0]))
    _, sigma = ctrl_bloq.call_graph(keep=_keep_and)
    assert sigma == {And(0, 0): 1, CTestAtom('nn'): 1, And(0, 0).adjoint(): 1}
