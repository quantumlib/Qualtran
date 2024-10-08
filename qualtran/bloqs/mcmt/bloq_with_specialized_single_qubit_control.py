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
from typing import cast, Iterable, Optional, Protocol, runtime_checkable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qualtran import AddControlledT, Bloq, BloqBuilder, CtrlSpec, SoquetT


@runtime_checkable
class BloqWithSpecializedControl(Protocol):
    """mixin for a bloq that has a specialized single qubit controlled version."""

    @property
    def cv(self) -> Optional[int]: ...

    def with_cv(self, *, cv: Optional[int]) -> 'Bloq': ...

    @property
    def ctrl_reg_name(self) -> str: ...


def get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
    bloq: 'BloqWithSpecializedControl', ctrl_spec: 'CtrlSpec'
) -> tuple['Bloq', 'AddControlledT']:
    from qualtran import Bloq, CtrlSpec, Soquet
    from qualtran.bloqs.mcmt import ControlledViaAnd

    if ctrl_spec != CtrlSpec():
        assert isinstance(bloq, Bloq)
        return ControlledViaAnd.make_ctrl_system(bloq=bloq, ctrl_spec=ctrl_spec)

    assert isinstance(
        bloq, BloqWithSpecializedControl
    ), f"{bloq} must implement protocol {BloqWithSpecializedControl}"

    if bloq.cv is None:
        # the easy case: use the controlled bloq
        ctrl_bloq = bloq.with_cv(cv=1)
        assert isinstance(ctrl_bloq, BloqWithSpecializedControl)
        ctrl_reg_name = ctrl_bloq.ctrl_reg_name

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            in_soqs |= {ctrl_reg_name: ctrl}

            out_soqs = bb.add_d(ctrl_bloq, **in_soqs)

            ctrl = out_soqs.pop(ctrl_reg_name)
            return [ctrl], out_soqs.values()

    else:
        # the difficult case: must combine the two controls into one
        un_ctrl_bloq = bloq.with_cv(cv=None)
        ctrl_bloq = ControlledViaAnd(un_ctrl_bloq, CtrlSpec(cvs=[1, bloq.cv]))

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            # extract the two control bits
            (ctrl0,) = ctrl_soqs
            ctrl1 = in_soqs.pop('ctrl')

            ctrl0 = cast(Soquet, ctrl0)
            ctrl1 = cast(Soquet, ctrl1)

            # add the singly controlled bloq
            ctrls, *out_soqs = bb.add_t(ctrl_bloq, ctrl=[ctrl0, ctrl1], **in_soqs)
            assert isinstance(ctrls, np.ndarray)
            ctrl0, ctrl1 = ctrls

            return [ctrl0], [ctrl1, *out_soqs]

    return ctrl_bloq, _adder
