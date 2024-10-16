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
from typing import cast, Iterable, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qualtran import AddControlledT, Bloq, BloqBuilder, CtrlSpec, SoquetT


ControlBit = Optional[0 | 1]


def get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
    *,
    ctrl_spec: 'CtrlSpec',
    current_ctrl_bit: ControlBit,
    bloq_with_ctrl: 'Bloq',
    ctrl_reg_name: str,
    bloq_without_ctrl: 'Bloq',
    bloq_with_ctrl_0: Optional['Bloq'] = None,
) -> tuple['Bloq', 'AddControlledT']:
    """Build the control system for a bloq with a specialized single-qubit controlled variant.

    Args:
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        bloq_with_ctrl: The variant of this bloq with control bit `1`.
        ctrl_reg_name: The name of the control qubit register.
        bloq_without_ctrl: The variant of this bloq without a control.
        bloq_with_ctrl_0: (optional) the variant of this bloq controlled by a qubit in the 0 state.
    """
    from qualtran import CtrlSpec, Soquet
    from qualtran.bloqs.mcmt import ControlledViaAnd

    def _get_default_fallback():
        current_bloq: 'Bloq'
        if current_ctrl_bit is None:
            current_bloq = bloq_without_ctrl
        elif current_ctrl_bit == 1:
            current_bloq = bloq_with_ctrl
        elif current_ctrl_bit == 0:
            current_bloq = bloq_with_ctrl_0
        else:
            raise ValueError(f"invalid control bit {current_ctrl_bit}")

        return ControlledViaAnd.make_ctrl_system(bloq=current_bloq, ctrl_spec=ctrl_spec)

    if ctrl_spec.num_qubits != 1:
        return _get_default_fallback()

    control_bit = ctrl_spec.get_single_ctrl_bit()

    if current_ctrl_bit is None:
        # the easy case: use the controlled bloq
        ctrl_bloq = bloq_with_ctrl if control_bit == 1 else bloq_with_ctrl_0
        if ctrl_bloq is None:
            return _get_default_fallback()

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
        ctrl_bloq = ControlledViaAnd(
            bloq_without_ctrl, CtrlSpec(cvs=[control_bit, current_ctrl_bit])
        )

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
