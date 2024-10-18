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
from typing import Callable, cast, Iterable, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qualtran import AddControlledT, Bloq, BloqBuilder, CtrlSpec, SoquetT
    from qualtran._infra.controlled import ControlBit


def get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
    *,
    ctrl_spec: 'CtrlSpec',
    current_ctrl_bit: Optional['ControlBit'],
    bloq_without_ctrl: 'Bloq',
    get_ctrl_bloq_and_ctrl_reg_name: Callable[['ControlBit'], Optional[tuple['Bloq', str]]],
) -> tuple['Bloq', 'AddControlledT']:
    """Build the control system for a bloq with a specialized single-qubit controlled variant.

    Uses the provided specialized implementation when a singly-controlled variant of the bloq is
    requested. When controlled by multiple qubits, the controls are reduced to a single qubit
    and the singly-controlled bloq is used.

    The user can provide specializations for the bloq controlled by `1` and (optionally) by `0`.
    The specialization for control bit `1` must be provided.
    In case a specialization for a control bit `0` is not provided, the default fallback is used
    instead, which wraps the bloq using the `Controlled` metabloq.

    Args:
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        bloq_without_ctrl: The variant of this bloq without a control.
        get_ctrl_bloq_and_ctrl_reg_name: A callable that accepts a control bit (`0` or `1`),
            and returns the controlled variant of this bloq and the name of the control register.
            If the callable returns `None`, then the default fallback is used.
    """
    from qualtran import CtrlSpec, Soquet
    from qualtran.bloqs.mcmt import ControlledViaAnd

    def _get_default_fallback():
        current_bloq: 'Bloq'
        if current_ctrl_bit is None:
            current_bloq = bloq_without_ctrl
        else:
            ctrl_bloq_and_reg = get_ctrl_bloq_and_ctrl_reg_name(current_ctrl_bit)
            if ctrl_bloq_and_reg is None:
                raise ValueError(
                    f"Expected a controlled bloq (self) matching the current control bit {current_ctrl_bit}, got None"
                )
            current_bloq, _ = ctrl_bloq_and_reg

        return ControlledViaAnd.make_ctrl_system(bloq=current_bloq, ctrl_spec=ctrl_spec)

    if ctrl_spec.num_qubits != 1:
        return _get_default_fallback()

    ctrl_bit = ctrl_spec.get_single_ctrl_bit()

    if current_ctrl_bit is None:
        # the easy case: use the controlled bloq
        ctrl_bloq_and_ctrl_reg_name = get_ctrl_bloq_and_ctrl_reg_name(ctrl_bit)
        if ctrl_bloq_and_ctrl_reg_name is None:
            return _get_default_fallback()

        ctrl_bloq, ctrl_reg_name = ctrl_bloq_and_ctrl_reg_name

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
        ctrl_bloq = ControlledViaAnd(bloq_without_ctrl, CtrlSpec(cvs=[ctrl_bit, current_ctrl_bit]))
        (ctrl_reg_name,) = ctrl_bloq.ctrl_reg_names
        _, in_ctrl_reg_name = get_ctrl_bloq_and_ctrl_reg_name(1)

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            # extract the two control bits
            (ctrl0,) = ctrl_soqs
            ctrl1 = in_soqs.pop(in_ctrl_reg_name)

            ctrl0 = cast(Soquet, ctrl0)
            ctrl1 = cast(Soquet, ctrl1)

            # add the singly controlled bloq
            in_soqs |= {ctrl_reg_name: [ctrl0, ctrl1]}
            ctrls, *out_soqs = bb.add_t(ctrl_bloq, **in_soqs)
            assert isinstance(ctrls, np.ndarray)
            ctrl0, ctrl1 = ctrls

            return [ctrl0], [ctrl1, *out_soqs]

    return ctrl_bloq, _adder


def get_ctrl_system_for_bloq_with_specialized_single_qubit_control_from_list(
    *,
    ctrl_spec: 'CtrlSpec',
    current_ctrl_bit: Optional['ControlBit'],
    bloq_without_ctrl: 'Bloq',
    bloq_with_ctrl_1: 'Bloq',
    ctrl_reg_name: 'str',
    bloq_with_ctrl_0: Optional['Bloq'],
) -> tuple['Bloq', 'AddControlledT']:
    """Helper to construct the control system given uncontrolled and singly-controlled variants of a bloq.

    See :meth:`get_ctrl_system_for_bloq_with_specialized_single_qubit_control` for details on usage.

    Args:
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        bloq_without_ctrl: The variant of this bloq without a control.
        bloq_with_ctrl_1: The variant of this bloq controlled by a single qubit in the `1` basis state.
        ctrl_reg_name: The name of the control register for the controlled bloq variant(s).
        bloq_with_ctrl_0: (optional) The variant of this bloq controlled by a single qubit in the `1` basis state.
    """

    def get_ctrl_bloq_and_ctrl_reg_name(cv: 'ControlBit') -> Optional[tuple['Bloq', str]]:
        if cv == 1:
            return bloq_with_ctrl_1, ctrl_reg_name
        else:
            if bloq_with_ctrl_0 is None:
                return None
            return bloq_with_ctrl_0, ctrl_reg_name

    return get_ctrl_system_for_bloq_with_specialized_single_qubit_control(
        ctrl_spec=ctrl_spec,
        current_ctrl_bit=current_ctrl_bit,
        bloq_without_ctrl=bloq_without_ctrl,
        get_ctrl_bloq_and_ctrl_reg_name=get_ctrl_bloq_and_ctrl_reg_name,
    )
