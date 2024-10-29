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
import enum
from typing import Iterable, Sequence

import attrs

from qualtran import AddControlledT, Adjoint, Bloq, BloqBuilder, CompositeBloq, CtrlSpec, SoquetT


class SpecializeOnCtrlBit(enum.Flag):
    """Control-specs to propagate to the subbloq.

    Currently only allows pushing a single-qubit-control.
    """

    NONE = enum.auto()
    ZERO = enum.auto()
    ONE = enum.auto()
    BOTH = ZERO | ONE


@attrs.frozen()
class AdjointWithSpecializedCtrl(Adjoint):
    specialize_on_ctrl: SpecializeOnCtrlBit = SpecializeOnCtrlBit.NONE

    def _specialize_control(self, ctrl_spec: 'CtrlSpec') -> bool:
        """if True, push the control to the subbloq"""
        if ctrl_spec.num_qubits != 1:
            return False

        cv = ctrl_spec.get_single_ctrl_bit()
        cv_flag = SpecializeOnCtrlBit.ONE if cv == 1 else SpecializeOnCtrlBit.ZERO
        return cv_flag in self.specialize_on_ctrl

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran._infra.controlled import _get_nice_ctrl_reg_names

        if not self._specialize_control(ctrl_spec):
            # no specialized controlled version available, fallback to default
            return super().get_ctrl_system(ctrl_spec)

        # get the builder for the controlled version of subbloq
        ctrl_subbloq, ctrl_subbloq_adder = self.subbloq.get_ctrl_system(ctrl_spec)
        ctrl_bloq = attrs.evolve(self, subbloq=ctrl_subbloq)
        (ctrl_reg_name,) = _get_nice_ctrl_reg_names([reg.name for reg in self.subbloq.signature], 1)

        # build a composite bloq using the control-adder
        def _get_adj_cbloq() -> 'CompositeBloq':
            bb, initial_soqs = BloqBuilder.from_signature(
                self.subbloq.signature, add_registers_allowed=True
            )
            ctrl = bb.add_register(ctrl_reg_name, 1)
            bb.add_register_allowed = False

            (ctrl,), out_soqs_t = ctrl_subbloq_adder(bb, [ctrl], initial_soqs)

            out_soqs = dict(zip([reg.name for reg in self.subbloq.signature.rights()], out_soqs_t))
            out_soqs |= {ctrl_reg_name: ctrl}

            cbloq = bb.finalize(**out_soqs)
            return cbloq.adjoint()

        adj_cbloq = _get_adj_cbloq()

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            in_soqs |= {ctrl_reg_name: ctrl}
            soqs = bb.add_from(adj_cbloq, **in_soqs)

            # locate the correct control soquet
            soqs = list(soqs)
            ctrl_soq = None
            for soq, reg in zip(soqs, adj_cbloq.signature.rights()):
                if reg.name == ctrl_reg_name:
                    ctrl_soq = soq
                    soqs.remove(soq)
                    break
            assert ctrl_soq is not None, "ctrl_soq must be present in output soqs"

            return [ctrl_soq], soqs

        return ctrl_bloq, _adder
