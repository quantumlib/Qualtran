#  Copyright 2023 Google LLC
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

import abc
from typing import Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran._infra.bloq import Bloq
from qualtran._infra.controlled import CtrlSpec
from qualtran._infra.registers import Register

if TYPE_CHECKING:
    from qualtran import AddControlledT, BloqBuilder, SoquetT


class SpecializedSingleQubitControlledExtension(Bloq):
    """Add a specialized single-qubit controlled version of a Bloq.

    `control_val` is an optional single-bit control. When `control_val` is provided,
     the `control_registers` property should return a single named qubit register,
     and otherwise return an empty tuple.

    Example usage:

        @attrs.frozen
        class MyGate(SpecializedSingleQubitControlledExtension):
            control_val: Optional[int] = None

            @property
            def control_registers() -> Tuple[Register, ...]:
                return () if self.control_val is None else (Register('control', QBit()),)
    """

    control_val: Optional[int]

    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[Register, ...]:
        ...

    def get_single_qubit_controlled_bloq(
        self, control_val: int
    ) -> 'SpecializedSingleQubitControlledExtension':
        """Override this to provide a custom controlled bloq"""
        return attrs.evolve(self, control_val=control_val)  # type: ignore[misc]

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        if self.control_val is None and ctrl_spec.shapes in [((),), ((1,),)]:
            control_val = int(ctrl_spec.cvs[0].item())
            cbloq = self.get_single_qubit_controlled_bloq(control_val)

            if not hasattr(cbloq, 'control_registers'):
                raise TypeError("{cbloq} should have attribute `control_registers`")

            (ctrl_reg,) = cbloq.control_registers

            def adder(
                bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
            ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
                soqs = {ctrl_reg.name: ctrl_soqs[0]} | in_soqs
                soqs = bb.add_d(cbloq, **soqs)
                ctrl_soqs = [soqs.pop(ctrl_reg.name)]
                return ctrl_soqs, soqs.values()

            return cbloq, adder

        return super().get_ctrl_system(ctrl_spec)
