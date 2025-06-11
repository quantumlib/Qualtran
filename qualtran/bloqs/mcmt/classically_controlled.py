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
from typing import Tuple

import attrs

from qualtran import AddControlledT, Bloq, CDType, CtrlSpec, QCDType
from qualtran._infra.controlled import _ControlledBase


@attrs.frozen
class ClassicallyControlled( _ControlledBase):

    subbloq: 'Bloq'
    ctrl_spec: 'CtrlSpec'

    def __attrs_post_init__(self):
        for qcdtype in self.ctrl_spec.qdtypes:
            if not isinstance(qcdtype, QCDType):
                raise ValueError(f"Invalid type found in `ctrl_spec`: {qcdtype}")
            if not isinstance(qcdtype, CDType):
                raise ValueError(f"Invalid type found in `ctrl_spec`: {qcdtype}")

    @classmethod
    def make_ctrl_system(
        cls, bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
    ) -> Tuple['_ControlledBase', 'AddControlledT']:
        cb = cls(subbloq=bloq, ctrl_spec=ctrl_spec)
        return cls._make_ctrl_system(cb)