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
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

from qualtran import Bloq, BloqBuilder, SoquetT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:

    from qualtran import AddControlledT, CtrlSpec


class _BookkeepingBloq(Bloq, metaclass=abc.ABCMeta):
    """Base class for utility bloqs used for bookkeeping.

    This bloq:
    - has trivial controlled versions, which pass through the control register.
    - does not affect T complexity.
    """

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> Tuple['Bloq', 'AddControlledT']:
        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            # ignore `ctrl_soq` and pass it through for bookkeeping operation.
            out_soqs = bb.add_t(self, **in_soqs)
            return ctrl_soqs, out_soqs

        return self, add_controlled

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity()
