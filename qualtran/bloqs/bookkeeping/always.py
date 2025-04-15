#  Copyright 2025 Google LLC
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
from typing import Iterable, Optional, Sequence

import attrs

from qualtran import AddControlledT, Bloq, BloqBuilder, CtrlSpec, Signature, SoquetT


@attrs.frozen
class Always(Bloq):
    """Apply the wrapped bloq as-is, ignoring any controls.

    Useful when writing decompositions which have bloqs that occur in compute-uncompute pairs.
    Simply wrap the compute and uncompute bloq in `Always`, and controlled versions of
    the whole bloq will skip controls for the wrapped subbloqs.

    Caution:
        This wrapper should be used with care, _only_ when ignoring the controls
        does not affect the action of the bloq.

    Args:
        subbloq: The bloq to wrap, for which controls are ignores.
    """

    subbloq: Bloq

    @property
    def signature(self) -> 'Signature':
        return self.subbloq.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return bb.add_d(self.subbloq, **soqs)

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> tuple['Bloq', 'AddControlledT']:
        """Pass-through the control registers as-is"""

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            out_soqs = bb.add_t(self, **in_soqs)
            return ctrl_soqs, out_soqs

        return self, add_controlled

    def adjoint(self) -> 'Always':
        return Always(self.subbloq.adjoint())

    def __str__(self) -> str:
        return str(self.subbloq)
