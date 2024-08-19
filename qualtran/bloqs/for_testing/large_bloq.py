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

from typing import Dict, List

import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QUInt, Signature, Soquet, SoquetT
from qualtran.bloqs.for_testing.atom import TestAtom
from qualtran.bloqs.mcmt import And


@frozen
class LargeBloq(Bloq):
    n_select: int
    n_ops: int

    @property
    def signature(self) -> 'Signature':
        return Signature.build(select=self.n_select, target=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', select, target) -> Dict[str, 'SoquetT']:
        sel = bb.split(select)
        ancs: List[Soquet] = [None] * self.n_select  # type: ignore
        ancs[0] = sel[0]

        cvs = QUInt(self.n_select - 1).to_bits_array(np.arange(self.n_ops) % (self.n_select - 1))
        assert cvs.shape == (self.n_ops, self.n_select - 1)

        for op_i in range(self.n_ops):

            # Ladder of ands
            for i in range(self.n_select - 1):
                and_op = And(cv1=cvs[op_i, i], cv2=1)
                [ancs[i], sel[i + 1]], ancs[i + 1] = bb.add(
                    and_op, ctrl=np.array([ancs[i], sel[i + 1]])
                )

            # The placeholder op
            ancs[-1], target = bb.add(TestAtom().controlled(), ctrl=ancs[-1], q=target)

            # Un-ladder of ands
            for i in range(self.n_select - 1 - 1, 0 - 1, -1):
                and_op = And(cv1=cvs[op_i, i], cv2=1)
                [ancs[i], sel[i + 1]] = bb.add(
                    and_op.adjoint(), ctrl=np.array([ancs[i], sel[i + 1]]), target=ancs[i + 1]
                )

        sel[0] = ancs[0]
        return {'select': bb.join(sel), 'target': target}
