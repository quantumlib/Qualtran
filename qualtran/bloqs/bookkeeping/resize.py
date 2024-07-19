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
from functools import cached_property
from typing import Dict, List, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    bloq_example,
    BloqDocSpec,
    QInt, QUInt, QMontgomeryUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.basic_gates import CNOT

if TYPE_CHECKING:
    from qualtran import Soquet, BloqBuilder, Bloq
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Resize(_BookkeepingBloq):


    dtype: QInt | QUInt | QMontgomeryUInt
    inp_bitsize: int
    out_bitsize: int


    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', dtype=self.dtype(self.inp_bitsize), side=Side.LEFT),
                Register('reg', dtype=self.dtype(self.out_bitsize), side=Side.RIGHT),
            ]
        )
    
    def adjoint(self) -> 'Bloq':
        return Resize(self.dtype, self.out_bitsize, self.inp_bitsize)
    
    def _is_signed(self) -> bool:
        return self.dtype is QInt

    def on_classical_vals(self, reg: int) -> Dict[str, 'ClassicalValT']:
        if self.out_bitsize < self.inp_bitsize:
            if self._is_signed():
                half_n = 1 << (self.out_bitsize - 1)
                reg = (reg + half_n)%(1 << self.out_bitsize) - half_n
            else:
                reg = reg%(1 << self.out_bitsize)
        return {'reg': reg}
    
    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, reg: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        inp_arr = bb.split(reg)
        if self.inp_bitsize < self.out_bitsize:
            prefix = bb.split(bb.allocate(self.out_bitsize - self.inp_bitsize))
            if self._is_signed():
                for i in range(len(prefix)):
                    inp_arr[0], prefix[i] = bb.add(CNOT(), ctrl=inp_arr[0], target=prefix[i])
            inp_arr = np.concatenate([prefix, inp_arr])
    
        if self.inp_bitsize > self.out_bitsize:
            prefix, inp_arr = inp_arr[:self.inp_bitsize-self.out_bitsize], inp_arr[-self.out_bitsize:]
            if self._is_signed():
                for i in range(len(prefix)):
                    inp_arr[0], prefix[i] = bb.add(CNOT(), ctrl=inp_arr[0], target=prefix[i])
            bb.free(bb.join(prefix))
        return {'reg': bb.join(inp_arr, self.dtype(self.out_bitsize))}