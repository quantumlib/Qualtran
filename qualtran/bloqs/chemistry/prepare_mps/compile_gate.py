from typing import Dict
import attrs
from __future__ import annotations
from math import acos, sqrt

import numpy as np
from numpy.typing import ArrayLike

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.qrom import QROM


@attrs.frozen
class CompileGate(Bloq):
    n: int  # number of qubits that the gate acts on
    k: int  # number of rows of the gate specified

    @property
    def signature(self):
        return Signature.build(x=self.n)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, x: SoquetT, cols: ArrayLike[ArrayLike]
    ) -> Dict[str, SoquetT]:
        for i, col in enumerate(cols):
            i_state = np.zeros(col.shape)
            i_state[i] = 1/np.sqrt(2)
            w = np.concatenate(1/np.sqrt(2)*col, i_state)
            
        return {"x": x}


