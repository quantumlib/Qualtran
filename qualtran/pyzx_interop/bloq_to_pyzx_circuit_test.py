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
import numpy as np
import pyzx as zx

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import OneEffect, XGate, ZeroState, ZGate
from qualtran.pyzx_interop.bloq_to_pyzx_circuit import bloq_to_pyzx_circuit


class TestBloq(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'Soquet') -> dict[str, 'SoquetT']:
        q = bb.add(ZGate(), q=q)
        q = bb.add(XGate(), q=q)

        a = bb.add(ZeroState())
        a = bb.add(XGate(), q=a)
        bb.add(OneEffect(), q=a)

        return {'q': q}


def test_bloq_to_pyzx_circuit():
    bloq = TestBloq()
    circ = bloq_to_pyzx_circuit(bloq)
    tensor = bloq.tensor_contract()

    assert zx.compare_tensors(circ, tensor)
    np.testing.assert_allclose(np.imag(zx.find_scalar_correction(circ, tensor)), 0, atol=1e-7)
