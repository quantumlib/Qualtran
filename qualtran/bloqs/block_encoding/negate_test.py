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

from typing import cast

import numpy as np

from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import TGate, ZeroEffect, ZeroState
from qualtran.bloqs.block_encoding.negate import _negate_block_encoding


def test_negate(bloq_autotester):
    bloq_autotester(_negate_block_encoding)


def test_negate_signature():
    assert _negate_block_encoding().signature == Signature(
        [Register("system", QAny(1)), Register("resource", QAny(1))]
    )


def test_negate_params():
    bloq = _negate_block_encoding()
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 0
    assert bloq.resource_bitsize == 1


def test_negate_tensors():
    bb = BloqBuilder()
    system = cast(Soquet, bb.add_register("system", 1))
    resource = cast(Soquet, bb.add(ZeroState()))
    system, resource = bb.add_t(_negate_block_encoding(), system=system, resource=resource)
    bb.add(ZeroEffect(), q=resource)
    bloq = bb.finalize(system=system)
    from_gate = -TGate().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)
