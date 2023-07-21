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

import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import PlusEffect, PlusState


def _make_plus_state():
    from qualtran.bloqs.basic_gates import PlusState

    return PlusState()


def test_plus_state():
    bloq = PlusState()
    vector = bloq.tensor_contract()
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def _make_plus_effect():
    from qualtran.bloqs.basic_gates import PlusEffect

    return PlusEffect()


def test_plus_effect():
    bloq = PlusEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def test_plus_state_effect():
    bb = BloqBuilder()

    q0 = bb.add(PlusState())
    bb.add(PlusEffect(), q=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)
