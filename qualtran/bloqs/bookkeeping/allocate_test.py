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
import pytest

from qualtran import BloqBuilder, QAny
from qualtran.bloqs.bookkeeping import Allocate, Free
from qualtran.bloqs.bookkeeping.allocate import _alloc
from qualtran.testing import execute_notebook


def test_alloc(bloq_autotester):
    bloq_autotester(_alloc)


def test_free_nonzero_state_vector_leads_to_unnormalized_state():
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.on_each import OnEach

    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    qs2 = bb.add(OnEach(10, Hadamard()), q=qs1)
    no_return = bb.add(Free(QAny(10)), reg=qs2)
    assert np.allclose(bb.finalize().tensor_contract(), np.sqrt(1 / 2**10))


@pytest.mark.notebook
def test_notebook():
    execute_notebook('bookkeeping')
