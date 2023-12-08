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

from qualtran import DecomposeTypeError
from qualtran.bloqs.for_testing.atom import TestAtom, TestTwoBitOp


def test_test_atom():
    ta = TestAtom()
    assert ta.short_name() == 'Atom'
    with pytest.raises(DecomposeTypeError):
        ta.decompose_bloq()


def test_test_two_bit_op():
    tba = TestTwoBitOp()
    assert len(tba.signature) == 2
    np.testing.assert_allclose(
        tba.tensor_contract(), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    )
