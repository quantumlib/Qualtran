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

import pytest

from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.mod_arithmetic import CModAddK, CtrlScaleModAdd, ModAdd, ModAddK
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.testing import assert_valid_bloq_decomposition


def identity_map(n: int):
    """Returns a dict of size `2**n` mapping each integer in range [0, 2**n) to itself."""
    return {i: i for i in range(2**n)}


def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match="must be between"):
        _ = ModAddK(3, 10)
    add_one = ModAddK(3, 5, 1)
    add_two = ModAddK(3, 5, 2, cvs=[1, 0])

    assert add_one == ModAddK(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cvs == (1, 0)


def add_constant_mod_n_ref_t_complexity_(b: ModAddK) -> TComplexity:
    # Rough cost as given in https://arxiv.org/abs/1905.09749
    return 5 * Add(QUInt(b.bitsize)).t_complexity()


@pytest.mark.parametrize('bitsize', [3, 9])
def test_add_mod_n_gate_counts(bitsize):
    bloq = ModAddK(bitsize, mod=8, add_val=2, cvs=[0, 1])
    assert bloq.t_complexity() == add_constant_mod_n_ref_t_complexity_(bloq)


def test_ctrl_scale_mod_add():
    bloq = CtrlScaleModAdd(k=123, mod=13 * 17, bitsize=8)

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 8


def test_ctrl_mod_add_k():
    bloq = CModAddK(k=123, mod=13 * 17, bitsize=8)

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 5


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_mod_add_valid_decomp(bitsize, p):
    bloq = ModAdd(bitsize=bitsize, mod=p)
    assert_valid_bloq_decomposition(bloq)
