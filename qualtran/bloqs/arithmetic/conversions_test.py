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

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, QFxp, QUInt
from qualtran.bloqs.arithmetic.conversions import (
    _signed_to_twos,
    _to_contg_index,
    _to_fxp,
    SignedIntegerToTwosComplement,
    ToContiguousIndex,
    ToFxp,
)
from qualtran.bloqs.basic_gates import TGate


def test_to_contigous_index(bloq_autotester):
    bloq_autotester(_to_contg_index)


def test_signed_to_twos(bloq_autotester):
    bloq_autotester(_signed_to_twos)


def test_to_contiguous_index_t_complexity():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('mu', bitsize)
    q1 = bb.add_register('nu', bitsize)
    out = bb.add_register('s', 2 * bitsize)
    q0, q1, out = bb.add(ToContiguousIndex(bitsize, 2 * bitsize), mu=q0, nu=q1, s=out)
    cbloq = bb.finalize(mu=q0, nu=q1, s=out)
    assert cbloq.t_complexity().t == 4 * 29


def test_signed_to_twos_complement_t_complexity():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    q0 = bb.add(SignedIntegerToTwosComplement(bitsize), x=q0)
    cbloq = bb.finalize(x=q0)
    _, sigma = cbloq.call_graph()
    assert sigma[TGate()] == 4 * (5 - 2)


def test_to_fxp(bloq_autotester):
    bloq_autotester(_to_fxp)


def test_to_fxp_checks():
    with pytest.raises(ValueError):
        _ = ToFxp(QUInt(6), QFxp(5, 3))
    with pytest.raises(ValueError):
        _ = ToFxp(QUInt(6), QFxp(6, 3))
    with pytest.raises(ValueError):
        _ = ToFxp(QUInt(6), QFxp(5, 1, signed=True), 1)


def test_to_fxp_correct():
    tot = 1 << 8
    want = np.zeros((tot, tot))
    for a_b in range(tot):
        a, b = a_b >> 4, a_b & ((1 << 4) - 1)
        c = a
        want[(a << 4) | (c ^ b)][a_b] = 1
    got = _to_fxp().tensor_contract()
    np.testing.assert_allclose(got, want)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook("conversions")
