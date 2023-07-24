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

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import (
    IntEffect,
    IntState,
    OneEffect,
    OneState,
    ZeroEffect,
    ZeroState,
)


def _make_zero_state():
    from qualtran.bloqs.basic_gates import ZeroState

    return ZeroState()


def test_zero_state():
    bloq = ZeroState()
    assert str(bloq) == 'ZeroState(n=1)'
    assert not bloq.bit
    vector = bloq.tensor_contract()
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    (x,) = bloq.call_classically()
    assert x == 0


def test_multiq_zero_state():
    # Verifying the attrs trickery that I can plumb through *some*
    # of the attributes but pre-specify others.
    with pytest.raises(NotImplementedError):
        _ = ZeroState(n=10)


def test_one_state():
    bloq = OneState()
    assert bloq.bit
    assert bloq.state
    vector = bloq.tensor_contract()
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    (x,) = bloq.call_classically()
    assert x == 1


def test_zero_effect():
    bloq = ZeroEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 0])
    np.testing.assert_allclose(should_be, vector)

    ret = bloq.call_classically(q=0)
    assert ret == ()

    with pytest.raises(AssertionError):
        bloq.call_classically(q=1)

    with pytest.raises(ValueError, match=r'.*q should be an integer, not \[0\, 0\, 0\]'):
        bloq.call_classically(q=[0, 0, 0])


def test_one_effect():
    bloq = OneEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([0, 1])
    np.testing.assert_allclose(should_be, vector)

    ret = bloq.call_classically(q=1)
    assert ret == ()

    with pytest.raises(AssertionError):
        bloq.call_classically(q=0)


@pytest.mark.parametrize('bit', [False, True])
def test_zero_state_effect(bit):
    bb = BloqBuilder()

    if bit:
        state = OneState()
        eff = OneEffect()
    else:
        state = ZeroState()
        eff = ZeroEffect()

    q0 = bb.add(state)
    bb.add(eff, q=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)

    res = cbloq.call_classically()
    assert res == ()


def test_int_state():
    k = IntState(255, bitsize=8)
    assert k.short_name() == '255'
    assert k.pretty_name() == '|255>'
    (val,) = k.call_classically()
    assert val == 255

    with pytest.raises(ValueError):
        _ = IntState(255, bitsize=7)
    with pytest.raises(ValueError):
        _ = IntState(-1, bitsize=8)

    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_int_effect():
    k = IntEffect(255, bitsize=8)
    assert k.short_name() == '255'
    assert k.pretty_name() == '<255|'
    ret = k.call_classically(val=255)
    assert ret == ()

    with pytest.raises(AssertionError):
        k.call_classically(val=245)
