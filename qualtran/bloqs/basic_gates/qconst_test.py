#  Copyright 2026 Google LLC
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
from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates.qconst import (
    _qint_effect,
    _qint_state,
    _quint_effect,
    _quint_state,
    QIntEffect,
    QIntState,
    QUIntEffect,
    QUIntState,
)


def test_qint_state_autotester(bloq_autotester):
    bloq_autotester(_qint_state)


def test_qint_effect_autotester(bloq_autotester):
    bloq_autotester(_qint_effect)


def test_quint_state_autotester(bloq_autotester):
    bloq_autotester(_quint_state)


def test_quint_effect_autotester(bloq_autotester):
    bloq_autotester(_quint_effect)


def test_qint_state_manual():
    # Valid signed values for 8-bit signed integer: [-128, 127]
    k = QIntState(-5, bitsize=8)
    assert str(k) == 'QIntState(-5)'
    (val,) = k.call_classically()
    assert val == -5

    # Test boundaries
    assert QIntState(127, bitsize=8).call_classically() == (127,)
    assert QIntState(-128, bitsize=8).call_classically() == (-128,)

    with pytest.raises(ValueError):
        QIntState(128, bitsize=8)

    with pytest.raises(ValueError):
        QIntState(-129, bitsize=8)

    # Test decomposition and tensor contracting
    qlt_testing.assert_valid_bloq_decomposition(k)
    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_qint_effect_manual():
    k = QIntEffect(-5, bitsize=8)
    assert str(k) == 'QIntEffect(-5)'
    ret = k.call_classically(val=-5)
    assert ret == ()

    with pytest.raises(AssertionError):
        k.call_classically(val=-6)

    qlt_testing.assert_valid_bloq_decomposition(k)
    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_quint_state_manual():
    # Valid unsigned values for 8-bit unsigned integer: [0, 255]
    k = QUIntState(5, bitsize=8)
    assert str(k) == 'QUIntState(5)'
    (val,) = k.call_classically()
    assert val == 5

    # Test boundaries
    assert QUIntState(255, bitsize=8).call_classically() == (255,)
    assert QUIntState(0, bitsize=8).call_classically() == (0,)

    with pytest.raises(ValueError):
        QUIntState(256, bitsize=8)

    with pytest.raises(ValueError):
        QUIntState(-1, bitsize=8)

    qlt_testing.assert_valid_bloq_decomposition(k)
    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_quint_effect_manual():
    k = QUIntEffect(5, bitsize=8)
    assert str(k) == 'QUIntEffect(5)'
    ret = k.call_classically(val=5)
    assert ret == ()

    with pytest.raises(AssertionError):
        k.call_classically(val=6)

    qlt_testing.assert_valid_bloq_decomposition(k)
    np.testing.assert_allclose(k.tensor_contract(), k.decompose_bloq().tensor_contract())


def test_state_effect_adjoint():
    state = QIntState(-10, bitsize=6)
    effect = QIntEffect(-10, bitsize=6)
    assert state.adjoint() == effect
    assert effect.adjoint() == state

    state_u = QUIntState(10, bitsize=6)
    effect_u = QUIntEffect(10, bitsize=6)
    assert state_u.adjoint() == effect_u
    assert effect_u.adjoint() == state_u


def test_state_effect_composition():
    bb = BloqBuilder()
    q0 = bb.add(QIntState(-10, bitsize=6))
    bb.add(QIntEffect(-10, bitsize=6), val=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()
    np.testing.assert_allclose(val, 1.0)

    bb = BloqBuilder()
    q0 = bb.add(QUIntState(10, bitsize=6))
    bb.add(QUIntEffect(10, bitsize=6), val=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()
    np.testing.assert_allclose(val, 1.0)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('qconst')
