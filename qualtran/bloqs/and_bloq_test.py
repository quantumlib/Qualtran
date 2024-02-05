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

import itertools
from functools import cached_property
from typing import Dict

import numpy as np
import pytest
from attrs import frozen

import qualtran.testing as qlt_testing
from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.and_bloq import _and_bloq, _multi_and, And, MultiAnd
from qualtran.bloqs.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from qualtran.drawing import Circle, get_musical_score_data


def test_and_bloq(bloq_autotester):
    bloq_autotester(_and_bloq)


def test_multi_and(bloq_autotester):
    bloq_autotester(_multi_and)


def _iter_and_truth_table(cv1: int, cv2: int):
    # Iterate over And bra/ketted by all possible inputs
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]

    for a, b in itertools.product([0, 1], repeat=2):
        bb = BloqBuilder()
        q_a = bb.add(state[a])
        q_b = bb.add(state[b])
        (q_a, q_b), res = bb.add(And(cv1, cv2), ctrl=[q_a, q_b])
        bb.add(eff[a], q=q_a)
        bb.add(eff[b], q=q_b)
        cbloq = bb.finalize(res=res)
        yield cbloq, a, b


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_truth_table(cv1, cv2):
    for cbloq, a, b in _iter_and_truth_table(cv1, cv2):
        vec = cbloq.tensor_contract()
        if (a == cv1) and (b == cv2):
            np.testing.assert_allclose([0, 1], vec)
        else:
            np.testing.assert_allclose([1, 0], vec)


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_truth_table_classical(cv1, cv2):
    for cbloq, a, b in _iter_and_truth_table(cv1, cv2):
        (res,) = cbloq.call_classically()
        if (a == cv1) and (b == cv2):
            assert res == 1
        else:
            assert res == 0


@pytest.mark.parametrize('cv2', [0, 1])
@pytest.mark.parametrize('cv1', [0, 1])
def test_bad_adjoint(cv1, cv2):
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]
    and_ = And(cv1, cv2, uncompute=True)

    for a, b in itertools.product([0, 1], repeat=2):
        bb = BloqBuilder()
        q_a = bb.add(state[a])
        q_b = bb.add(state[b])
        if (a == cv1) and (b == cv2):
            res = bb.add(ZeroState())
        else:
            res = bb.add(OneState())

        q_a, q_b = bb.add(and_, ctrl=[q_a, q_b], target=res)
        bb.add(eff[a], q=q_a)
        bb.add(eff[b], q=q_b)
        cbloq = bb.finalize()

        val = cbloq.tensor_contract()
        assert np.abs(val) < 1e-8


def test_inverse():
    bb = BloqBuilder()
    q0 = bb.add_register('q0', 1)
    q1 = bb.add_register('q1', 1)
    qs, trg = bb.add(And(), ctrl=[q0, q1])
    qs = bb.add(And(uncompute=True), ctrl=qs, target=trg)
    cbloq = bb.finalize(q0=qs[0], q1=qs[1])

    mat = cbloq.tensor_contract()
    np.testing.assert_allclose(np.eye(4), mat)


def test_multi_truth_table():
    state = [ZeroState(), OneState()]
    eff = [ZeroEffect(), OneEffect()]

    n = 4
    rs = np.random.RandomState(52)
    all_cvs = rs.choice([0, 1], size=(2, n))
    # ctrl_strings = np.array(list(itertools.product([0,1], repeat=n)))
    ctrl_strings = rs.choice([0, 1], size=(10, n))

    for cvs in all_cvs:
        for ctrl_string in ctrl_strings:
            bb = BloqBuilder()
            ctrl_qs = [bb.add(state[c]) for c in ctrl_string]

            ctrl_qs, junk, res = bb.add_from(MultiAnd(cvs), ctrl=ctrl_qs)

            for c, q in zip(ctrl_string, ctrl_qs):
                bb.add(eff[c], q=q)

            cbloq = bb.finalize(junk=junk, res=res)

            # Tensor simulation
            vec = cbloq.tensor_contract()
            should_be = np.all(ctrl_string == cvs)
            *junk_is, res_i = np.where(abs(vec.reshape((2,) * (n - 1))) > 1e-10)
            assert res_i == should_be, ctrl_string

            # Classical simulation
            junk, res = cbloq.call_classically()
            assert res == should_be


def test_multiand_consistent_apply_classical():
    rs = np.random.RandomState(52)
    n = 5
    all_cvs = rs.choice([0, 1], size=(2, n))
    # ctrl_strings = np.array(list(itertools.product([0,1], repeat=n)))
    ctrl_strings = rs.choice([0, 1], size=(10, n))

    for cvs, ctrl_string in itertools.product(all_cvs, ctrl_strings):
        bloq = MultiAnd(cvs=cvs)
        cbloq = bloq.decompose_bloq()

        bloq_classical = bloq.call_classically(ctrl=ctrl_string)
        cbloq_classical = cbloq.call_classically(ctrl=ctrl_string)

        assert len(bloq_classical) == len(cbloq_classical)
        for i in range(len(bloq_classical)):
            np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])


def test_multi_validate():
    with pytest.raises(ValueError):
        _ = MultiAnd(cvs=(0,))
    with pytest.raises(ValueError):
        _ = MultiAnd(cvs=[0])
    with pytest.raises(ValueError):
        _ = MultiAnd(cvs=(0, 0))


def test_notebook():
    qlt_testing.execute_notebook('and_bloq')


@frozen
class AndIdentity(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q0=1, q1=1)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q0: 'SoquetT', q1: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        qs, trg = bb.add(And(), ctrl=[q0, q1])
        q0, q1 = bb.add(And(uncompute=True), ctrl=qs, target=trg)
        return {'q0': q0, 'q1': q1}


def test_and_identity_bloq():
    bloq = AndIdentity()
    np.testing.assert_allclose(np.eye(4), bloq.tensor_contract())
    np.testing.assert_allclose(np.eye(4), bloq.decompose_bloq().tensor_contract())


def test_and_musical_score():
    msd = get_musical_score_data(And(cv1=1, cv2=1))
    # soq[0] and [1] are the dangling symbols
    assert msd.soqs[2].symb == Circle(filled=True)
    assert msd.soqs[3].symb == Circle(filled=True)

    msd = get_musical_score_data(And(cv1=1, cv2=0))
    # soq[0] and [1] are the dangling symbols
    assert msd.soqs[2].symb == Circle(filled=True)
    assert msd.soqs[3].symb == Circle(filled=False)

    msd = get_musical_score_data(And(cv1=0, cv2=0))
    # soq[0] and [1] are the dangling symbols
    assert msd.soqs[2].symb == Circle(filled=False)
    assert msd.soqs[3].symb == Circle(filled=False)


def test_multiand_adjoint():
    bb = BloqBuilder()
    q0 = bb.add_register('q0', 1)
    q1 = bb.add_register('q1', 1)
    q2 = bb.add_register('q2', 1)

    qs, junk, trg = bb.add(MultiAnd((1, 1, 1)), ctrl=[q0, q1, q2])
    qs = bb.add(MultiAnd((1, 1, 1)).adjoint(), ctrl=qs, target=trg, junk=junk)

    cbloq = bb.finalize(q0=qs[0], q1=qs[1], q2=qs[2])
    qlt_testing.assert_valid_cbloq(cbloq)

    ret = cbloq.call_classically(q0=1, q1=1, q2=1)
    assert ret == (1, 1, 1)
