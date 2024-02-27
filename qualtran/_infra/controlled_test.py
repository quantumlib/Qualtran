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
from typing import List

import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import Controlled, CtrlSpec, QBit, QInt
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.for_testing import TestAtom, TestParallelCombo, TestSerialCombo
from qualtran.drawing import get_musical_score_data
from qualtran.drawing.musical_score import Circle, SoqData, TextBox


def test_ctrl_spec():
    cspec1 = CtrlSpec()
    assert cspec1 == CtrlSpec(QBit(), cvs=1)

    cspec2 = CtrlSpec(cvs=np.ones(27).reshape((3, 3, 3)))
    assert cspec2.shape == (3, 3, 3)
    assert cspec2 != cspec1

    test_hashable = {cspec1: 1, cspec2: 2}
    assert test_hashable[cspec2] == 2

    cspec3 = CtrlSpec(QInt(64), cvs=np.int64(234234))
    assert cspec3 != cspec1
    assert cspec3.qdtype.num_qubits == 64
    assert cspec3.cvs == 234234
    assert cspec3.cvs[tuple()] == 234234
    assert repr(cspec3) == 'CtrlSpec(QInt(bitsize=64), array(234234))'


def test_ctrl_spec_shape():
    c1 = CtrlSpec(QBit(), cvs=1)
    c2 = CtrlSpec(QBit(), cvs=(1,))
    assert c1.shape != c2.shape
    assert c1 != c2


def test_ctrl_spec_activation_1():
    cspec1 = CtrlSpec()

    assert cspec1.is_active(1)
    assert not cspec1.is_active(0)


def test_ctrl_spec_activation_2():
    cspec2 = CtrlSpec(cvs=np.ones(27).reshape((3, 3, 3)))
    arr = np.ones(27).reshape((3, 3, 3))
    assert cspec2.is_active(arr)
    arr[1, 1, 1] = 0
    assert not cspec2.is_active(arr)
    with pytest.raises(ValueError):
        cspec2.is_active(0)
    with pytest.raises(ValueError):
        cspec2.is_active(np.ones(27))


def test_ctrl_spec_activation_3():
    cspec3 = CtrlSpec(QInt(64), cvs=np.int64(234234))
    assert cspec3.is_active(234234)
    assert not cspec3.is_active(432432)


def test_controlled_serial():
    bloq = Controlled(subbloq=TestSerialCombo(), ctrl_spec=CtrlSpec())
    cbloq = qlt_testing.assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[TestAtom('atom0')]<0>
  LeftDangle.ctrl -> ctrl
  LeftDangle.reg -> q
  ctrl -> C[TestAtom('atom1')]<1>.ctrl
  q -> C[TestAtom('atom1')]<1>.q
--------------------
C[TestAtom('atom1')]<1>
  C[TestAtom('atom0')]<0>.ctrl -> ctrl
  C[TestAtom('atom0')]<0>.q -> q
  ctrl -> C[TestAtom('atom2')]<2>.ctrl
  q -> C[TestAtom('atom2')]<2>.q
--------------------
C[TestAtom('atom2')]<2>
  C[TestAtom('atom1')]<1>.ctrl -> ctrl
  C[TestAtom('atom1')]<1>.q -> q
  ctrl -> RightDangle.ctrl
  q -> RightDangle.reg"""
    )


def test_controlled_parallel():
    bloq = Controlled(subbloq=TestParallelCombo(), ctrl_spec=CtrlSpec())
    cbloq = qlt_testing.assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
Split(dtype=QAny(bitsize=3))<0>
  LeftDangle.reg -> reg
  reg[0] -> C[TestAtom()]<1>.q
  reg[1] -> C[TestAtom()]<2>.q
  reg[2] -> C[TestAtom()]<3>.q
--------------------
C[TestAtom()]<1>
  LeftDangle.ctrl -> ctrl
  Split(dtype=QAny(bitsize=3))<0>.reg[0] -> q
  ctrl -> C[TestAtom()]<2>.ctrl
  q -> Join(dtype=QAny(bitsize=3))<4>.reg[0]
--------------------
C[TestAtom()]<2>
  C[TestAtom()]<1>.ctrl -> ctrl
  Split(dtype=QAny(bitsize=3))<0>.reg[1] -> q
  ctrl -> C[TestAtom()]<3>.ctrl
  q -> Join(dtype=QAny(bitsize=3))<4>.reg[1]
--------------------
C[TestAtom()]<3>
  C[TestAtom()]<2>.ctrl -> ctrl
  Split(dtype=QAny(bitsize=3))<0>.reg[2] -> q
  q -> Join(dtype=QAny(bitsize=3))<4>.reg[2]
  ctrl -> RightDangle.ctrl
--------------------
Join(dtype=QAny(bitsize=3))<4>
  C[TestAtom()]<1>.q -> reg[0]
  C[TestAtom()]<2>.q -> reg[1]
  C[TestAtom()]<3>.q -> reg[2]
  reg -> RightDangle.reg"""
    )


def test_doubly_controlled():
    bloq = Controlled(Controlled(TestAtom(), ctrl_spec=CtrlSpec()), ctrl_spec=CtrlSpec())
    assert (
        bloq.as_composite_bloq().debug_text()
        == """\
C[C[TestAtom()]]<0>
  LeftDangle.ctrl2 -> ctrl2
  LeftDangle.ctrl -> ctrl
  LeftDangle.q -> q
  ctrl2 -> RightDangle.ctrl2
  ctrl -> RightDangle.ctrl
  q -> RightDangle.q"""
    )


def test_bit_vector_ctrl():
    bloq = Controlled(subbloq=TestAtom(), ctrl_spec=CtrlSpec(QBit(), cvs=(1, 0, 1)))
    msd = get_musical_score_data(bloq)
    # select SoqData for the 0th moment, sorted top to bottom
    soqdatas: List[SoqData] = sorted(
        (sd for sd in msd.soqs if sd.rpos.seq_x == 0), key=lambda sd: sd.rpos.y
    )

    # select symbol info
    symbols = [sd.symb for sd in soqdatas]

    # match cvs 1 0 1 and target
    assert symbols == [
        Circle(filled=True),
        Circle(filled=False),
        Circle(filled=True),
        TextBox(text='q'),
    ]


def test_classical_sim_simple():
    ctrl_spec = CtrlSpec()
    bloq = Controlled(XGate(), ctrl_spec=ctrl_spec)
    vals = bloq.call_classically(ctrl=0, q=0)
    assert vals == (0, 0)

    vals = bloq.call_classically(ctrl=1, q=0)
    assert vals == (1, 1)


def test_classical_sim_array():
    ctrl_spec = CtrlSpec(cvs=np.zeros(9).reshape((3, 3)))
    bloq = Controlled(XGate(), ctrl_spec=ctrl_spec)
    ones = np.ones(9).reshape((3, 3))
    ctrl, q = bloq.call_classically(ctrl=ones, q=0)
    np.testing.assert_array_equal(ctrl, ones)
    assert q == 0

    zeros = np.zeros(9).reshape((3, 3))
    ctrl, q = bloq.call_classically(ctrl=zeros, q=0)
    np.testing.assert_array_equal(ctrl, zeros)
    assert q == 1


def test_classical_sim_int():
    ctrl_spec = CtrlSpec(QInt(32), cvs=88)
    bloq = Controlled(XGate(), ctrl_spec=ctrl_spec)
    vals = bloq.call_classically(ctrl=87, q=0)
    assert vals == (87, 0)

    vals = bloq.call_classically(ctrl=88, q=0)
    assert vals == (88, 1)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('../Controlled')
