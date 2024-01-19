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

import qualtran.testing as qlt_testing
from qualtran import ControlledBloq
from qualtran.bloqs.for_testing import TestAtom, TestParallelCombo, TestSerialCombo


def test_controlled_serial():
    bloq = ControlledBloq(subbloq=TestSerialCombo())
    cbloq = qlt_testing.assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[TestAtom('atom0')]<0>
  LeftDangle.control -> control
  LeftDangle.reg -> q
  control -> C[TestAtom('atom1')]<1>.control
  q -> C[TestAtom('atom1')]<1>.q
--------------------
C[TestAtom('atom1')]<1>
  C[TestAtom('atom0')]<0>.control -> control
  C[TestAtom('atom0')]<0>.q -> q
  control -> C[TestAtom('atom2')]<2>.control
  q -> C[TestAtom('atom2')]<2>.q
--------------------
C[TestAtom('atom2')]<2>
  C[TestAtom('atom1')]<1>.control -> control
  C[TestAtom('atom1')]<1>.q -> q
  control -> RightDangle.control
  q -> RightDangle.reg"""
    )


def test_controlled_parallel():
    bloq = ControlledBloq(subbloq=TestParallelCombo())
    cbloq = qlt_testing.assert_valid_bloq_decomposition(bloq)
    print()
    print(cbloq.debug_text())
    print()
    assert (
        cbloq.debug_text()
        == """\
C[Split(n=3)]<0>
  LeftDangle.control -> control
  LeftDangle.reg -> reg
  control -> C[TestAtom()]<1>.control
  reg[0] -> C[TestAtom()]<1>.q
  reg[1] -> C[TestAtom()]<2>.q
  reg[2] -> C[TestAtom()]<3>.q
--------------------
C[TestAtom()]<1>
  C[Split(n=3)]<0>.control -> control
  C[Split(n=3)]<0>.reg[0] -> q
  control -> C[TestAtom()]<2>.control
  q -> C[Join(n=3)]<4>.reg[0]
--------------------
C[TestAtom()]<2>
  C[TestAtom()]<1>.control -> control
  C[Split(n=3)]<0>.reg[1] -> q
  control -> C[TestAtom()]<3>.control
  q -> C[Join(n=3)]<4>.reg[1]
--------------------
C[TestAtom()]<3>
  C[TestAtom()]<2>.control -> control
  C[Split(n=3)]<0>.reg[2] -> q
  control -> C[Join(n=3)]<4>.control
  q -> C[Join(n=3)]<4>.reg[2]
--------------------
C[Join(n=3)]<4>
  C[TestAtom()]<3>.control -> control
  C[TestAtom()]<3>.q -> reg[2]
  C[TestAtom()]<1>.q -> reg[0]
  C[TestAtom()]<2>.q -> reg[1]
  control -> RightDangle.control
  reg -> RightDangle.reg"""
    )


def test_doubly_controlled():
    with pytest.raises(NotImplementedError):
        # TODO: https://github.com/quantumlib/Qualtran/issues/149
        ControlledBloq(ControlledBloq(TestAtom()))


def test_notebook():
    qlt_testing.execute_notebook('../Controlled')
