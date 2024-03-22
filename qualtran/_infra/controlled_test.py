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

import cirq
import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import Bloq, Controlled, CtrlSpec, QBit, QInt, QUInt
from qualtran._infra.gate_with_registers import get_named_qubits, merge_qubits
from qualtran.bloqs.basic_gates import Swap, XGate
from qualtran.bloqs.for_testing import TestAtom, TestParallelCombo, TestSerialCombo
from qualtran.drawing import get_musical_score_data
from qualtran.drawing.musical_score import Circle, SoqData, TextBox


def test_ctrl_spec():
    cspec1 = CtrlSpec()
    assert cspec1 == CtrlSpec(QBit(), cvs=1)

    cspec2 = CtrlSpec(cvs=np.ones(27).reshape((3, 3, 3)))
    assert cspec2.shapes == ((3, 3, 3),)
    assert cspec2 != cspec1

    test_hashable = {cspec1: 1, cspec2: 2}
    assert test_hashable[cspec2] == 2

    cspec3 = CtrlSpec(QInt(64), cvs=np.int64(234234))
    assert cspec3 != cspec1
    assert cspec3.qdtypes[0].num_qubits == 64
    assert cspec3.cvs[0] == 234234
    assert cspec3.cvs[0][tuple()] == 234234
    assert repr(cspec3) == 'CtrlSpec(qdtypes=(QInt(bitsize=64),), cvs=(array(234234),))'


def test_ctrl_spec_shape():
    c1 = CtrlSpec(QBit(), cvs=1)
    c2 = CtrlSpec(QBit(), cvs=(1,))
    assert c1.shapes != c2.shapes
    assert c1 != c2


def test_ctrl_spec_to_cirq_cv_roundtrip():
    cirq_cv = cirq.ProductOfSums([0, 1, 0, 1])
    assert CtrlSpec.from_cirq_cv(cirq_cv) == CtrlSpec(cvs=[0, 1, 0, 1])

    ctrl_specs = [
        CtrlSpec(qdtypes=QUInt(4), cvs=0b0101),
        CtrlSpec(cvs=[0, 1, 0, 1]),
        CtrlSpec(qdtypes=[QBit()] * 4, cvs=[[0], [1], [0], [1]]),
    ]

    for ctrl_spec in ctrl_specs:
        assert ctrl_spec.to_cirq_cv() == cirq_cv.expand()
        assert CtrlSpec.from_cirq_cv(cirq_cv, qdtypes=ctrl_spec.qdtypes, shapes=ctrl_spec.shapes)


def test_ctrl_bloq_as_cirq_op():
    subbloq = XGate()

    def _test_cirq_equivalence(bloq: Bloq, gate: cirq.Gate):
        left_quregs = get_named_qubits(bloq.signature.lefts())
        circuit1, right_quregs = bloq.as_composite_bloq().to_cirq_circuit(**left_quregs)
        circuit2 = cirq.Circuit(
            gate.on(*merge_qubits(bloq.signature, **get_named_qubits(bloq.signature)))
        )
        cirq.testing.assert_same_circuits(circuit1, circuit2)

    # Simple ctrl spec
    _test_cirq_equivalence(subbloq, cirq.X)
    _test_cirq_equivalence(subbloq.controlled(), cirq.X.controlled())

    # Different ways of specifying qubit registers get "expanded" into a flat list of qubits when
    # converting to Cirq.
    cirq_gate = cirq.X.controlled(control_values=[0, 1, 0, 1])
    _test_cirq_equivalence(subbloq.controlled(CtrlSpec(qdtypes=QUInt(4), cvs=0b0101)), cirq_gate)

    _test_cirq_equivalence(subbloq.controlled(CtrlSpec(cvs=[0, 1, 0, 1])), cirq_gate)

    _test_cirq_equivalence(
        subbloq.controlled(CtrlSpec(qdtypes=[QBit()] * 4, cvs=[[0], [1], [0], [1]])), cirq_gate
    )
    # Also works for more complicated bloqs that can decompose into a Cirq circuit.
    bloq = Controlled(Swap(5), CtrlSpec(qdtypes=QUInt(4), cvs=0b0101))
    quregs = get_named_qubits(bloq.signature)
    ctrl, x, y = quregs['ctrl'], quregs['x'], quregs['y']
    circuit1, _ = bloq.decompose_bloq().to_cirq_circuit(**quregs)
    circuit2 = cirq.Circuit(
        cirq.SWAP(x[i], y[i]).controlled_by(*ctrl, control_values=[0, 1, 0, 1]) for i in range(5)
    )
    cirq.testing.assert_same_circuits(circuit1, circuit2)


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


def test_ctrl_spec_activation_4():
    cspec3 = CtrlSpec([QInt(32), QInt(64)], cvs=[np.array(1234), np.array(234234)])
    assert cspec3.is_active(1234, 234234)
    assert not cspec3.is_active(12345, 432432)


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


def test_classical_sim_int_arr():
    ctrl_spec = CtrlSpec(QInt(64), cvs=[1234, 234234])
    bloq = Controlled(XGate(), ctrl_spec=ctrl_spec)

    vals = bloq.call_classically(ctrl=[1234, 234234], q=0)
    np.testing.assert_array_equal(vals[0], (1234, 234234))
    assert vals[1] == 1

    vals = bloq.call_classically(ctrl=[123, 234234], q=0)
    np.testing.assert_array_equal(vals[0], (123, 234234))
    assert vals[1] == 0


def test_classical_sim_int_multi_reg():
    ctrl_spec = CtrlSpec([QInt(32), QInt(64)], cvs=[np.array(1234), np.array(234234)])
    bloq = Controlled(XGate(), ctrl_spec=ctrl_spec)

    vals = bloq.call_classically(ctrl1=1234, ctrl2=234234, q=0)
    assert vals == (1234, 234234, 1)

    vals = bloq.call_classically(ctrl1=123, ctrl2=234234, q=0)
    assert vals == (123, 234234, 0)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('../Controlled')
