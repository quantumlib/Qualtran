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
from typing import cast

import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import Adjoint, CompositeBloq, Side
from qualtran._infra.adjoint import _adjoint_cbloq
from qualtran.bloqs.basic_gates import CNOT, ZeroState
from qualtran.bloqs.for_testing.atom import TestAtom
from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph
from qualtran.bloqs.for_testing.with_decomposition import TestParallelCombo, TestSerialCombo
from qualtran.drawing import LarrowTextBox, RarrowTextBox, Text


def test_serial_combo_adjoint():
    # The normal decomposition is three `TestAtom` tagged atom{0,1,2}.
    assert (
        TestSerialCombo().decompose_bloq().debug_text()
        == """\
TestAtom('atom0')<0>
  LeftDangle.reg -> q
  q -> TestAtom('atom1')<1>.q
--------------------
TestAtom('atom1')<1>
  TestAtom('atom0')<0>.q -> q
  q -> TestAtom('atom2')<2>.q
--------------------
TestAtom('atom2')<2>
  TestAtom('atom1')<1>.q -> q
  q -> RightDangle.reg"""
    )

    # The adjoint reverses the order to atom{2,1,0} and wraps each in `Adjoint`.
    assert (
        TestSerialCombo().adjoint().decompose_bloq().debug_text()
        == """\
Adjoint(subbloq=TestAtom('atom2'))<0>
  LeftDangle.reg -> q
  q -> Adjoint(subbloq=TestAtom('atom1'))<1>.q
--------------------
Adjoint(subbloq=TestAtom('atom1'))<1>
  Adjoint(subbloq=TestAtom('atom2'))<0>.q -> q
  q -> Adjoint(subbloq=TestAtom('atom0'))<2>.q
--------------------
Adjoint(subbloq=TestAtom('atom0'))<2>
  Adjoint(subbloq=TestAtom('atom1'))<1>.q -> q
  q -> RightDangle.reg"""
    )


def test_cbloq_adjoint_function():
    cbloq = qlt_testing.assert_valid_bloq_decomposition(TestSerialCombo())
    assert isinstance(cbloq, CompositeBloq)
    adj_cbloq = _adjoint_cbloq(cbloq)
    assert isinstance(adj_cbloq, CompositeBloq)
    qlt_testing.assert_valid_cbloq(adj_cbloq)

    assert (
        adj_cbloq.debug_text()
        == """\
Adjoint(subbloq=TestAtom('atom2'))<0>
  LeftDangle.reg -> q
  q -> Adjoint(subbloq=TestAtom('atom1'))<1>.q
--------------------
Adjoint(subbloq=TestAtom('atom1'))<1>
  Adjoint(subbloq=TestAtom('atom2'))<0>.q -> q
  q -> Adjoint(subbloq=TestAtom('atom0'))<2>.q
--------------------
Adjoint(subbloq=TestAtom('atom0'))<2>
  Adjoint(subbloq=TestAtom('atom1'))<1>.q -> q
  q -> RightDangle.reg"""
    )


def test_adjoint_signature():
    cnot = CNOT()
    adj = Adjoint(cnot)  # specifically use the Adjoint wrapper for testing
    assert cnot.signature == adj.signature  # all thru

    zero = ZeroState()
    adj = Adjoint(zero)  # specifically use the Adjoint wrapper for testing
    assert len(zero.signature) == len(adj.signature)
    (reg,) = zero.signature
    (adj_reg,) = adj.signature
    assert reg.name == adj_reg.name
    assert reg.side == Side.RIGHT
    assert adj_reg.side == Side.LEFT


def test_adjoint_adjoint():
    zero = ZeroState()
    adj = Adjoint(zero)  # specifically use the Adjoint wrapper for testing
    assert adj.adjoint() == zero


def test_bloq_counts():
    n = sympy.Symbol('_n0')
    bc = Adjoint(TestBloqWithCallGraph()).bloq_counts()
    assert bc == {
        Adjoint(subbloq=TestAtom()): n,
        Adjoint(subbloq=TestSerialCombo()): 1,
        Adjoint(subbloq=TestParallelCombo()): 1,
    }


def test_call_graph():
    graph, _ = Adjoint(TestBloqWithCallGraph()).call_graph()
    edge_strs = {f'{repr(caller)} -> {repr(callee)}' for caller, callee in graph.edges}
    assert edge_strs == {
        'Adjoint(subbloq=TestBloqWithCallGraph()) -> Adjoint(subbloq=TestAtom())',
        'Adjoint(subbloq=TestBloqWithCallGraph()) -> Adjoint(subbloq=TestParallelCombo())',
        'Adjoint(subbloq=TestBloqWithCallGraph()) -> Adjoint(subbloq=TestSerialCombo())',
        'Adjoint(subbloq=TestParallelCombo()) -> Join(dtype=QAny(bitsize=3))',
        'Adjoint(subbloq=TestParallelCombo()) -> Split(dtype=QAny(bitsize=3))',
        'Adjoint(subbloq=TestParallelCombo()) -> Adjoint(subbloq=TestAtom())',
        "Adjoint(subbloq=TestSerialCombo()) -> Adjoint(subbloq=TestAtom('atom0'))",
        "Adjoint(subbloq=TestSerialCombo()) -> Adjoint(subbloq=TestAtom('atom1'))",
        "Adjoint(subbloq=TestSerialCombo()) -> Adjoint(subbloq=TestAtom('atom2'))",
    }


def test_names():
    atom = TestAtom()
    assert str(atom) == "TestAtom"
    assert cast(Text, atom.wire_symbol(reg=None)).text == "TestAtom"

    adj_atom = Adjoint(atom)
    assert str(adj_atom) == "Adjoint(subbloq=TestAtom)"
    assert cast(Text, adj_atom.wire_symbol(reg=None)).text == "TestAtom†"


def test_wire_symbol():
    zero = ZeroState()
    (reg,) = zero.signature
    adj = Adjoint(zero)  # specifically use the Adjoint wrapper for testing

    ws = zero.wire_symbol(reg)
    adj_ws = adj.wire_symbol(reg.adjoint())
    assert isinstance(ws, LarrowTextBox)
    assert isinstance(adj_ws, RarrowTextBox)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('../Adjoint')
