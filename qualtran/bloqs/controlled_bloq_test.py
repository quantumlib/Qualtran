import pytest

from qualtran._infra.composite_bloq_test import Atom, TestParallelBloq, TestSerialBloq
from qualtran.bloqs.controlled_bloq import ControlledBloq
from qualtran.testing import assert_valid_bloq_decomposition


def test_controlled_serial():
    bloq = ControlledBloq(subbloq=TestSerialBloq())
    cbloq = assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[Atom()]<0>
  LeftDangle.ctrl -> ctrl
  LeftDangle.stuff -> stuff
  ctrl -> C[Atom()]<1>.ctrl
  stuff -> C[Atom()]<1>.stuff
--------------------
C[Atom()]<1>
  C[Atom()]<0>.ctrl -> ctrl
  C[Atom()]<0>.stuff -> stuff
  ctrl -> C[Atom()]<2>.ctrl
  stuff -> C[Atom()]<2>.stuff
--------------------
C[Atom()]<2>
  C[Atom()]<1>.ctrl -> ctrl
  C[Atom()]<1>.stuff -> stuff
  ctrl -> RightDangle.ctrl
  stuff -> RightDangle.stuff"""
    )


def test_controlled_parallel():
    bloq = ControlledBloq(subbloq=TestParallelBloq())
    cbloq = assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[Split(n=3)]<0>
  LeftDangle.ctrl -> ctrl
  LeftDangle.stuff -> split
  ctrl -> C[Atom()]<1>.ctrl
  split[0] -> C[Atom()]<1>.stuff
  split[1] -> C[Atom()]<2>.stuff
  split[2] -> C[Atom()]<3>.stuff
--------------------
C[Atom()]<1>
  C[Split(n=3)]<0>.ctrl -> ctrl
  C[Split(n=3)]<0>.split[0] -> stuff
  ctrl -> C[Atom()]<2>.ctrl
  stuff -> C[Join(n=3)]<4>.join[0]
--------------------
C[Atom()]<2>
  C[Atom()]<1>.ctrl -> ctrl
  C[Split(n=3)]<0>.split[1] -> stuff
  ctrl -> C[Atom()]<3>.ctrl
  stuff -> C[Join(n=3)]<4>.join[1]
--------------------
C[Atom()]<3>
  C[Atom()]<2>.ctrl -> ctrl
  C[Split(n=3)]<0>.split[2] -> stuff
  ctrl -> C[Join(n=3)]<4>.ctrl
  stuff -> C[Join(n=3)]<4>.join[2]
--------------------
C[Join(n=3)]<4>
  C[Atom()]<3>.ctrl -> ctrl
  C[Atom()]<3>.stuff -> join[2]
  C[Atom()]<1>.stuff -> join[0]
  C[Atom()]<2>.stuff -> join[1]
  ctrl -> RightDangle.ctrl
  join -> RightDangle.stuff"""
    )


def test_doubly_controlled():
    with pytest.raises(NotImplementedError):
        # TODO: https://github.com/quantumlib/cirq-qubitization/issues/149
        ControlledBloq(ControlledBloq(Atom()))
