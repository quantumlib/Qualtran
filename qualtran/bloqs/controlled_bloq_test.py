import pytest

from qualtran.bloqs.controlled_bloq import ControlledBloq
from qualtran.components.composite_bloq_test import Atom, TestParallelBloq, TestSerialBloq
from qualtran.testing import assert_valid_bloq_decomposition


def test_controlled_serial():
    bloq = ControlledBloq(subbloq=TestSerialBloq())
    cbloq = assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[Atom()]<0>
  LeftDangle.control -> control
  LeftDangle.stuff -> stuff
  control -> C[Atom()]<1>.control
  stuff -> C[Atom()]<1>.stuff
--------------------
C[Atom()]<1>
  C[Atom()]<0>.control -> control
  C[Atom()]<0>.stuff -> stuff
  control -> C[Atom()]<2>.control
  stuff -> C[Atom()]<2>.stuff
--------------------
C[Atom()]<2>
  C[Atom()]<1>.control -> control
  C[Atom()]<1>.stuff -> stuff
  control -> RightDangle.control
  stuff -> RightDangle.stuff"""
    )


def test_controlled_parallel():
    bloq = ControlledBloq(subbloq=TestParallelBloq())
    cbloq = assert_valid_bloq_decomposition(bloq)
    assert (
        cbloq.debug_text()
        == """\
C[Split(n=3)]<0>
  LeftDangle.control -> control
  LeftDangle.stuff -> split
  control -> C[Atom()]<1>.control
  split[0] -> C[Atom()]<1>.stuff
  split[1] -> C[Atom()]<2>.stuff
  split[2] -> C[Atom()]<3>.stuff
--------------------
C[Atom()]<1>
  C[Split(n=3)]<0>.control -> control
  C[Split(n=3)]<0>.split[0] -> stuff
  control -> C[Atom()]<2>.control
  stuff -> C[Join(n=3)]<4>.join[0]
--------------------
C[Atom()]<2>
  C[Atom()]<1>.control -> control
  C[Split(n=3)]<0>.split[1] -> stuff
  control -> C[Atom()]<3>.control
  stuff -> C[Join(n=3)]<4>.join[1]
--------------------
C[Atom()]<3>
  C[Atom()]<2>.control -> control
  C[Split(n=3)]<0>.split[2] -> stuff
  control -> C[Join(n=3)]<4>.control
  stuff -> C[Join(n=3)]<4>.join[2]
--------------------
C[Join(n=3)]<4>
  C[Atom()]<3>.control -> control
  C[Atom()]<3>.stuff -> join[2]
  C[Atom()]<1>.stuff -> join[0]
  C[Atom()]<2>.stuff -> join[1]
  control -> RightDangle.control
  join -> RightDangle.stuff"""
    )


def test_doubly_controlled():
    with pytest.raises(NotImplementedError):
        # TODO: https://github.com/quantumlib/cirq-qubitization/issues/149
        ControlledBloq(ControlledBloq(Atom()))
