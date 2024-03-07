import dataclasses
from typing import Union

import attrs
import cirq
import numpy as np
import pytest
import sympy

from qualtran import Bloq, Signature
from qualtran._infra.composite_bloq_test import TestTwoCNOT
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.factoring.mod_exp import ModExp
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.cirq_interop._cirq_to_bloq_test import TestCNOT as TestCNOTCirq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.protos import registers_pb2
from qualtran.serialization import bloq as bloq_serialization
from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.qrom import QROM

def test_qrom():
    array = np.array([1,2,3])
    qrom = QROM.build(array, num_controls=3)
    proto_lib = bloq_serialization.bloqs_to_proto(qrom)
    assert len(proto_lib.table) == 11
    deserialized = bloq_serialization.bloqs_from_proto(proto_lib)
    assert isinstance(deserialized[0],QROM)
    assert deserialized[0].data[0].shape = array[0].shape
if __name__ == "__main__":
    test_qrom()