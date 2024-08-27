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

from functools import cached_property
from typing import Dict

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QAny, Register, Signature, Soquet, SoquetT
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.bookkeeping.partition import _partition
from qualtran.bloqs.for_testing import TestMultiRegister
from qualtran.simulation.tensor import bloq_to_dense, cbloq_to_quimb
from qualtran.testing import assert_valid_bloq_decomposition


def test_partition(bloq_autotester):
    bloq_autotester(_partition)


def test_partition_check():
    with pytest.raises(ValueError):
        _ = Partition(n=0, regs=())
    with pytest.raises(ValueError):
        _ = Partition(n=1, regs=(Register('x', QAny(2)),))
    with pytest.raises(ValueError):
        _ = Partition(n=4, regs=(Register('x', QAny(1)), Register('x', QAny(3))))


@frozen
class TestPartition(Bloq):
    test_bloq: Bloq

    @cached_property
    def bitsize(self):
        return sum(reg.total_bits() for reg in self.test_bloq.signature)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(test_regs=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', test_regs: 'SoquetT') -> Dict[str, 'Soquet']:
        bloq_regs = self.test_bloq.signature
        partition = Partition(self.bitsize, bloq_regs)  # type: ignore[arg-type]
        out_regs = bb.add(partition, x=test_regs)
        out_regs = bb.add(self.test_bloq, **{reg.name: sp for reg, sp in zip(bloq_regs, out_regs)})
        test_regs = bb.add(
            partition.adjoint(), **{reg.name: sp for reg, sp in zip(bloq_regs, out_regs)}
        )
        return {'test_regs': test_regs}


def test_partition_wrapper():
    bloq = TestPartition(test_bloq=CNOT())
    assert_valid_bloq_decomposition(bloq)

    bloq = TestPartition(test_bloq=TestMultiRegister())
    assert_valid_bloq_decomposition(bloq)


def test_partition_wrapper_tensor_contract():
    bloq = TestPartition(test_bloq=TestMultiRegister())
    tn = cbloq_to_quimb(bloq.as_composite_bloq().flatten())
    assert tn.shape == (2,) * (2 * int(np.log2(4096)))

    tn = tn.rank_simplify()
    for tensor in tn.tensors:
        assert np.prod(tensor.shape) <= 2**4


def test_partition_wrapper_as_cirq_op():
    bloq = TestPartition(test_bloq=CNOT())
    quregs = get_named_qubits(bloq.signature.lefts())
    op, quregs = bloq.as_cirq_op(cirq.ops.SimpleQubitManager(), **quregs)
    assert op is not None
    unitary = cirq.unitary(cirq.Circuit(op))
    assert np.allclose(unitary, bloq_to_dense(CNOT()))

    bloq = TestPartition(test_bloq=TestMultiRegister())
    circuit = bloq.decompose_bloq().to_cirq_circuit(
        qubit_manager=cirq.ops.SimpleQubitManager(),
        cirq_quregs={'test_regs': cirq.NamedQubit.range(12, prefix='system')},
    )
    assert (
        circuit.to_text_diagram(transpose=True)
        == """\
system0 system1  system2  system3  system4  system5  system6  system7  system8  system9 system10 system11
│       │        │        │        │        │        │        │        │        │       │        │
xx──────yy[0, 0]─yy[0, 0]─yy[0, 1]─yy[0, 1]─yy[1, 0]─yy[1, 0]─yy[1, 1]─yy[1, 1]─zz──────zz───────zz
│       │        │        │        │        │        │        │        │        │       │        │"""
    )


def test_partition_call_classically():
    regs = (Register('xx', QAny(2), shape=(2, 2)), Register('yy', QAny(3)))
    bitsize = sum(reg.total_bits() for reg in regs)
    bloq = Partition(n=bitsize, regs=regs)
    out = bloq.call_classically(x=64)
    flat_out = np.concatenate([v.ravel() if isinstance(v, np.ndarray) else [v] for v in out])
    # 6th set bit == 64
    assert flat_out[2] == 2
    out = bloq.adjoint().call_classically(**{reg.name: val for (reg, val) in zip(regs, out)})
    assert out[0] == 64
