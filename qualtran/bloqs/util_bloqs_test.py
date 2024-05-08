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
from typing import Dict, Type, Union

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QAny, QFxp, QInt, Register, Side, Signature, Soquet, SoquetT
from qualtran._infra.gate_with_registers import GateWithRegisters, get_named_qubits
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.for_testing import TestAtom, TestCastToFrom, TestMultiRegister
from qualtran.bloqs.for_testing.atom import TestGWRAtom
from qualtran.bloqs.util_bloqs import Allocate, Cast, Free, Join, Partition, Power, Split
from qualtran.simulation.classical_sim import call_cbloq_classically
from qualtran.simulation.tensor import bloq_to_dense, cbloq_to_quimb
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


@pytest.mark.parametrize('n', [5, 123])
@pytest.mark.parametrize('bloq_cls', [Split, Join])
def test_register_sizes_add_up(bloq_cls: Union[Type[Split], Type[Join]], n):
    bloq = bloq_cls(QAny(n))
    for name, group_regs in bloq.signature.groups():
        if any(reg.side is Side.THRU for reg in group_regs):
            assert not any(reg.side != Side.THRU for reg in group_regs)
            continue

        lefts = [reg for reg in group_regs if reg.side & Side.LEFT]
        left_size = np.prod([l.total_bits() for l in lefts])
        rights = [reg for reg in group_regs if reg.side & Side.RIGHT]
        right_size = np.prod([r.total_bits() for r in rights])

        assert left_size > 0
        assert left_size == right_size


def test_util_bloqs():
    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    assert isinstance(qs1, Soquet)
    qs2 = bb.add(Split(QAny(10)), reg=qs1)
    assert qs2.shape == (10,)
    qs3 = bb.add(Join(QAny(10)), reg=qs2)
    assert isinstance(qs3, Soquet)
    no_return = bb.add(Free(QAny(10)), reg=qs3)
    assert no_return is None
    assert bb.finalize().tensor_contract() == 1.0


def test_free_nonzero_state_vector_leads_to_unnormalized_state():
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.on_each import OnEach

    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    qs2 = bb.add(OnEach(10, Hadamard()), q=qs1)
    no_return = bb.add(Free(QAny(10)), reg=qs2)
    assert np.allclose(bb.finalize().tensor_contract(), np.sqrt(1 / 2**10))


def test_util_bloqs_tensor_contraction():
    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    qs2 = bb.add(Split(QAny(10)), reg=qs1)
    qs3 = bb.add(Join(QAny(10)), reg=qs2)
    cbloq = bb.finalize(out=qs3)
    expected = np.zeros(2**10)
    expected[0] = 1
    np.testing.assert_allclose(cbloq.tensor_contract(), expected)


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


def test_partition():
    bloq = TestPartition(test_bloq=CNOT())
    assert_valid_bloq_decomposition(bloq)

    bloq = TestPartition(test_bloq=TestMultiRegister())
    assert_valid_bloq_decomposition(bloq)


def test_partition_tensor_contract():
    bloq = TestPartition(test_bloq=TestMultiRegister())
    tn, _ = cbloq_to_quimb(bloq.decompose_bloq())
    assert len(tn.tensors) == 3
    assert tn.shape == (4096, 4096)


def test_partition_as_cirq_op():
    bloq = TestPartition(test_bloq=CNOT())
    quregs = get_named_qubits(bloq.signature.lefts())
    op, quregs = bloq.as_cirq_op(cirq.ops.SimpleQubitManager(), **quregs)
    assert op is not None
    unitary = cirq.unitary(cirq.Circuit(op))
    assert np.allclose(unitary, bloq_to_dense(CNOT()))

    bloq = TestPartition(test_bloq=TestMultiRegister())
    circuit, _ = bloq.decompose_bloq().to_cirq_circuit(
        cirq.ops.SimpleQubitManager(), test_regs=cirq.NamedQubit.range(12, prefix='system')
    )
    assert (
        circuit.to_text_diagram(transpose=True)
        == """\
system0 system1  system2  system3  system4  system5  system6  system7  system8  system9 system10 system11
│       │        │        │        │        │        │        │        │        │       │        │
xx──────yy[0, 0]─yy[0, 1]─yy[1, 0]─yy[1, 1]─yy[0, 0]─yy[0, 1]─yy[1, 0]─yy[1, 1]─zz──────zz───────zz
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


def test_classical_sim():
    bb = BloqBuilder()
    x = bb.allocate(4)
    xs = bb.split(x)
    xs_1_orig = xs[1]  # keep a copy for later
    xs[1] = bb.add(XGate(), q=xs[1])
    y = bb.join(xs)
    cbloq = bb.finalize(y=y)

    ret, assign = call_cbloq_classically(cbloq.signature, vals={}, binst_graph=cbloq._binst_graph)
    assert assign[x] == 0

    assert assign[xs[0]] == 0
    assert assign[xs_1_orig] == 0
    assert assign[xs[2]] == 0
    assert assign[xs[3]] == 0

    assert assign[xs[1]] == 1
    assert assign[y] == 4

    assert ret == {'y': 4}


def test_classical_sim_dtypes():
    s = Split(QAny(8))
    (xx,) = s.call_classically(reg=255)
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    # TODO: Re-enable when Split/Join have real data types
    #  https://github.com/quantumlib/Qualtran/issues/446
    # with pytest.raises(ValueError):
    #     _ = s.call_classically(reg=256)

    # with numpy types
    (xx,) = s.call_classically(reg=np.uint8(255))
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    # Warning: numpy will wrap too-large values
    (xx,) = s.call_classically(reg=np.uint8(256))
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    # TODO: Re-enable when Split/Join have real data types
    #  https://github.com/quantumlib/Qualtran/issues/446
    # with pytest.raises(ValueError):
    #     _ = s.call_classically(reg=np.uint16(256))


def test_cast_tensor_contraction():
    bloq = TestCastToFrom()
    tn, _ = cbloq_to_quimb(bloq.decompose_bloq())
    assert len(tn.tensors) == 3
    assert tn.shape == (2**4,) * 4


def test_cast_classical_sim():
    c = Cast(QInt(8), QFxp(8, 8))
    (y,) = c.call_classically(reg=7)
    assert y == 7
    bloq = TestCastToFrom()
    (a, b) = bloq.call_classically(a=7, b=2)
    assert a == 7
    assert b == 9


def test_power():
    with pytest.raises(ValueError, match="THRU"):
        from qualtran.bloqs.mcmt import And

        _ = Power(And(), 2)

    bloq = TestMultiRegister()
    with pytest.raises(ValueError, match="positive"):
        _ = Power(bloq, -2)

    bloq_raised_to_power = Power(bloq, 10)
    assert bloq_raised_to_power.signature == bloq.signature
    cbloq = bloq_raised_to_power.decompose_bloq()
    assert [binst.bloq for binst, _, _ in cbloq.iter_bloqnections()] == [bloq] * 10


def test_power_of_power():
    bloq = TestAtom()
    assert Power(bloq, 6) == Power(bloq, 2) ** 3

    gate = TestGWRAtom()
    assert gate**-3 == gate.adjoint() ** 3
    assert gate**6 == (gate**2) ** 3
    assert gate**6 == (gate**-2) ** -3


def test_power_circuit_diagram():
    def to_cirq_circuit(bloq: GateWithRegisters) -> cirq.Circuit:
        op = bloq.on(*cirq.LineQubit.range(bloq.num_qubits()))
        return cirq.Circuit(op)

    power_atom = Power(TestGWRAtom(), 4)

    cirq.testing.assert_has_diagram(to_cirq_circuit(power_atom), '0: ───TestGWRAtom^4───')

    power_multi_reg = Power(TestMultiRegister(), 4)
    cirq.testing.assert_has_diagram(
        to_cirq_circuit(power_multi_reg),
        '''
0: ────Power^4───
       │
1: ────yy^4──────
       │
2: ────yy^4──────
       │
3: ────yy^4──────
       │
4: ────yy^4──────
       │
5: ────yy^4──────
       │
6: ────yy^4──────
       │
7: ────yy^4──────
       │
8: ────yy^4──────
       │
9: ────zz^4──────
       │
10: ───zz^4──────
       │
11: ───zz^4──────''',
    )


@pytest.mark.notebook
def test_notebook():
    execute_notebook('util_bloqs')
