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
from typing import Dict, Tuple

import attr
import cirq
import numpy as np
import pytest
from attrs import frozen

import qualtran
from qualtran import (
    Bloq,
    BloqBuilder,
    DecomposeNotImplementedError,
    GateWithRegisters,
    Register,
    Side,
    Signature,
)
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.basic_gates import OneState
from qualtran.bloqs.util_bloqs import Allocate, Free, Join, Split
from qualtran.cirq_interop import cirq_optree_to_cbloq, CirqGateAsBloq, CirqQuregT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


@frozen
class TestCNOT(Bloq):
    @property
    def signature(self) -> Signature:
        return Signature.build(control=1, target=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        ctrl, target = soqs['control'], soqs['target']
        ctrl, target = bb.add(CirqGateAsBloq(cirq.CNOT), q=[ctrl, target])
        return {'control': ctrl, 'target': target}

    def as_cirq_op(
        self, qubit_manager: cirq.QubitManager, **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        (control,) = cirq_quregs['control']
        (target,) = cirq_quregs['target']
        return cirq.CNOT(control, target), cirq_quregs


def test_cirq_gate_as_bloq_for_trivial_gates():
    x = CirqGateAsBloq(cirq.X)
    rx = CirqGateAsBloq(cirq.Rx(rads=0.123 * np.pi))
    toffoli = CirqGateAsBloq(cirq.TOFFOLI)

    for b in [x, rx, toffoli]:
        assert len(b.signature) == 1
        assert b.signature[0].side == Side.THRU

    assert x.signature[0].shape == ()
    assert toffoli.signature[0].shape == (3,)

    assert str(x) == 'X'
    assert x.pretty_name() == 'cirq.X'
    assert x.short_name() == 'cirq.X'

    assert rx.pretty_name() == 'cirq.Rx'
    assert rx.short_name() == 'cirq.Rx'

    assert toffoli.pretty_name() == 'cirq.TOFFOLI'
    assert toffoli.short_name() == 'cirq.TOF..'


def test_cirq_gate_as_bloq_tensor_contract_for_and_gate():
    and_gate = And()
    bb = BloqBuilder()
    ctrl = [bb.add(OneState()) for _ in range(2)]
    ctrl, target = bb.add(CirqGateAsBloq(and_gate), ctrl=ctrl)
    cbloq = bb.finalize(ctrl=ctrl, target=target)
    state_vector = cbloq.tensor_contract()
    assert np.isclose(state_vector[7], 1)

    with pytest.raises(NotImplementedError, match="supported only for unitary gates"):
        _ = CirqGateAsBloq(And(uncompute=True)).as_composite_bloq().tensor_contract()


def test_bloq_decompose():
    tb = TestCNOT()
    assert len(tb.signature) == 2
    ctrl, trg = tb.signature
    assert ctrl.bitsize == 1
    assert ctrl.side == Side.THRU
    assert tb.pretty_name() == 'TestCNOT'

    cirq_quregs = get_named_qubits(tb.signature.lefts())
    circuit, _ = tb.decompose_bloq().to_cirq_circuit(**cirq_quregs)
    assert circuit == cirq.Circuit(cirq.CNOT(*cirq_quregs['control'], *cirq_quregs['target']))
    assert tb.t_complexity() == TComplexity(clifford=1)


def test_cirq_circuit_to_cbloq():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)

    circuit.append(cirq.global_phase_operation(-1j))

    cbloq = cirq_optree_to_cbloq(circuit)

    bloq_unitary = cbloq.tensor_contract()
    cirq_unitary = circuit.unitary(qubits)
    np.testing.assert_allclose(cirq_unitary, bloq_unitary, atol=1e-8)


def test_cbloq_to_cirq_circuit():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)
    cbloq = cirq_optree_to_cbloq(circuit)

    # important! we lose moment structure
    circuit = cirq.Circuit(circuit.all_operations())

    # Note: a 1d `shape` bloq register is actually two-dimensional in cirq-world
    # because of the implicit `bitsize` dimension (which must be explicit in cirq-world).
    # CirqGate has registers of bitsize=1 and shape=(n,); hence the list transpose below.
    circuit2, _ = cbloq.to_cirq_circuit(
        **{'qubits': [[q] for q in qubits]}, qubit_manager=cirq.ops.SimpleQubitManager()
    )

    assert circuit == circuit2


def test_cirq_optree_to_cbloq():
    @attr.frozen
    class CirqGateWithRegisters(GateWithRegisters):
        reg: Register

        @property
        def signature(self) -> Signature:
            return Signature([self.reg])

    reg1 = Register('x', shape=(3, 4), bitsize=2)
    reg2 = Register('y', shape=12, bitsize=2)
    anc_reg = Register('anc', shape=4, bitsize=2)
    qubits = cirq.LineQubit.range(24)
    anc_qubits = cirq.NamedQubit.range(4, prefix='anc')
    circuit = cirq.Circuit(
        CirqGateWithRegisters(reg1).on(*qubits),
        CirqGateWithRegisters(anc_reg).on(*anc_qubits, *qubits[:4]),
        CirqGateWithRegisters(reg2).on(*qubits),
    )
    # Test-1: When no signature is specified, the method uses a default signature. Ancilla qubits
    # are also included in the signature itself, so no allocations / deallocations are needed.
    cbloq = cirq_optree_to_cbloq(circuit)
    assert cbloq.signature == qualtran.Signature(
        [qualtran.Register(name='qubits', bitsize=1, shape=(28,))]
    )
    bloq_instances = [binst for binst, _, _ in cbloq.iter_bloqnections()]
    assert all(bloq_instances[i].bloq == Join(2) for i in range(14))
    assert bloq_instances[14].bloq == CirqGateWithRegisters(reg1)
    assert bloq_instances[14].bloq.signature == qualtran.Signature(
        [qualtran.Register(name='x', bitsize=2, shape=(3, 4))]
    )
    assert bloq_instances[15].bloq == CirqGateWithRegisters(anc_reg)
    assert bloq_instances[15].bloq.signature == qualtran.Signature(
        [qualtran.Register(name='anc', bitsize=2, shape=(4,))]
    )
    assert bloq_instances[16].bloq == CirqGateWithRegisters(reg2)
    assert bloq_instances[16].bloq.signature == qualtran.Signature(
        [qualtran.Register(name='y', bitsize=2, shape=(12,))]
    )
    assert all(bloq_instances[-i].bloq == Split(2) for i in range(1, 15))
    # Test-2: If you provide an explicit signature, you must also provide a mapping of cirq qubits
    # matching the signature. The additional ancilla allocations are automatically handled.
    new_signature = qualtran.Signature(
        [
            qualtran.Register('xx', bitsize=3, shape=(3, 2)),
            qualtran.Register('yy', bitsize=1, shape=(2, 3)),
        ]
    )
    cirq_quregs = {
        'xx': np.asarray(qubits[:18]).reshape((3, 2, 3)),
        'yy': np.asarray(qubits[18:]).reshape((2, 3, 1)),
    }
    cbloq = cirq_optree_to_cbloq(
        circuit, signature=new_signature, in_quregs=cirq_quregs, out_quregs=cirq_quregs
    )
    assert cbloq.signature == new_signature
    # Splits, joins, Alloc, Free are automatically inserted.
    bloqs_list = [binst.bloq for binst in cbloq.bloq_instances]
    assert bloqs_list.count(Split(3)) == 6
    assert bloqs_list.count(Join(3)) == 6
    assert bloqs_list.count(Allocate(2)) == 2
    assert bloqs_list.count(Free(2)) == 2


def test_cirq_gate_as_bloq_for_left_only_gates():
    class LeftOnlyGate(GateWithRegisters):
        @property
        def signature(self):
            return Signature([Register('junk', 2, side=Side.LEFT)])

        def decompose_from_registers(self, *, context, junk) -> cirq.OP_TREE:
            yield cirq.CNOT(*junk)
            yield cirq.reset_each(*junk)

    # Using InteropQubitManager enables support for LeftOnlyGate's in CirqGateAsBloq.
    cbloq = CirqGateAsBloq(gate=LeftOnlyGate()).decompose_bloq()
    bloqs_list = [binst.bloq for binst in cbloq.bloq_instances]
    assert bloqs_list.count(Split(2)) == 1
    assert bloqs_list.count(Free(1)) == 2
    assert bloqs_list.count(CirqGateAsBloq(cirq.CNOT)) == 1
    assert bloqs_list.count(CirqGateAsBloq(cirq.ResetChannel())) == 2


def test_cirq_gate_as_bloq_decompose_raises():
    bloq = CirqGateAsBloq(cirq.X)
    with pytest.raises(DecomposeNotImplementedError, match="does not declare a decomposition"):
        _ = bloq.decompose_bloq()
