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

import cirq
import numpy as np
import pytest
from cirq.ops import SimpleQubitManager

from qualtran import BloqBuilder, Controlled, CtrlSpec
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import (
    CZ,
    CZPowGate,
    GlobalPhase,
    Rx,
    Ry,
    Rz,
    SGate,
    TGate,
    XPowGate,
    YPowGate,
    ZGate,
    ZPowGate,
)
from qualtran.bloqs.basic_gates.rotation import _crz, _cz_pow, _rx, _ry, _rz, _z_pow, CRz
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.resource_counting.classify_bloqs import bloq_is_rotation, bloq_is_t_like


def test_zpow_gate(bloq_autotester):
    bloq_autotester(_z_pow)


def test_zpow_is_controlled_gphase():
    rs = np.random.RandomState(52)
    t = rs.uniform(0, 2)
    cgphase = GlobalPhase(exponent=t).controlled().tensor_contract()
    zpow = ZPowGate(exponent=t).tensor_contract()
    np.testing.assert_allclose(zpow, cgphase)

    a = GlobalPhase(exponent=t).tensor_contract()
    manual_cgphase = np.diag([1, a])
    np.testing.assert_allclose(zpow, manual_cgphase)


def test_zpow_from_rz():
    rs = np.random.RandomState(52)
    t = rs.uniform(0, 2)

    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(Rz(angle=t * np.pi), q=q)
    bb.add(GlobalPhase(exponent=t / 2))
    rz_with_phase = bb.finalize(q=q)

    np.testing.assert_allclose(ZPowGate(t).tensor_contract(), rz_with_phase.tensor_contract())


def test_rz_from_zpow():
    rs = np.random.RandomState(52)
    theta = rs.uniform(0, 2 * np.pi)

    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(ZPowGate(exponent=theta / np.pi), q=q)
    bb.add(GlobalPhase(exponent=-theta / (2 * np.pi)))
    zpow_with_phase = bb.finalize(q=q)

    np.testing.assert_allclose(Rz(angle=theta).tensor_contract(), zpow_with_phase.tensor_contract())


def test_zpow_special_exponents():
    zpow_1 = ZPowGate(exponent=1)
    np.testing.assert_allclose(ZGate().tensor_contract(), zpow_1.tensor_contract())

    zpow_half = ZPowGate(exponent=0.5)
    np.testing.assert_allclose(SGate().tensor_contract(), zpow_half.tensor_contract())

    zpow_quarter = ZPowGate(exponent=0.25)
    np.testing.assert_allclose(TGate().tensor_contract(), zpow_quarter.tensor_contract())


def test_czpow(bloq_autotester):
    bloq_autotester(_cz_pow)


def test_czpow_tensor():
    rs = np.random.RandomState(52)
    t = rs.uniform(0, 2)
    u1 = CZPowGate(exponent=t).tensor_contract()
    u2 = cirq.unitary(cirq.ZPowGate(exponent=t).controlled())
    u3 = Controlled(ZPowGate(exponent=t), CtrlSpec()).tensor_contract()
    np.testing.assert_allclose(u1, u2, atol=1e-8)
    np.testing.assert_allclose(u1, u3, atol=1e-8)


def test_czpow_special_exponents():
    czpow_1 = CZPowGate(exponent=1)
    np.testing.assert_allclose(CZ().tensor_contract(), czpow_1.tensor_contract())


def test_czpow_from_controlled_z_pow():
    rs = np.random.RandomState(52)
    t = rs.uniform(0, 2)
    zpow = ZPowGate(exponent=t)
    assert zpow.controlled() == CZPowGate(exponent=t)

    cbloq = Controlled(zpow.as_composite_bloq(), CtrlSpec()).decompose_bloq()
    (czpow_inst,) = list(cbloq.bloq_instances)
    assert czpow_inst.bloq == CZPowGate(exponent=t)
    np.testing.assert_allclose(CZPowGate(exponent=t).tensor_contract(), cbloq.tensor_contract())


def test_crz(bloq_autotester):
    bloq_autotester(_crz)


def test_crz_tensor():
    rs = np.random.RandomState(52)
    angle = rs.uniform(0, 2 * np.pi)
    u1 = CRz(angle=angle).tensor_contract()
    u2 = cirq.unitary(cirq.Rz(rads=angle).controlled())
    u3 = Controlled(Rz(angle=angle), CtrlSpec()).tensor_contract()
    np.testing.assert_allclose(u1, u2, atol=1e-8)
    np.testing.assert_allclose(u1, u3, atol=1e-8)


def test_crz_from_controlled_rz():
    rs = np.random.RandomState()
    angle = rs.uniform(0, 2 * np.pi)
    rz = Rz(angle=angle)
    assert rz.controlled() == CRz(angle=angle)

    cbloq = Controlled(rz.as_composite_bloq(), CtrlSpec()).decompose_bloq()
    (crz_inst,) = list(cbloq.bloq_instances)
    assert crz_inst.bloq == CRz(angle=angle)
    np.testing.assert_allclose(CRz(angle=angle).tensor_contract(), cbloq.tensor_contract())


def test_t_like_rotation_gates():
    angle = np.pi / 4.0
    # In prior versions of the library, only Rz(pi/4) would simplify to a T gate in gate counts.
    # The others would report the synthesis cost for an arbitrary angle, which was reported as
    # 52 T-gates.
    assert not bloq_is_rotation(Rx(angle))
    assert not bloq_is_rotation(Ry(angle))
    assert not bloq_is_rotation(Rz(angle))
    assert bloq_is_t_like(Rx(angle))
    assert bloq_is_t_like(Ry(angle))
    assert bloq_is_t_like(Rz(angle))

    assert get_cost_value(Rx(angle), QECGatesCost()) == GateCounts(t=1)
    assert get_cost_value(Ry(angle), QECGatesCost()) == GateCounts(t=1)
    assert get_cost_value(Rz(angle), QECGatesCost()) == GateCounts(t=1)

    assert Rx(angle).t_complexity().t_incl_rotations() == 1
    assert Ry(angle).t_complexity().t_incl_rotations() == 1
    assert Rz(angle).t_complexity().t_incl_rotations() == 1


@pytest.mark.parametrize(
    "bloq",
    [Rx(0.01), Ry(0.01), Rz(0.01), ZPowGate(0.01), YPowGate(0.01), XPowGate(0.01), CZPowGate(0.01)],
)
def test_rotation_gates_adjoint(bloq):
    assert type(bloq) == type(bloq.adjoint())
    np.testing.assert_allclose(
        bloq.tensor_contract() @ bloq.adjoint().tensor_contract(),
        np.identity(2 ** bloq.signature.n_qubits()),
        atol=1e-8,
    )


def test_as_cirq_op():
    bloq = Rx(angle=np.pi / 4.0, eps=1e-8)
    quregs = get_named_qubits(bloq.signature.lefts())
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rx(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Ry(angle=np.pi / 4.0, eps=1e-8)
    quregs = get_named_qubits(bloq.signature.lefts())
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Ry(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Rz(angle=np.pi / 4.0, eps=1e-8)
    quregs = get_named_qubits(bloq.signature.lefts())
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rz(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = XPowGate(exponent=1 / 5, global_shift=-0.5)
    quregs = get_named_qubits(bloq.signature)
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(
        cirq.XPowGate(exponent=1 / 5, global_shift=-0.5).on(cirq.NamedQubit("q"))
    )
    bloq = YPowGate(exponent=1 / 5, global_shift=-0.5)
    quregs = get_named_qubits(bloq.signature)
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(
        cirq.YPowGate(exponent=1 / 5, global_shift=-0.5).on(cirq.NamedQubit("q"))
    )
    bloq = ZPowGate(exponent=1 / 5)
    quregs = get_named_qubits(bloq.signature)
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.ZPowGate(exponent=1 / 5).on(cirq.NamedQubit("q")))


def test_pl_interop():
    import pennylane as qml

    bloq = Rx(angle=0.4)
    pl_op_from_bloq = bloq.as_pl_op(wires=[0])
    pl_op = qml.RX(phi=0.4, wires=[0])
    assert pl_op_from_bloq == pl_op

    matrix = pl_op.matrix()
    should_be = bloq.tensor_contract()
    np.testing.assert_allclose(should_be, matrix)

    bloq = Ry(angle=0.4)
    pl_op_from_bloq = bloq.as_pl_op(wires=[0])
    pl_op = qml.RY(phi=0.4, wires=[0])
    assert pl_op_from_bloq == pl_op

    matrix = pl_op.matrix()
    should_be = bloq.tensor_contract()
    np.testing.assert_allclose(should_be, matrix)

    bloq = Rz(angle=0.4)
    pl_op_from_bloq = bloq.as_pl_op(wires=[0])
    pl_op = qml.RZ(phi=0.4, wires=[0])
    assert pl_op_from_bloq == pl_op

    matrix = pl_op.matrix()
    should_be = bloq.tensor_contract()
    np.testing.assert_allclose(should_be, matrix)


def test_str():
    assert str(ZPowGate()) == "Z**1.0"
    assert str(XPowGate()) == "X**1.0"
    assert str(YPowGate()) == "Y**1.0"
    assert str(_ry()) == "Ry(0.7853981633974483)"
    assert str(_rx()) == "Rx(0.7853981633974483)"
    assert str(_rz()) == "Rz(a)"

    assert str(CZPowGate(1.0)) == 'CZ**1.0'
    assert str(CZPowGate(0.9)) == 'CZ**0.9'


def test_rx(bloq_autotester):
    bloq_autotester(_rx)


def test_ry(bloq_autotester):
    bloq_autotester(_ry)


def test_rz(bloq_autotester):
    bloq_autotester(_rz)
