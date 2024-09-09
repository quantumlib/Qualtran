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

from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import CZPowGate, Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate
from qualtran.bloqs.basic_gates.rotation import _rx, _ry, _rz
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.resource_counting.classify_bloqs import bloq_is_rotation, bloq_is_t_like


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
    bloq = ZPowGate(exponent=1 / 5, global_shift=-0.5)
    quregs = get_named_qubits(bloq.signature)
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    assert op is not None
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(
        cirq.ZPowGate(exponent=1 / 5, global_shift=-0.5).on(cirq.NamedQubit("q"))
    )


def test_pretty_name():
    assert str(ZPowGate()) == "Z**1.0"
    assert str(XPowGate()) == "X**1.0"
    assert str(YPowGate()) == "Y**1.0"
    assert str(_ry()) == "Ry(0.7853981633974483)"
    assert str(_rx()) == "Rx(0.7853981633974483)"
    assert str(_rz()) == "Rz(0.7853981633974483)"

    assert str(CZPowGate(1.0)) == 'CZ**1.0'
    assert str(CZPowGate(0.9)) == 'CZ**0.9'


def test_rx(bloq_autotester):
    bloq_autotester(_rx)


def test_ry(bloq_autotester):
    bloq_autotester(_ry)


def test_rz(bloq_autotester):
    bloq_autotester(_rz)
