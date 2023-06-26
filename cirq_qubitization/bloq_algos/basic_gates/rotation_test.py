import cirq
import numpy as np
from cirq.ops import SimpleQubitManager

from cirq_qubitization.bloq_algos.basic_gates import Rx, Ry, Rz


def _make_Rx():
    from cirq_qubitization.bloq_algos.basic_gates import Rx

    return Rx(angle=np.pi / 4.0)


def _make_Ry():
    from cirq_qubitization.bloq_algos.basic_gates import Ry

    return Ry(angle=np.pi / 4.0)


def _make_Rz():
    from cirq_qubitization.bloq_algos.basic_gates import Rz

    return Rz(angle=np.pi / 4.0)


def test_rotation_gates():
    angle = np.pi / 4.0
    tcount = 52
    assert Rx(angle).t_complexity().t == tcount
    assert Ry(angle).t_complexity().t == tcount
    assert Rz(angle).t_complexity().t == tcount


def test_as_cirq_op():
    bloq = Rx(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.registers.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rx(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Ry(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.registers.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Ry(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Rz(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.registers.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rz(rads=bloq.angle).on(cirq.NamedQubit("q")))
