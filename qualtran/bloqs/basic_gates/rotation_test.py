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
from cirq.ops import SimpleQubitManager

from qualtran.bloqs.basic_gates import Rx, Ry, Rz


def _make_Rx():
    from qualtran.bloqs.basic_gates import Rx

    return Rx(angle=np.pi / 4.0)


def _make_Ry():
    from qualtran.bloqs.basic_gates import Ry

    return Ry(angle=np.pi / 4.0)


def _make_Rz():
    from qualtran.bloqs.basic_gates import Rz

    return Rz(angle=np.pi / 4.0)


def test_rotation_gates():
    angle = np.pi / 4.0
    tcount = 52
    assert Rx(angle).t_complexity().t == tcount
    assert Ry(angle).t_complexity().t == tcount
    assert Rz(angle).t_complexity().t == tcount


def test_as_cirq_op():
    bloq = Rx(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.signature.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rx(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Ry(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.signature.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Ry(rads=bloq.angle).on(cirq.NamedQubit("q")))
    bloq = Rz(angle=np.pi / 4.0, eps=1e-8)
    quregs = bloq.signature.get_cirq_quregs()
    op, _ = bloq.as_cirq_op(SimpleQubitManager(), **quregs)
    circuit = cirq.Circuit(op)
    assert circuit == cirq.Circuit(cirq.Rz(rads=bloq.angle).on(cirq.NamedQubit("q")))
