#  Copyright 2026 Google LLC
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

import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import (
    CHadamard,
    CRy,
    CRz,
    CSwap,
    CYGate,
    Discard,
    Identity,
    MeasureX,
    MeasureZ,
    OnEach,
    Rx,
    Ry,
    Rz,
    SGate,
    SU2RotationGate,
    Swap,
    TGate,
    TwoBitCSwap,
    TwoBitSwap,
    XGate,
    XPowGate,
    YGate,
    YPowGate,
    ZeroState,
    ZPowGate,
)


def test_qcalls():
    bb = BloqBuilder()

    # Allocate qubits
    q0 = bb.add(ZeroState())
    q1 = bb.add(ZeroState())
    q2 = bb.add(ZeroState())

    # 1. Hadamard & CHadamard
    q0, q1 = CHadamard.qcall(q0, q1)

    # 2. SGate & TGate
    q0 = SGate.qcall(q0)
    q1 = TGate.qcall(q1, is_adjoint=True)

    # 3. Rotations
    q0 = ZPowGate.qcall(q0, exponent=0.5)
    q1 = XPowGate.qcall(q1, exponent=0.25)
    q2 = YPowGate.qcall(q2, exponent=0.125)

    # 4. Rz, Rx, Ry, CRz, CRy
    q0 = Rz.qcall(q0, angle=np.pi / 4)
    q1 = Rx.qcall(q1, angle=np.pi / 8)
    q2 = Ry.qcall(q2, angle=np.pi / 16)
    q0, q1 = CRz.qcall(q0, q1, angle=np.pi / 2)
    q1, q2 = CRy.qcall(q1, q2, angle=np.pi / 4)

    # 5. SU2RotationGate
    q0 = SU2RotationGate.qcall(q0, theta=0.1, phi=0.2, lambd=0.3)

    # 6. YGate & CYGate & XGate
    q0 = YGate.qcall(q0)
    q0, q1 = CYGate.qcall(q0, q1)

    # 7. Swap gates
    q0, q1 = TwoBitSwap.qcall(q0, q1)
    q0, q1, q2 = TwoBitCSwap.qcall(q0, q1, q2)

    # 8. Measure & Effects
    c0 = MeasureZ.qcall(q0)
    c1 = MeasureX.qcall(q1)

    # Discards
    Discard.qcall(c0)
    Discard.qcall(c1)


def test_templated_and_parameterized_qcalls():
    bb = BloqBuilder()

    # Allocate multi-qubit registers
    qvar_x = bb.add_register("x", bitsize=4)
    qvar_y = bb.add_register("y", bitsize=4)

    # 1. Identity
    qvar_x = Identity.qcall(qvar_x)

    # 2. OnEach
    qvar_x = OnEach.qcall(qvar_x, gate=XGate())

    # 4. Swap & CSwap on registers
    qvar_x, qvar_y = Swap.qcall(qvar_x, qvar_y)

    # CSwap needs a control qubit
    ctrl = bb.add(ZeroState())
    ctrl, qvar_x, qvar_y = CSwap.qcall(ctrl, qvar_x, qvar_y)
