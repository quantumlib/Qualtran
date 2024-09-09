#  Copyright 2024 Google LLC
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
from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, Controlled, CtrlSpec
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class ControlledViaAnd(Controlled):
    """Reduces a generic controlled bloq to a singly-controlled bloq using an And ladder.

    Implements a generic controlled version of the subbloq, by first reducing the
    arbitrary control to a single qubit, and then using a single-qubit-controlled
    variant of the subbloq.

    For signature, see :class:`Controlled`.

    Args:
        subbloq: The bloq we are controlling.
        ctrl_spec: The specification for how to control the bloq.
    """

    subbloq: Bloq
    ctrl_spec: CtrlSpec

    def _is_single_bit_control(self) -> bool:
        return self.ctrl_spec.num_qubits == 1

    @cached_property
    def _single_control_value(self) -> int:
        assert self._is_single_bit_control()
        return self.ctrl_spec._cvs_tuple[0]

    def adjoint(self) -> 'ControlledViaAnd':
        return ControlledViaAnd(self.subbloq.adjoint(), self.ctrl_spec)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        # compute the control bit
        if self._is_single_bit_control():
            ctrl_soqs = None
            q = soqs.pop(self.ctrl_reg_names[0])
            if self.ctrl_spec.shapes[0] == (1,):
                assert isinstance(q, np.ndarray)
                q = q[0]

            if self._single_control_value == 0:
                q = bb.add(XGate(), q=q)
        else:
            # map input control soqs to the CtrlSpecAnd controls
            ctrl_spec_and_bloq = CtrlSpecAnd(self.ctrl_spec)
            ctrl_soqs = {
                inner_ctrl_reg.name: soqs.pop(name)
                for name, inner_ctrl_reg in zip(
                    self.ctrl_reg_names, ctrl_spec_and_bloq.control_registers
                )
            }

            # compute the control bit
            ctrl_soqs = bb.add_d(CtrlSpecAnd(self.ctrl_spec), **ctrl_soqs)
            q = ctrl_soqs.pop('target')

        # add single-controlled-subbloq
        _, adder = self.subbloq.get_ctrl_system(CtrlSpec())
        out_q_iter, out_soqs_iter = adder(bb, [q], soqs)
        (q,) = tuple(out_q_iter)
        soqs = dict(zip([reg.name for reg in self.subbloq.signature], out_soqs_iter))

        # uncompute the control bit
        if self._is_single_bit_control():
            if self._single_control_value == 0:
                q = bb.add(XGate(), q=q)

            if self.ctrl_spec.shapes[0] == (1,):
                q = np.array([q])
            soqs[self.ctrl_reg_names[0]] = q
        else:
            # uncompute the control bit
            ctrl_spec_and_bloq = CtrlSpecAnd(self.ctrl_spec)
            assert ctrl_soqs is not None
            ctrl_soqs['target'] = q
            ctrl_soqs = bb.add_d(ctrl_spec_and_bloq.adjoint(), **ctrl_soqs)

            # map the control soqs back to the original bloq's control soqs
            soqs |= {
                name: ctrl_soqs[inner_ctrl_reg.name]
                for name, inner_ctrl_reg in zip(
                    self.ctrl_reg_names, ctrl_spec_and_bloq.control_registers
                )
            }

        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts: Counter[Bloq] = Counter()
        counts[self.subbloq.controlled()] += 1

        if self._is_single_bit_control():
            if self._single_control_value == 0:
                counts[XGate()] += 2
        else:
            ctrl = CtrlSpecAnd(self.ctrl_spec)
            counts[ctrl] += 1
            counts[ctrl.adjoint()] += 1

        return counts


@bloq_example
def _controlled_via_and_qbits() -> ControlledViaAnd:
    from qualtran.bloqs.basic_gates import Hadamard

    controlled_via_and_qbits = ControlledViaAnd(Hadamard(), CtrlSpec(cvs=(np.array([0, 1, 1, 0]),)))
    return controlled_via_and_qbits


@bloq_example
def _controlled_via_and_ints() -> ControlledViaAnd:
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.basic_gates import Hadamard

    controlled_via_and_ints = ControlledViaAnd(
        Hadamard(),
        CtrlSpec(
            qdtypes=(QUInt(4), QInt(4)), cvs=(np.array([0, 1, 2, 3]), np.array([0, 1, -1, -2]))
        ),
    )
    return controlled_via_and_ints


_CONTROLLED_VIA_AND_DOC = BloqDocSpec(
    bloq_cls=ControlledViaAnd, examples=[_controlled_via_and_ints, _controlled_via_and_qbits]
)
