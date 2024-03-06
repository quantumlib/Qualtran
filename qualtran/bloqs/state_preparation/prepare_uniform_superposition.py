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
from typing import Tuple

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Signature
from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd


@attrs.frozen
class PrepareUniformSuperposition(GateWithRegisters):
    r"""Prepares a uniform superposition over first $n$ basis states using $O(log(n))$ T-gates.

    Performs a single round of amplitude amplification and prepares a uniform superposition over
    the first $n$ basis states $|0>, |1>, ..., |n - 1>$. The expected T-complexity should be
    $10 * log(L) + 2 * K$ T-gates and $2$ single qubit rotation gates, where $n = L * 2^K$.

    However, the current T-complexity is $12 * log(L)$ T-gates and $2 + 2 * (K + log(L))$ rotations
    because of two open issues:
     - https://github.com/quantumlib/Qualtran/issues/233 and
     - https://github.com/quantumlib/Qualtran/issues/235

    Args:
        n: The gate prepares a uniform superposition over first $n$ basis states.
        cvs: Control values for each control qubit. If specified, a controlled version
            of the gate is constructed.

    References:
        See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
    """

    n: int
    cvs: Tuple[int, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=len(self.cvs), target=(self.n - 1).bit_length())

    def short_name(self) -> str:
        return r'$\sum_l |l\rangle$'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        control_symbols = ["@" if cv else "@(0)" for cv in self.cvs]
        target_symbols = ['target'] * self.signature.get_left('target').total_bits()
        target_symbols[0] = f"UNIFORM({self.n})"
        return cirq.CircuitDiagramInfo(wire_symbols=control_symbols + target_symbols)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        controls, target = quregs.get('ctrl', ()), quregs['target']
        # Find K and L as per https://arxiv.org/abs/1805.03662 Fig 12.
        n, k = self.n, 0
        while n > 1 and n % 2 == 0:
            k += 1
            n = n // 2
        l, logL = int(n), self.signature.get_left('target').total_bits() - k
        logL_qubits = target[:logL]

        yield [
            op.controlled_by(*controls, control_values=self.cvs) for op in cirq.H.on_each(*target)
        ]
        if not len(logL_qubits):
            return

        ancilla = context.qubit_manager.qalloc(1)
        theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=ancilla)
        yield cirq.Rz(rads=theta)(*ancilla)
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=ancilla) ** -1
        context.qubit_manager.qfree(ancilla)

        yield cirq.H.on_each(*logL_qubits)

        and_ancilla = context.qubit_manager.qalloc(len(self.cvs) + logL - 2)
        and_target = context.qubit_manager.qalloc(1)
        and_cv = (0,) * logL + self.cvs
        ctrl = np.asarray([*logL_qubits, *controls])[:, np.newaxis]
        junk = np.asarray(and_ancilla)[:, np.newaxis]
        if len(and_cv) <= 2:
            and_op = And(*and_cv).on_registers(ctrl=ctrl, target=and_target)
        else:
            and_op = MultiAnd(cvs=and_cv).on_registers(ctrl=ctrl, junk=junk, target=and_target)

        yield and_op
        yield cirq.Rz(rads=theta)(*and_target)
        yield cirq.inverse(and_op)

        yield cirq.H.on_each(*logL_qubits)
        context.qubit_manager.qfree([*and_target, *and_ancilla])


@bloq_example
def _prep_uniform() -> PrepareUniformSuperposition:
    prep_uniform = PrepareUniformSuperposition(n=5)
    return prep_uniform


@bloq_example
def _c_prep_uniform() -> PrepareUniformSuperposition:
    c_prep_uniform = PrepareUniformSuperposition(n=5, cvs=[1])
    return c_prep_uniform


_PREP_UNIFORM_DOC = BloqDocSpec(
    bloq_cls=PrepareUniformSuperposition,
    import_line='from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition',
    examples=(_prep_uniform, _c_prep_uniform),
)
