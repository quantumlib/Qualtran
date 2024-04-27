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
from functools import cached_property
from typing import Optional, Tuple

import attrs
import cirq
import scipy
from numpy.typing import NDArray

from qualtran import Signature
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.resource_counting.symbolic_counting_utils import SymbolicFloat


@attrs.frozen
class PrepareUniformSuperpositionTest(PrepareOracle):
    n: int
    cvs: Tuple[int, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    qlambda: Optional[float] = None

    @cached_property
    def selection_registers(self) -> Signature:
        return Signature.build(selection=(self.n - 1).bit_length())

    @cached_property
    def junk_registers(self) -> Signature:
        return Signature.build()

    @cached_property
    def l1_norm_of_coeffs(self) -> Optional['SymbolicFloat']:
        return self.qlambda

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        yield PrepareUniformSuperposition(self.n, self.cvs).on_registers(target=quregs['selection'])


def get_uniform_pauli_qubitized_walk(target_bitsize: int):
    ham = cirq.PauliSum()
    paulis = [cirq.X, cirq.Y, cirq.Z]
    q = cirq.LineQubit.range(target_bitsize)
    for i in range(target_bitsize):
        ham += paulis[i % 3].on(q[i]) * paulis[(i + 1) % 3].on(q[(i + 1) % target_bitsize])
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    ham_dps = [ps.dense(q) for ps in ham]

    assert scipy.linalg.ishermitian(ham.matrix())
    prepare = PrepareUniformSuperpositionTest(len(ham_coeff), qlambda=sum(ham_coeff))
    select = SelectPauliLCU(
        (len(ham_coeff) - 1).bit_length(), select_unitaries=ham_dps, target_bitsize=target_bitsize
    )
    return ham, QubitizationWalkOperator(select=select, prepare=prepare)
