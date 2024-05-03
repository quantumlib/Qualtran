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
from typing import Optional, Sequence, Tuple

import attrs
import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import BloqBuilder, BoundedQUInt, QBit, Register, SoquetT
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle


@frozen
class TestPrepareOracle(PrepareOracle):
    r"""Gate that prepares a fixed state with real amplitudes using a MatrixGate.

    It is an $n$-qubit unitary $U$ such that

    $$
        U |0^n\rangle> = \sum_{x = 0}^{2^n-1} \sqrt{\alpha_x} |x\rangle
    $$

    where $\alpha_x \ge 0$ such that $\sum_x \alpha_x = 1$.

    Args:
        U: A `MatrixGate` whose matrix is the unitary of the oracle.

    Registers:
        selection: $n$-qubit register
    """

    U: MatrixGate

    def __attrs_post_init__(self):
        # check that first column is all reals
        column = np.array(self.U.matrix)[:, 0]
        np.testing.assert_almost_equal(np.imag(column), 0)

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register('selection', BoundedQUInt(bitsize=self.U.bitsize)),)

    @property
    def l1_norm_of_coeffs(self) -> float:
        return 1.0

    @classmethod
    def random(cls, bitsize: int, *, random_state: np.random.RandomState):
        """Generate a random unitary s.t. the first column has all real amplitudes"""
        matrix = MatrixGate.random(bitsize, random_state=random_state).matrix
        matrix = np.array(matrix)

        # make the first column (weights \sqrt{alpha_i}) all reals
        column = matrix[:, 0]
        matrix = matrix * (column.conj() / np.abs(column))[:, None]

        return cls(MatrixGate(bitsize, matrix))

    def build_composite_bloq(self, bb: BloqBuilder, selection: SoquetT) -> dict[str, SoquetT]:
        selection = bb.add(self.U, q=selection)
        return {'selection': selection}

    @cached_property
    def alphas(self):
        return np.array(self.U.matrix)[:, 0] ** 2


@frozen
class TestPauliSelectOracle(SelectOracle):
    r"""Paulis acting on $m$ qubits, controlled by an $n$-qubit register.

    Given $2^n$ multi-qubit-Paulis (acting on $m$ qubits) $U_j$,
    this gate implements the following $m + n$-qubit unitary:

    $$
        \sum_{j = 0}^{2^n - 1} |j \rangle\langle j| \otimes U_j
    $$


    Args:
        select_bitsize: $n$-qubit control register that selects the $i$-th Pauli
        target_bitsize: $m$-qubit register on which the Paulis act
        select_unitaries: A sequence of $2^n$ multi-qubit-Paulis, the $i$-th one acting when the selection register is $i$.
        control_val: optional control bit.

    Registers:
        control: (optional) one qubit register if `control_val` is not None
        selection: $n$-qubit integer register
        target: $m$-qubit register
    """

    select_bitsize: int
    target_bitsize: int
    select_unitaries: tuple[cirq.DensePauliString, ...]
    control_val: Optional[int] = None

    @classmethod
    def random(
        cls, select_bitsize: int, target_bitsize: int, *, random_state: np.random.RandomState
    ) -> 'TestPauliSelectOracle':
        dps = tuple(
            cirq.DensePauliString(random_state.randint(0, 4, size=target_bitsize))
            for _ in range(2**select_bitsize)
        )
        return cls(select_bitsize, target_bitsize, dps)

    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', BoundedQUInt(bitsize=self.select_bitsize)),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', BoundedQUInt(bitsize=self.target_bitsize)),)

    def adjoint(self):
        return self

    def __pow__(self, power):
        if abs(power) == 1:
            return self
        return NotImplemented

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values=None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'cirq.Gate':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and isinstance(control_values[0], int)
            and len(control_values) == 1
            and self.control_val is None
        ):
            return attrs.evolve(self, control_val=control_values[0])
        raise NotImplementedError()

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        selection: NDArray[cirq.Qid],  # type: ignore[type-var]
        target: NDArray[cirq.Qid],  # type: ignore[type-var]
        **quregs: NDArray[cirq.Qid],  # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        if self.control_val is not None:
            selection = np.concatenate([selection, quregs['control']])

        for cv, U in enumerate(self.select_unitaries):
            bits = tuple(map(int, bin(cv)[2:].zfill(self.select_bitsize)))[::-1]
            if self.control_val is not None:
                bits = (*bits, self.control_val)
            yield U.on(*target).controlled_by(*selection, control_values=bits)


def random_qubitization_walk_operator(
    select_bitsize: int, target_bitsize: int, *, random_state: np.random.RandomState
) -> tuple[QubitizationWalkOperator, cirq.PauliSum]:
    r"""Szegedy Walk operator for a randomly generated Hamiltonian $H$ of $2^n$ $m$-qubit Paulis.

    $$
        H = \sum_{j = 0}^{2^n - 1} \alpha_j U_j
    $$

    where $U_j$ are randomly selected $m$-qubit Paulis
    and $\alpha_j \ge 0$ such that $\sum_j \alpha_j = 1$ are randomly chosen weights.

    Args:
        select_bitsize: number of qubits $n$ of the selection register
        target_bitsize: number of qubits $m$ that Hamiltonian $H$ acts on
        random_state: optional random state

    Returns:
        WalkOperator for $H$, and the Hamiltonian $H$.
    """
    prepare = TestPrepareOracle.random(select_bitsize, random_state=random_state)
    select = TestPauliSelectOracle.random(select_bitsize, target_bitsize, random_state=random_state)

    np.testing.assert_allclose(np.linalg.norm(prepare.alphas, 1), 1)

    ham = cirq.PauliSum.from_pauli_strings(
        [
            dp.on(*cirq.LineQubit.range(target_bitsize)) * alpha
            for dp, alpha in zip(select.select_unitaries, prepare.alphas)
        ]
    )

    return QubitizationWalkOperator(prepare=prepare, select=select), ham
