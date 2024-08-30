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

import abc
from functools import cached_property
from typing import Iterator, Sequence, Tuple

import cirq
import numpy as np
from cirq._compat import cached_method
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqDocSpec,
    BQUInt,
    GateWithRegisters,
    QAny,
    QUInt,
    Register,
    Signature,
)
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.data_loading.qrom import QROM


class ProgrammableRotationGateArrayBase(GateWithRegisters):
    """Base class for a composite gate to apply multiplexed rotations on target register.

    Programmable rotation gate array is used to apply `M` rotations on a target register,
    potentially interleaved with `M - 1` arbitrary unitaries, via the following steps:

        - Represent each rotation angle `theta` as a an integer approximation of `B` bits.
        - Use an ancilla register of size `kappa` (a configurable parameter) and load multiplexed
          bits of rotations angles, in batches of size at-most `kappa`, using `QROM` reads.
            - Thus, a total of `⌈M * B / kappa⌉ + 1` QROM reads are required.
        - After every QROM read, apply `kappa` (singly) controlled rotations on the target
          register, each controlled on a different qubit from the `kappa` ancilla register.
            - Thus, a total of `M * B` controlled rotations are applied on the target register.

    Note that naively applying multiplexed controlled rotations on target register using a unary
    iteration loop would require us to apply `O(iteration_length)` controlled rotations for a
    single multiplexed rotations array. On the contrary, this approach requires us to apply only
    `B` controlled rotations on the target register; which is usually much smaller than the
    iteration length.

    Users should derive from this base class and override the `interleaved_unitary` and
    `interleaved_unitary_target` abstract methods to specify the information regarding
    the unitaries that should be interleaved between `M` rotations.

    For more details, see the reference below:

    References:
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
        Burg, Low et. al. 2021. Page 45; Section VII.B.1
    """

    def __init__(self, *angles: Sequence[int], kappa: int, rotation_gate: cirq.Gate):
        """Initializes ProgrammableRotationGateArrayBase

        Args:
            angles: Sequence of integer-approximated rotation angles s.t.
                `rotation_gate ** float_from_integer_approximation(angles[i][k])` should be applied
                to the target register when the selection register of ith multiplexed rotation array
                stores integer `k`.
            kappa: Length of temporary data register to use for storing integer approximated bits
                of multiplexed rotation angles.
            rotation_gate: Exponential of this gate, depending on `angles`, shall be applied on the
                target register.

        Raises:
            ValueError: If all multiplexed `angles` sequences are not of the same length.
        """
        if len(set(len(thetas) for thetas in angles)) != 1:
            raise ValueError("All multiplexed angles sequences to apply must be of same length.")
        self._angles = tuple(tuple(thetas) for thetas in angles)
        self._selection_bitsize = (len(self._angles[0]) - 1).bit_length()
        self._target_bitsize = cirq.num_qubits(rotation_gate)
        self._kappa = kappa
        self._rotation_gate = rotation_gate

    @property
    def kappa(self) -> int:
        return self._kappa

    @property
    def angles(self) -> Tuple[Tuple[int, ...], ...]:
        return self._angles

    @cached_method
    def rotation_gate(self, exponent: int = -1) -> cirq.Gate:
        """Returns `self._rotation_gate` ** 1 / (2 ** (1 + power))`"""
        power = 1 / 2 ** (1 + exponent)
        return cirq.pow(self._rotation_gate, power)

    @abc.abstractmethod
    def interleaved_unitary(
        self, index: int, **qubit_regs: NDArray[cirq.Qid]  # type:ignore[type-var]
    ) -> cirq.Operation:
        pass

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', BQUInt(self._selection_bitsize, len(self.angles[0]))),)

    @cached_property
    def kappa_load_target(self) -> Tuple[Register, ...]:
        return (Register('kappa_load_target', QAny(self.kappa)),)

    @cached_property
    def rotations_target(self) -> Tuple[Register, ...]:
        return (Register('rotations_target', QAny(self._target_bitsize)),)

    @property
    @abc.abstractmethod
    def interleaved_unitary_target(self) -> Tuple[Register, ...]:
        pass

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.selection_registers,
                *self.kappa_load_target,
                *self.rotations_target,
                *self.interleaved_unitary_target,
            ]
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        selection, kappa_load_target = quregs.pop('selection'), quregs.pop('kappa_load_target')
        rotations_target = quregs.pop('rotations_target')
        interleaved_unitary_target = quregs

        # 1. Find a convenient way to process batches of size kappa.
        num_bits = sum(max(thetas).bit_length() for thetas in self.angles)
        iteration_length = int(self.selection_registers[0].dtype.iteration_length_or_zero())
        selection_bitsizes = [s.total_bits() for s in self.selection_registers]
        angles_bits = np.zeros(shape=(iteration_length, num_bits), dtype=np.intc)
        angles_bit_pow = np.zeros(shape=(num_bits,), dtype=np.intc)
        angles_idx = np.zeros(shape=(num_bits,), dtype=np.intc)
        st, en = 0, 0
        for i, thetas in enumerate(self.angles):
            bit_width = max(thetas).bit_length()
            st, en = en, en + bit_width
            angles_bits[:, st:en] = [QUInt(bit_width).to_bits(t) for t in thetas]
            angles_bit_pow[st:en] = [*range(bit_width)][::-1]
            angles_idx[st:en] = i
        assert en == num_bits
        # 2. Process batches of size kappa.
        power_of_2s = 2 ** np.arange(self.kappa)[::-1]
        last_id = 0
        data = np.zeros(iteration_length, dtype=int)
        for st in range(0, num_bits, self.kappa):
            en = min(st + self.kappa, num_bits)
            data ^= angles_bits[:, st:en].dot(power_of_2s[: en - st])
            yield QROM(
                [data], selection_bitsizes=tuple(selection_bitsizes), target_bitsizes=(self.kappa,)
            ).on_registers(selection=selection, target0_=kappa_load_target)
            data = angles_bits[:, st:en].dot(power_of_2s[: en - st])
            for cqid, bpow, idx in zip(kappa_load_target, angles_bit_pow[st:en], angles_idx[st:en]):
                if idx != last_id:
                    yield self.interleaved_unitary(
                        last_id, rotations_target=rotations_target, **interleaved_unitary_target
                    )
                    last_id = idx
                yield self.rotation_gate(bpow).on(*rotations_target).controlled_by(cqid)
        yield QROM(
            [data], selection_bitsizes=tuple(selection_bitsizes), target_bitsizes=(self.kappa,)
        ).on_registers(selection=selection, target0_=kappa_load_target)


class ProgrammableRotationGateArray(ProgrammableRotationGateArrayBase):
    """An implementation of `ProgrammableRotationGateArrayBase` base class


    This implementation of the `ProgrammableRotationGateArray` base class expects
    all interleaved_unitaries to act on the `rotations_target` register.

    See docstring of `ProgrammableRotationGateArrayBase` for more details.
    """

    def __init__(
        self,
        *angles: Sequence[int],
        kappa: int,
        rotation_gate: cirq.Gate,
        interleaved_unitaries: Sequence[cirq.Gate] = (),
    ):
        super().__init__(*angles, kappa=kappa, rotation_gate=rotation_gate)
        if not interleaved_unitaries:
            identity_gate = cirq.IdentityGate(total_bits(self.rotations_target))
            interleaved_unitaries = (identity_gate,) * (len(angles) - 1)
        assert len(interleaved_unitaries) == len(angles) - 1
        assert all(cirq.num_qubits(u) == self._target_bitsize for u in interleaved_unitaries)
        self._interleaved_unitaries = tuple(interleaved_unitaries)

    def interleaved_unitary(self, index: int, **qubit_regs: NDArray[cirq.Qid]) -> cirq.Operation:
        return self._interleaved_unitaries[index].on(*qubit_regs['rotations_target'])

    @cached_property
    def interleaved_unitary_target(self) -> Tuple[Register, ...]:
        return ()


@bloq_example
def _programmable_rotation_gate_array() -> ProgrammableRotationGateArray:
    programmable_rotation_gate_array = ProgrammableRotationGateArray(
        [1, 2, 3, 4], kappa=2, rotation_gate=cirq.Z
    )
    return programmable_rotation_gate_array


_PROGRAMMABLE_ROTATAION_GATE_ARRAY_DOC = BloqDocSpec(
    bloq_cls=ProgrammableRotationGateArray, examples=(_programmable_rotation_gate_array,)
)
