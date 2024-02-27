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
from functools import lru_cache
from typing import Optional

import cirq
import numpy as np
from attrs import frozen

from qualtran.surface_code.magic_count import MagicCount
from qualtran.surface_code.magic_state_factory import MagicStateFactory
from qualtran.surface_code.quantum_error_correction_scheme_summary import (
    FowlerSuperconductingQubits,
    QuantumErrorCorrectionSchemeSummary,
)
from qualtran.surface_code.t_factory_utils import NoisyPauliRotation, storage_error


@frozen
class FifteenToOne(MagicStateFactory):
    """15-to-1 Magic T state factory.

    reference:
        [Magic State Distillation: Not as Costly as You Think] https://arxiv.org/abs/1905.06903

    Attributes:
        d_X: Side length of the surface code along which X measurements happen.
        d_Z: Side length of the surface code along which Z measurements happen.
        d_m: Number of code cycles used in lattice surgery.
        qec: Quantum error correction scheme being used.
    """

    d_X: int
    d_Z: int
    d_m: int
    qec: QuantumErrorCorrectionSchemeSummary = FowlerSuperconductingQubits
    reference: Optional[str] = None

    def __attrs_post_init__(self):
        assert 0 < self.d_X <= 3 * self.d_m
        assert self.d_m > 0
        assert self.d_Z > 0

    def footprint(self) -> int:
        # source: page 11 of https://arxiv.org/abs/1905.06903
        return 2 * (self.d_X + 4 * self.d_Z) * 3 * self.d_X + 4 * self.d_m

    @lru_cache(8)
    def _final_state(self, phys_err: float):
        factory = _build_factory(phys_err, self.d_X, self.d_Z, self.d_m, self.qec)
        return (
            cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(factory).final_density_matrix
        )

    @lru_cache(8)
    def p_fail(self, phys_err: float) -> float:
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        return np.real_if_close(1 - np.trace(projector @ self._final_state(phys_err))).item()

    @lru_cache(8)
    def p_out(self, phys_err: float) -> float:
        # I \otimes ones \otimes ones \otimes ones \otimes ones / 16
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        project_state = (
            1
            / (1 - self.p_fail(phys_err))
            * (projector @ self._final_state(phys_err) @ projector.T.conj())
        )
        # |T><T| \otimes ones \otimes ones \otimes ones \otimes ones / 16
        T_state = np.array([1, np.exp(-1j * np.pi / 4)]).reshape((1, 2)) / np.sqrt(2)
        target_density = np.kron(T_state.T.conj() @ T_state, np.ones((16, 16)) / 16)
        return np.real_if_close(1 - np.trace(project_state @ target_density)).item()

    def n_cycles(self, n_magic: MagicCount, phys_err: float) -> int:
        """The number of cycles (time) required to produce the requested number of magic states.

        Unlike the same method for other factories. This method reports the *expected* number of cycles
        until producing the needed magic states while taking into account possible failures.

        reference: page 11 of https://arxiv.org/abs/1905.06903
        """
        num_t = n_magic.n_t + 4 * n_magic.n_ccz
        return np.ceil(num_t * 6 * self.d_m / (1 - self.p_fail(phys_err)))

    def distillation_error(self, n_magic: MagicCount, phys_err: float) -> float:
        """The total error expected from distilling magic states with a given physical error rate."""
        num_t = n_magic.n_t + 4 * n_magic.n_ccz
        return self.p_out(phys_err) * num_t


def _build_factory(
    phys_err: float, d_X: int, d_Z: int, d_m: int, qec: QuantumErrorCorrectionSchemeSummary
) -> cirq.Circuit:
    """Builds the 15-to-1 factory with its associated cost model.

    The cost model turns the unitaries into channels (NoisyPauliRotation) and
    adds X and Z errors (storage_errors).
    The probabilities of those channels are the same as those in the supplementary
    material of https://arxiv.org/abs/1905.06903.

    Args:
        phys_err: physical error rate.
        d_X: Side length of the surface code along which X measurements happen.
        d_Z: Side length of the surface code along which Z measurements happen.
        d_m: Number of code cycles used in lattice surgery.
        qec: Quantum error correction scheme being used.


    Returns:
        The factory as a cirq circuit.
    """
    qs = cirq.LineQubit.range(5)
    px = qec.logical_error_rate(d_X, phys_err)
    pz = qec.logical_error_rate(d_Z, phys_err)
    pm = qec.logical_error_rate(d_m, phys_err)

    factory = cirq.Circuit.from_moments(
        cirq.H.on_each(qs),
        # 1
        NoisyPauliRotation(
            'IZIII',
            phys_err / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            phys_err / 3 + 0.5 * d_Z * pm,
            phys_err / 3,
        )(*qs),
        # 2
        NoisyPauliRotation(
            'IIZII',
            phys_err / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            phys_err / 3 + 0.5 * d_Z * pm,
            phys_err / 3,
        )(*qs),
        # 3
        NoisyPauliRotation(
            'IIIZI',
            phys_err / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            phys_err / 3 + 0.5 * d_Z * pm,
            phys_err / 3,
        )(*qs),
        # 5
        NoisyPauliRotation(
            'IZZZI',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (3 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        *storage_error(
            'X',
            [
                0,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0,
            ],
            qs,
        ),
        # 6
        NoisyPauliRotation(
            'ZZZII',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 2 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 7
        NoisyPauliRotation(
            'ZZIZI',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 3 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 2 * d_Z) + (d_X + 3 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0,
            ],
            qs,
        ),
        # 8
        NoisyPauliRotation(
            'ZIZZI',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 3 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 9
        NoisyPauliRotation(
            'ZIIZZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 4
        NoisyPauliRotation(
            'IIIIZ',
            phys_err / 3 + 0.5 * (d_m / d_Z) * pz * d_m,
            phys_err / 3 + 0.5 * d_Z * pm,
            phys_err / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 3 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 10
        NoisyPauliRotation(
            'ZZIIZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 11
        NoisyPauliRotation(
            'ZIZIZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        *storage_error(
            'Z', [0.5 * ((d_X + 4 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0], qs
        ),
        *storage_error(
            'X',
            [
                0.5 * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 12
        NoisyPauliRotation(
            'ZZZZZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (d_X + 4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 13
        NoisyPauliRotation(
            'IIZZZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (3 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        *storage_error('Z', [0.5 * (d_X + 4 * d_Z) / d_X * px * d_m, 0, 0, 0, 0], qs),
        *storage_error(
            'X',
            [
                0.5 * px * (d_m + 2 * d_X),
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0.5 * px * (d_m + 2 * d_X),
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
        # 14
        NoisyPauliRotation(
            'IZIZZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        # 15
        NoisyPauliRotation(
            'IZZIZ',
            phys_err / 3 + 0.5 * pm * d_m,
            phys_err / 3 + 0.5 * pm * d_m + 0.5 * (4 * d_Z) * d_X / d_m * pm,
            phys_err / 3,
        )(*qs),
        *storage_error(
            'X',
            [
                0,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
                0.5 * (d_Z / d_X) * px * d_m,
            ],
            qs,
        ),
        *storage_error(
            'Z',
            [
                0,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
                0.5 * (d_X / d_Z) * pz * d_m,
            ],
            qs,
        ),
    )
    return factory


FifteenToOne733 = FifteenToOne(7, 3, 3, reference='https://arxiv.org/abs/1905.06903')

FifteenToOne933 = FifteenToOne(9, 3, 3, reference='https://arxiv.org/abs/1905.06903')
