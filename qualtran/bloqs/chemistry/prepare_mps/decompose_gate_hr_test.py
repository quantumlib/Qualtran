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

from typing import Tuple

import numpy as np
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.chemistry.prepare_mps.decompose_gate_hr import DecomposeGateViaHR
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.testing import assert_valid_bloq_decomposition


# these gates can be approximated exactly with the given phase_bitsize
@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [2, ((0, ((0.5 + 0.5j), (0.5 - 0.5j))), (1, ((-0.5 - 0.5j), (0.5 - 0.5j))))],
        [
            4,
            (
                (
                    0,
                    (
                        (-0.191341716182545 + 0.961939766255644j),
                        (-0.038060233744357 + 0.191341716182545j),
                    ),
                ),
                (
                    1,
                    (
                        (0.038060233744356 - 0.191341716182545j),
                        (-0.191341716182545 + 0.961939766255644j),
                    ),
                ),
            ),
        ],
        [
            2,
            (
                (0, ((-0 - 0.5j), (0.5 - 0j), (-0.5 + 0.5j), -0j)),
                (1, ((0.5 + 0.5j), (0.5 - 0.5j), 0j, 0j)),
                (2, (0.5j, (-0.5 + 0j), (-0.5 + 0.5j), 0j)),
                (3, (0j, 0j, (-0 + 0j), (-1 + 0j))),
            ),
        ],
    ],
)
def test_exact_gate_compilation(
    phase_bitsize: int, gate_cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
):
    gate_compiler = DecomposeGateViaHR(phase_bitsize, tuple(gate_cols), internal_phase_grad=True)
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract()
    assert np.allclose(compiled_gate, np.array([gc[1] for gc in gate_cols]).T)


@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [
            4,
            (
                (
                    0,
                    (
                        (-0.191341716182545 + 0.961939766255644j),
                        (-0.038060233744357 + 0.191341716182545j),
                    ),
                ),
            ),
        ],
        [
            2,
            (
                (0, ((-0 - 0.5j), (0.5 - 0j), (-0.5 + 0.5j), -0j)),
                (1, ((0.5 + 0.5j), (0.5 - 0.5j), 0j, 0j)),
                (2, (0.5j, (-0.5 + 0j), (-0.5 + 0.5j), 0j)),
            ),
        ],
    ],
)
def test_partial_gate_compilation(
    phase_bitsize: int, gate_cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
):
    gate_compiler = DecomposeGateViaHR(phase_bitsize, tuple(gate_cols), internal_phase_grad=True)
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract().T
    assert np.allclose(
        compiled_gate[range(len(gate_cols)), :], np.array([gc[1] for gc in gate_cols])
    )


@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [2, ((0, ((0.5 + 0.5j), (0.5 - 0.5j))), (1, ((-0.5 - 0.5j), (0.5 - 0.5j))))],
        [
            2,
            (
                (0, ((-0 - 0.5j), (0.5 - 0j), (-0.5 + 0.5j), -0j)),
                (1, ((0.5 + 0.5j), (0.5 - 0.5j), 0j, 0j)),
                (2, (0.5j, (-0.5 + 0j), (-0.5 + 0.5j), 0j)),
                (3, (0j, 0j, (-0 + 0j), (-1 + 0j))),
            ),
        ],
    ],
)
def test_gate_compilation_adjoint(
    phase_bitsize: int, gate_cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
):
    gate_compiler = DecomposeGateViaHR(
        gate_cols=gate_cols,
        phase_bitsize=phase_bitsize,
        uncompute=False,
        internal_refl_ancilla=False,
    )
    gate_compiler_adj = DecomposeGateViaHR(
        gate_cols=gate_cols,
        phase_bitsize=phase_bitsize,
        uncompute=True,
        internal_refl_ancilla=False,
    )
    bb = BloqBuilder()
    inp = bb.add_register("gate_input", gate_compiler.gate_bitsize)
    pg = bb.add(PhaseGradientState(bitsize=phase_bitsize))
    ra = bb.allocate(1)
    ra, inp, pg = bb.add(gate_compiler, gate_input=inp, phase_grad=pg, refl_ancilla=ra)
    ra, inp, pg = bb.add(gate_compiler_adj, gate_input=inp, phase_grad=pg, refl_ancilla=ra)
    bb.free(ra)
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=pg)
    gate_coefs = bb.finalize(gate_input=inp).tensor_contract()
    assert np.allclose(gate_coefs, np.eye(2**gate_compiler.gate_bitsize))
