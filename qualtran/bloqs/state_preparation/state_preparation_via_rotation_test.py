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

import cirq
import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import CNOT, PlusState, ZeroState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
    _state_prep_via_rotation,
    _state_prep_via_rotation_symb,
    _state_prep_via_rotation_symb_phasegrad,
    PRGAViaPhaseGradient,
    StatePreparationViaRotations,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def accuracy(state1, state2):
    return abs(np.dot(state1, state2.conj()))


@pytest.mark.parametrize(
    "bloq_ex",
    [
        _state_prep_via_rotation,
        _state_prep_via_rotation_symb,
        _state_prep_via_rotation_symb_phasegrad,
    ],
    ids=lambda bloq_ex: bloq_ex.name,
)
def test_state_prep_via_rotation(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


def test_state_prep_via_rotation_symb_quick():
    bloq = _state_prep_via_rotation_symb.make()
    L, phase = bloq.n_coeff, bloq.phase_bitsize
    expected_t_count_expr = 16 * L + 8 * phase - 32
    assert isinstance(expected_t_count_expr, sympy.Expr)
    assert bloq.t_complexity().t == expected_t_count_expr


@pytest.mark.slow
def test_state_prep_via_rotation_symb():
    bloq = _state_prep_via_rotation_symb.make()
    L, phase = bloq.n_coeff, bloq.phase_bitsize
    expected_t_count_expr = 16 * L + 8 * phase - 32
    assert isinstance(expected_t_count_expr, sympy.Expr)
    assert bloq.t_complexity().t == expected_t_count_expr

    # Compare bloq counts via expression to actual bloq counts and make sure they
    # are "close enough"

    # The discrepency comes from the fact that in the concrete case, prga_prepare_amplitude
    # calls PRGA for selection bitsizes `0 + 1 + 2 + 3 + ... + n - 1`  (where n is size of selection
    # register) and corresponding rom_values of length `2**0 + 2**1 + 2**2 + ...  + 2**(n - 1)`
    # When n is symbolic, we can't simulate this in the build_call_graph so we instead return a
    # single call to PRGA with bitsize `n` and rom_values of size `2**n`. The constant factor of
    # the dominant cost of QROM scales as 4 * L and sum of `2**0 + 2**1 + 2**2 + ...  + 2**(n-1)`
    # is `2**n` so dominant cost matches. But the smaller costs like addition into phase gradient
    # register scale with selection bitsize; and so `0 + 1 + 2 + 3 + ... + n - 1` > `n` -- this
    # is where the discrepancy comes from.
    # Note that if we replace `QROM` with `SelectSwapQROM` the discrepency would increase because
    # sum of `2**(i // 2)` would not be equal to `2**(n // 2)` and thus our current symbolic
    # strategy would not work.
    N, phase_bitsize = 2**16, 10
    state_coefs = cirq.testing.random_superposition(N, random_state=1234)
    bloq_concrete = StatePreparationViaRotations(
        state_coefficients=state_coefs, phase_bitsize=phase_bitsize
    )
    concrete_t_counts = bloq_concrete.t_complexity().t
    # Symbolic T-counts
    symb_t_counts = int(expected_t_count_expr.subs({L: N, phase: phase_bitsize}))

    # assert they are "close enough"
    np.testing.assert_allclose(symb_t_counts, concrete_t_counts, rtol=1e-3)


# these states can be prepared exactly with the given phase_bitsize
@pytest.mark.parametrize(
    "phase_bitsize, state_coefs",
    [
        [2, ((-0.5 - 0.5j), (0.5 - 0.5j))],
        [
            4,
            (
                (-0.8154931568489165 - 0.16221167441072862j),
                (-0.46193976625564304 - 0.30865828381745486j),
            ),
        ],
        [
            3,
            (
                (-0.42677669529663675 - 0.1767766952966366j),
                (0.17677669529663664 - 0.4267766952966367j),
                (0.17677669529663675 - 0.1767766952966368j),
                (0.07322330470336305 - 0.07322330470336309j),
                (0.4267766952966366 - 0.17677669529663692j),
                (0.42677669529663664 + 0.17677669529663675j),
                (0.0732233047033631 + 0.17677669529663678j),
                (-0.07322330470336308 - 0.17677669529663678j),
            ),
        ],
    ],
)
def test_exact_state_prep_via_rotation_(phase_bitsize: int, state_coefs: Tuple[complex, ...]):
    # https://github.com/python/mypy/issues/5313
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs  # type: ignore[arg-type]
    )
    assert_valid_bloq_decomposition(qsp)
    bb = BloqBuilder()
    state = bb.allocate((len(state_coefs) - 1).bit_length())
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    state, phase_gradient = bb.add(qsp, target_state=state, phase_gradient=phase_gradient)
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(state=state)
    result = network.tensor_contract()
    assert np.isclose(accuracy(result, np.array(state_coefs)), 1)


@pytest.mark.parametrize(
    "phase_bitsize, state_coefs",
    [
        [
            4,
            (
                (-0.8154931568489165 - 0.16221167441072862j),
                (-0.46193976625564304 - 0.30865828381745486j),
            ),
        ],
        [
            3,
            (
                (0.3535533905932734 - 0.35355339059327373j),
                (0.3535533905932735 - 0.3535533905932736j),
                (-0.4999999999999997 + 1.1102230246251563e-16j),
                (0.35355339059327356 + 0.35355339059327356j),
            ),
        ],
    ],
)
def test_state_prep_via_rotation_adjoint(
    phase_bitsize: int, state_coefs: Tuple[complex, ...]
) -> None:
    # https://github.com/python/mypy/issues/5313
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs  # type: ignore[arg-type]
    )
    qsp_adj = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs, uncompute=True  # type: ignore[arg-type]
    )

    bb = BloqBuilder()
    state = bb.allocate((len(state_coefs) - 1).bit_length())
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    state, phase_gradient = bb.add(qsp, target_state=state, phase_gradient=phase_gradient)
    state, phase_gradient = bb.add(qsp_adj, target_state=state, phase_gradient=phase_gradient)
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(state=state)
    result = network.tensor_contract()
    assert np.isclose(result[0], 1)  # test that |result> = |0>


# these states can't be approximated exactly with the given
# phase_bitsize, check they are close enough
@pytest.mark.parametrize(
    "phase_bitsize, state_coefs",
    [
        [
            3,
            (
                (0.481145088606368 - 0.47950088720913586j),
                (-0.41617865941997106 - 0.604461434931144j),
            ),
        ],
        [
            3,
            (
                (-0.45832126811131957 - 0.18416419776558368j),
                (0.17057042949351137 + 0.24581249575498615j),
                (-0.6048757470423172 - 0.5250403822076705j),
                (-0.13534197956156918 - 0.08153272490768977j),
            ),
        ],
    ],
)
def test_approximate_state_prep_via_rotation(phase_bitsize: int, state_coefs: Tuple[complex, ...]):
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs  # type: ignore[arg-type]
    )
    assert_valid_bloq_decomposition(qsp)
    bb = BloqBuilder()
    state = bb.allocate((len(state_coefs) - 1).bit_length())
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    state, phase_gradient = bb.add(qsp, target_state=state, phase_gradient=phase_gradient)
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(state=state)
    result = network.tensor_contract()
    assert accuracy(result, np.array(state_coefs)) >= 0.95


@pytest.mark.parametrize(
    "phase_bitsize, state_coefs",
    [
        [
            2,
            (
                (-0.45832126811131957 - 0.18416419776558368j),
                (0.17057042949351137 + 0.24581249575498615j),
                (-0.6048757470423172 - 0.5250403822076705j),
                (-0.13534197956156918 - 0.08153272490768977j),
            ),
        ]
    ],
)
def test_controlled_state_preparation_via_rotation_do_not_prepare(
    phase_bitsize: int, state_coefs: Tuple[complex, ...]
):
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs, control_bitsize=1  # type: ignore[arg-type]
    )
    assert_valid_bloq_decomposition(qsp)
    bb = BloqBuilder()
    prepare_control = bb.allocate(1)
    state = bb.allocate((len(state_coefs) - 1).bit_length())
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    prepare_control, state, phase_gradient = bb.add(
        qsp, prepare_control=prepare_control, target_state=state, phase_gradient=phase_gradient
    )
    bb.free(prepare_control)
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(state=state)
    result = network.tensor_contract()
    assert np.allclose(
        result, np.array([1] + [0] * (2 ** (len(state_coefs) - 1).bit_length() - 1))
    )  # assert result = |0>


@pytest.mark.parametrize("phase_bitsize, state_coefs", [[2, ((-0.5 - 0.5j), 0, 0.5, -0.5)]])
def test_state_preparation_via_rotation_superposition_ctrl(
    phase_bitsize: int, state_coefs: Tuple[complex, ...]
):
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs, control_bitsize=1  # type: ignore[arg-type]
    )
    assert_valid_bloq_decomposition(qsp)
    bb = BloqBuilder()
    prepare_control = bb.add(PlusState())
    state = bb.allocate((len(state_coefs) - 1).bit_length())
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    prepare_control, state, phase_gradient = bb.add(
        qsp, prepare_control=prepare_control, target_state=state, phase_gradient=phase_gradient
    )
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(prepare_control=prepare_control, state=state)
    result = network.tensor_contract()
    correct = (
        1
        / np.sqrt(2)
        * np.array([1] + [0] * (2 ** (len(state_coefs) - 1).bit_length() - 1) + list(state_coefs))
    )
    # assert result = 1/sqrt(2)*(|0, 0> + |1, state>)
    assert np.allclose(result, correct)


@pytest.mark.parametrize("phase_bitsize, state_coefs", [[2, ((-0.5 - 0.5j), 0, 0.5, -0.5)]])
def test_state_preparation_via_rotation_multi_qubit_ctrl(
    phase_bitsize: int, state_coefs: Tuple[complex, ...]
):
    qsp = StatePreparationViaRotations(
        phase_bitsize=phase_bitsize, state_coefficients=state_coefs, control_bitsize=2  # type: ignore[arg-type]
    )
    state_bitsize = (len(state_coefs) - 1).bit_length()
    assert_valid_bloq_decomposition(qsp)
    bb = BloqBuilder()
    # set prepare control to |00> + |11>
    q0, q1 = bb.add(CNOT(), ctrl=bb.add(PlusState()), target=bb.add(ZeroState()))
    prepare_control = bb.join(np.array([q0, q1]))
    state = bb.allocate(state_bitsize)
    phase_gradient = bb.add(PhaseGradientState(phase_bitsize))
    prepare_control, state, phase_gradient = bb.add(
        qsp, prepare_control=prepare_control, target_state=state, phase_gradient=phase_gradient
    )
    bb.add(PhaseGradientState(bitsize=phase_bitsize).adjoint(), phase_grad=phase_gradient)
    network = bb.finalize(prepare_control=prepare_control, state=state)
    result = network.tensor_contract()
    zero_padding = [0] * (2 ** (state_bitsize + 2) - 2**state_bitsize - 1)
    correct = 1 / np.sqrt(2) * np.array([1] + zero_padding + list(state_coefs))
    # assert result = 1/sqrt(2)*(|00, 0> + |11, state>)
    assert np.allclose(result, correct)


@pytest.mark.notebook
def test_notebook():
    execute_notebook("state_preparation_via_rotation")


@pytest.mark.notebook
def test_notebook_tutorial():
    execute_notebook("state_preparation_via_rotation_tutorial")


@pytest.mark.parametrize("phase_bitsize, rom_vals", [[3, (6, 0, 5, 2)]])
def test_PRGAViaPhaseGradient(phase_bitsize, rom_vals):
    sel_bitsize = (len(rom_vals) - 1).bit_length()
    prga = PRGAViaPhaseGradient(sel_bitsize, phase_bitsize, rom_vals, 1)
    assert_valid_bloq_decomposition(prga)
    bb = BloqBuilder()
    control = bb.add(PlusState())
    sel = bb.join(np.array([bb.add(PlusState()) for _ in range(sel_bitsize)]))
    pg = bb.add(PhaseGradientState(phase_bitsize))
    control, sel, pg = bb.add(prga, control=control, selection=sel, phase_gradient=pg)
    bb.add(PhaseGradientState(phase_bitsize).adjoint(), phase_grad=pg)
    circuit = bb.finalize(control=control, sel=sel)
    result = circuit.tensor_contract()
    # get the angles that correspond to each rom value loaded
    angles = [2 * np.pi * rv / 2**phase_bitsize for rv in rom_vals]
    # make a vector corresponding to the state 1/sqrt(2)(|0> + |1>)|+...+>
    # with registers (|control, selection>)
    correct_state = [np.power(2.0, -(sel_bitsize + 1) / 2)] * (2 ** (sel_bitsize + 1))
    # give the |1>|+...+> term the corresponding rotations
    for i, ang in enumerate(angles):
        correct_state[i + 2**sel_bitsize] *= np.power(np.e, 1j * ang)
    assert np.allclose(correct_state, result)
