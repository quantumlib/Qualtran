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
import cirq
import numpy as np
import sympy

from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.rotations.programmable_ancilla_rotation import (
    _zpow_programmed_ancilla,
    _zpow_programmed_ancilla_symb,
    _zpow_using_programmed_ancilla,
    _zpow_using_programmed_ancilla_symb,
    _zpow_using_programmed_ancilla_symb_rounds,
    ZPowProgrammedAncilla,
    ZPowUsingProgrammedAncilla,
)
from qualtran.cirq_interop import CirqGateAsBloq


def test_rz_resource_state_examples(bloq_autotester):
    bloq_autotester(_zpow_programmed_ancilla)
    bloq_autotester(_zpow_programmed_ancilla_symb)


def test_rz_resource_state_tensor_on_random_angles():
    rng = np.random.default_rng(42)

    for exponent in rng.random(size=5):
        theta = 2 * np.pi * exponent
        bloq = ZPowProgrammedAncilla(theta)
        np.testing.assert_allclose(
            bloq.tensor_contract(),
            np.array([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]) / np.sqrt(2),
        )


def test_rz_via_par_examples(bloq_autotester):
    bloq_autotester(_zpow_using_programmed_ancilla)
    bloq_autotester(_zpow_using_programmed_ancilla_symb)
    bloq_autotester(_zpow_using_programmed_ancilla_symb_rounds)


def test_rz_via_par_call_graphs():
    _, sigma_rz = ZPowUsingProgrammedAncilla(np.pi / 4).call_graph(max_depth=1)
    assert sigma_rz == {
        CNOT(): 2,
        XGate(): 2,
        ZPowProgrammedAncilla(np.pi / 4, eps=1e-11 / 2): 1,
        ZPowProgrammedAncilla(np.pi / 2, eps=1e-11 / 2): 1,
        CirqGateAsBloq(cirq.MeasurementGate(1)): 2,
    }

    phi, eps = sympy.symbols(r"\phi \epsilon")

    _, sigma_rz_symb = ZPowUsingProgrammedAncilla(phi, eps=eps, n_rounds=3).call_graph(max_depth=1)
    assert sigma_rz_symb == {
        CNOT(): 3,
        XGate(): 3,
        ZPowProgrammedAncilla(phi, eps / 3): 1,
        ZPowProgrammedAncilla(2 * phi, eps / 3): 1,
        ZPowProgrammedAncilla(4 * phi, eps / 3): 1,
        CirqGateAsBloq(cirq.MeasurementGate(1)): 3,
    }

    phi0, eps0, n = sympy.symbols(r"_\phi0 _\epsilon0 n")
    _, sigma_rz_symb_rounds = ZPowUsingProgrammedAncilla(phi, eps=eps, n_rounds=n).call_graph(
        max_depth=1
    )
    assert sigma_rz_symb_rounds == {
        CNOT(): n,
        XGate(): n,
        ZPowProgrammedAncilla(phi0, eps0): n,
        CirqGateAsBloq(cirq.MeasurementGate(1)): n,
    }
