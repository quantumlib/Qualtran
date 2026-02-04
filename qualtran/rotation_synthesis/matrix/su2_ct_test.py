#  Copyright 2025 Google LLC
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

from typing import Optional

import numpy as np
import pytest

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.matrix as rsm
from qualtran.rotation_synthesis.matrix import _su2_ct
from qualtran.rotation_synthesis.rings import _zsqrt2, _zw

_GATES = [_su2_ct.ISqrt2, _su2_ct.SSqrt2, _su2_ct.HSqrt2, _su2_ct.Tx, _su2_ct.Ty, _su2_ct.Tz]
_SQRT2 = np.sqrt(2)
_LAMBDA = _zsqrt2.ZSqrt2(2, 1)
_LAMBDA_ZW = _zw.ZW.from_pair(_LAMBDA, _zsqrt2.Zero)


def _make_random_su(n: int, m: int, random_cliffords: bool = False, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    gates = [_su2_ct.Tx, _su2_ct.Ty, _su2_ct.Tz]
    if random_cliffords:
        gates += [_su2_ct.SSqrt2, _su2_ct.HSqrt2]
    for _ in range(n):
        res = _su2_ct.ISqrt2
        for i in rng.choice(len(gates), m):
            res = res @ gates[i]
        yield res


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_parametric_form(g: _su2_ct.SU2CliffordT):
    pf = g.parametric_form()
    got = _su2_ct.SU2CliffordT.from_parametric_form(pf)
    assert got == g


@pytest.mark.parametrize("seq", np.random.choice(6, size=(10, 5)))
def test_multiply(seq):
    g = _su2_ct.ISqrt2
    g_numpy = np.eye(2)
    k = 0
    for i in seq:
        g = g @ _GATES[i]
        k += i >= 3  # is a T gate.
        g_numpy = (g_numpy @ _GATES[i].matrix.astype(complex)) / _SQRT2
    assert g.det() == 2 * _LAMBDA**k
    np.testing.assert_allclose(g.matrix.astype(complex) / _SQRT2, g_numpy, atol=1e-9)


@pytest.mark.parametrize("g", _make_random_su(10, 3, random_cliffords=False, seed=0))
def test_adjoint(g):
    assert g @ g.adjoint() == _su2_ct.ISqrt2 * _LAMBDA_ZW**3


_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]])
_TX_numpy = (np.eye(2) + (np.eye(2) - 1j * _X) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TY_numpy = (np.eye(2) + (np.eye(2) - 1j * _Y) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TZ_numpy = (np.eye(2) + (np.eye(2) - 1j * _Z) / _SQRT2) / np.sqrt(2 + _SQRT2)


@pytest.mark.parametrize(
    ["g", "g_numpy"], [[_su2_ct.Tx, _TX_numpy], [_su2_ct.Ty, _TY_numpy], [_su2_ct.Tz, _TZ_numpy]]
)
def test_t_gates(g, g_numpy):
    np.testing.assert_allclose(g.matrix.astype(complex) / _SQRT2 / np.sqrt(2 + _SQRT2), g_numpy)
    np.testing.assert_allclose(
        g.adjoint().matrix.astype(complex) / _SQRT2 / np.sqrt(2 + _SQRT2), g_numpy.T.conjugate()
    )


def are_close_up_to_global_phase(u, v):
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.abs(u).argmax(), u.shape
    )
    return np.allclose(u * v[i, j] / u[i, j], v)


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
@pytest.mark.parametrize("config", [None, mc.NumpyConfig])
def test_rescale(g: _su2_ct.SU2CliffordT, config):
    np.testing.assert_allclose(g.numpy(config), g.rescale().numpy(config))


def test_num_t_gates():
    for clifford in rsm.generate_cliffords():
        assert clifford.num_t_gates() == 0

        for t in _su2_ct.Ts:
            assert (clifford @ t).num_t_gates() == 1

    for t1 in _su2_ct.Ts:
        for t2 in _su2_ct.Ts:
            assert (t1 @ t2).num_t_gates() == 2

    for t in _su2_ct.Ts:
        # still two T gates
        assert (t @ t.adjoint()).num_t_gates() == 2

        # We need to call .rescale to remove the common factor and reduce the T count.
        assert (t @ t.adjoint()).rescale().num_t_gates() == 0


_H_bloch_form_numpy = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
_S_bloch_form_numpy = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
_T_bloch_form_numpy = np.array([[1, -1, 0], [1, 1, 0], [0, 0, _SQRT2]]) / _SQRT2


@pytest.mark.parametrize(
    ["g", "bloch_form", "n"],
    [
        [_su2_ct.HSqrt2, _H_bloch_form_numpy, 0],
        [_su2_ct.SSqrt2, _S_bloch_form_numpy, 0],
        [_su2_ct.TSqrt2, _T_bloch_form_numpy, 1],
    ],
)
def test_bloch_sphere_form_generators(g: _su2_ct.SU2CliffordT, bloch_form: np.ndarray, n: int):
    g_so3, m = g.bloch_sphere_form()
    assert m == n
    np.testing.assert_allclose(g_so3.astype(float) / (2 * _SQRT2 * (2 + _SQRT2) ** m), bloch_form)


def _matrix_from_pauli(coords: np.ndarray) -> np.ndarray:
    return coords[0] * _X + coords[1] * _Y + coords[2] * _Z


@pytest.mark.parametrize("g", _make_random_su(10, 5, random_cliffords=True, seed=0))
@pytest.mark.parametrize("vector", np.random.choice(2, size=(10, 3)))
def test_bloch_sphere_form_random(g: _su2_ct.SU2CliffordT, vector: np.ndarray):
    g_so3, _ = g.bloch_sphere_form()
    g_action = (
        g.matrix.astype(complex) @ _matrix_from_pauli(vector) @ g.adjoint().matrix.astype(complex)
    )
    g_so3_action = g_so3.astype(float) @ vector.T / _SQRT2
    np.testing.assert_allclose(_matrix_from_pauli(g_so3_action), g_action, atol=1e-7)


_T_parity_numpy = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
_HT_parity_numpy = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
_SHT_parity_numpy = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0]])


@pytest.mark.parametrize(
    ["g", "parity"],
    [
        [_su2_ct.TSqrt2, _T_parity_numpy],
        [_su2_ct.HSqrt2 @ _su2_ct.TSqrt2, _HT_parity_numpy],
        [_su2_ct.SSqrt2 @ _su2_ct.HSqrt2 @ _su2_ct.TSqrt2, _SHT_parity_numpy],
    ],
)
def test_bloch_form_parity(g: _su2_ct.SU2CliffordT, parity: np.ndarray):
    g_parity = g.bloch_form_parity()
    np.testing.assert_equal(g_parity, parity)
