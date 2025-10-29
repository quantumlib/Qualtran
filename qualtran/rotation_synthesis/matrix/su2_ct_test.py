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

import cirq
import numpy as np
import pytest

from qualtran.rotation_synthesis.matrix import su2_ct
from qualtran.rotation_synthesis.rings import zsqrt2, zw

_GATES = [su2_ct.ISqrt2, su2_ct.SSqrt2, su2_ct.HSqrt2, su2_ct.Tx, su2_ct.Ty, su2_ct.Tz]
_SQRT2 = np.sqrt(2)
_LAMBDA = zsqrt2.ZSqrt2(2, 1)
_LAMBDA_ZW = zw.ZW.from_pair(_LAMBDA, zsqrt2.Zero)


def _make_random_su(n: int, m: int, random_cliffords: bool = False, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    gates = [su2_ct.Tx, su2_ct.Ty, su2_ct.Tz]
    if random_cliffords:
        gates += [su2_ct.SSqrt2, su2_ct.HSqrt2]
    for _ in range(n):
        res = su2_ct.ISqrt2
        for i in rng.choice(len(gates), m):
            res = res @ gates[i]
        yield res


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_parametric_form(g: su2_ct.SU2CliffordT):
    pf = g.parametric_form()
    got = su2_ct.SU2CliffordT.from_parametric_form(pf)
    assert got == g


@pytest.mark.parametrize("seq", np.random.choice(6, size=(10, 5)))
def test_multiply(seq):
    g = su2_ct.ISqrt2
    g_numpy = np.eye(2)
    k = 0
    for i in seq:
        g = g @ _GATES[i]
        k += i >= 3  # is a T gate.
        g_numpy = (g_numpy @ _GATES[i].numpy()) / _SQRT2
    assert g.det() == 2 * _LAMBDA**k
    np.testing.assert_allclose(g.numpy() / _SQRT2, g_numpy, atol=1e-9)


@pytest.mark.parametrize("g", _make_random_su(10, 3, random_cliffords=False, seed=0))
def test_adjoint(g):
    assert g @ g.adjoint() == su2_ct.ISqrt2 * _LAMBDA_ZW**3


_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]])
_TX_numpy = (np.eye(2) + (np.eye(2) - 1j * _X) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TY_numpy = (np.eye(2) + (np.eye(2) - 1j * _Y) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TZ_numpy = (np.eye(2) + (np.eye(2) - 1j * _Z) / _SQRT2) / np.sqrt(2 + _SQRT2)


@pytest.mark.parametrize(
    ["g", "g_numpy"], [[su2_ct.Tx, _TX_numpy], [su2_ct.Ty, _TY_numpy], [su2_ct.Tz, _TZ_numpy]]
)
def test_t_gates(g, g_numpy):
    np.testing.assert_allclose(g.numpy() / _SQRT2 / np.sqrt(2 + _SQRT2), g_numpy)
    np.testing.assert_allclose(
        g.adjoint().numpy() / _SQRT2 / np.sqrt(2 + _SQRT2), g_numpy.T.conjugate()
    )


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_seq(g):
    seq = g.to_sequence()
    got = su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g or got * -1 == g


def are_close_up_to_global_phase(u, v):
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.abs(u).argmax(), u.shape
    )
    return np.allclose(u * v[i, j] / u[i, j], v)


def test_generate_cliffords():
    cliffords = su2_ct.generate_cliffords()
    cirq_cliffords = [
        cirq.unitary(c) for c in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords
    ]
    assert np.allclose(np.abs([np.linalg.det(c.numpy()) for c in cliffords]), 2)
    sqrt2 = np.sqrt(2)
    for c in cliffords:
        u = c.numpy() / sqrt2
        assert np.any([are_close_up_to_global_phase(u, c) for c in cirq_cliffords])
