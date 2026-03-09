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

import itertools

import numpy as np
import pytest

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.lattice._grid_operators as go
import qualtran.rotation_synthesis.lattice._test_utils as tu
import qualtran.rotation_synthesis.rings as rings

_SQRT_2 = np.sqrt(2)
_LAMBDA = 1 + _SQRT_2


class TestGridOperator:
    @pytest.mark.parametrize(
        ["g", "u"],
        [
            [go.RSqrt2, np.array([[1, -1], [1, 1]]) / _SQRT_2],
            [go.KSqrt2, np.array([[-1 / _LAMBDA, -1], [_LAMBDA, 1]]) / _SQRT_2],
            [go.ASqrt2, np.array([[1, -2], [0, 1]])],
            [go.BSqrt2, np.array([[1, _SQRT_2], [0, 1]])],
            [go.XSqrt2, np.array([[0, 1], [1, 0]])],
            [go.ZSqrt2, np.array([[1, 0], [0, -1]])],
            [go.KConjSqrt2, np.array([[_LAMBDA, -1], [-1 / _LAMBDA, 1]]) / _SQRT_2],
        ],
    )
    def test_main_operators(self, g: go.GridOperator, u):
        np.testing.assert_allclose(g.value(_SQRT_2) / _SQRT_2, u)

    @pytest.mark.parametrize("g", tu.ALL_CORE_GRID_OPS + [*tu.random_grid_operator(100, 5)])
    def test_scaled_inverse(self, g: go.GridOperator):
        np.testing.assert_allclose((g.scaled_inverse() @ g).value(_SQRT_2) / _SQRT_2, np.eye(2))

    @pytest.mark.parametrize(
        ["a", "b", "c"],
        itertools.product(
            [go.RSqrt2, go.KSqrt2, go.ASqrt2, go.BSqrt2, go.XSqrt2, go.ZSqrt2, go.KConjSqrt2],
            repeat=3,
        ),
    )
    def test_matmul(self, a: go.GridOperator, b: go.GridOperator, c: go.GridOperator):
        np.testing.assert_allclose(
            (a @ b @ c).value(_SQRT_2),
            a.value(_SQRT_2) @ b.value(_SQRT_2) @ c.value(_SQRT_2) / 2,
            atol=1e-6,
        )

    @pytest.mark.parametrize("g", tu.ALL_CORE_GRID_OPS)
    @pytest.mark.parametrize("n", range(10))
    def test_pow(self, g: go.GridOperator, n: int):
        got = (g**n).value(_SQRT_2) / _SQRT_2
        want = np.linalg.matrix_power(g.value(_SQRT_2) / _SQRT_2, n)
        np.testing.assert_allclose(got, want, atol=1e-6)

    @pytest.mark.parametrize("g", tu.ALL_CORE_GRID_OPS)
    def test_sqrt2_conj(self, g: go.GridOperator):
        res = g.sqrt2_conj()
        for i in range(2):
            for j in range(2):
                assert res.matrix[i, j] == g.matrix[i, j].conj()

    @pytest.mark.parametrize("g", tu.ALL_CORE_GRID_OPS)
    @pytest.mark.parametrize("n", range(-10, 10 + 1))
    def test_shift_roundtrip(self, g: go.GridOperator, n: int):
        r = g.shift(n).shift(-n)
        assert np.all(g.matrix == r.matrix)

    @pytest.mark.parametrize("g", tu.random_grid_operator(10, 4))
    @pytest.mark.parametrize(
        "x",
        [
            rings.ZW(p)
            for p in np.array([*itertools.product(range(-3, 3), repeat=4)])[
                np.random.choice(6**4, 10)
            ]
        ],
    )
    def test_apply_roundtrip(self, g: go.GridOperator, x: rings.ZW):
        assert x == g.scaled_inverse().apply(g.apply(x))

    @pytest.mark.parametrize(
        "x",
        [
            rings.ZW(p)
            for p in np.array([*itertools.product(range(-3, 3), repeat=4)])[
                np.random.choice(6**4, 10)
            ]
        ],
    )
    @pytest.mark.parametrize("g", tu.random_grid_operator(10, 6, 0))
    def test_apply(self, g: go.GridOperator, x: rings.ZW):
        got = g.apply(x).value(mc.NumpyConfig.sqrt2)
        matrix = g.matrix.astype(float) / mc.NumpyConfig.sqrt2
        v = x.value(mc.NumpyConfig.sqrt2)
        v = matrix @ [v.real, v.imag]
        want = v[0] + 1j * v[1]
        np.testing.assert_allclose(got, want)

    @pytest.mark.parametrize(
        "x",
        [
            rings.ZW(p)
            for p in np.array([*itertools.product(range(-3, 3), repeat=4)])[
                np.random.choice(6**4, 10)
            ]
        ],
    )
    @pytest.mark.parametrize("g1", tu.random_grid_operator(10, 6, 0))
    @pytest.mark.parametrize("g2", tu.random_grid_operator(10, 6, 64634))
    def test_composite_apply(self, g1: go.GridOperator, g2: go.GridOperator, x: rings.ZW):
        assert g1.apply(g2.apply(x)) == (g1 @ g2).apply(x)

    @pytest.mark.parametrize("k", range(-4, 4))
    @pytest.mark.parametrize("g", tu.random_grid_operator(10, 6, 0))
    def test_shift_against_matrix(self, g: go.GridOperator, k: int):
        ng = g.shift(k)
        m = g.matrix.astype(float)
        l_value = 1 + np.sqrt(2)
        desired = 1 / l_value**k * np.diagflat([l_value**k, 1]) @ m @ np.diagflat([l_value**k, 1])
        np.testing.assert_allclose(ng.matrix.astype(float), desired)
