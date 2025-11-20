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

import numpy as np
import pytest

import qualtran.rotation_synthesis.math_config as mc
from qualtran.rotation_synthesis.lattice import grid_operators as go
from qualtran.rotation_synthesis.lattice import state as lattice_state
from qualtran.rotation_synthesis.lattice import test_utils as tu

_L_VALUE = 1 + mc.NumpyConfig.sqrt2


def sinh(x):
    l_value = 1 + mc.NumpyConfig.sqrt2
    return (l_value**x - l_value ** (-x)) / 2


def cosh(x):
    l_value = 1 + mc.NumpyConfig.sqrt2
    return (l_value**x + l_value ** (-x)) / 2


class TestState:
    @pytest.mark.parametrize("s", tu.make_states(10))
    def test_skew(self, s: lattice_state.SelingerState):
        np.testing.assert_allclose(
            s.skew(mc.NumpyConfig),
            s.m1.parametric_form(mc.NumpyConfig).b ** 2
            + s.m2.parametric_form(mc.NumpyConfig).b ** 2,
        )

    @pytest.mark.parametrize("s", tu.make_states(10))
    def test_bias(self, s: lattice_state.SelingerState):
        np.testing.assert_allclose(
            s.bias(mc.NumpyConfig),
            s.m2.parametric_form(mc.NumpyConfig).z - s.m1.parametric_form(mc.NumpyConfig).z,
        )

    @pytest.mark.parametrize("s", tu.make_states(20))
    @pytest.mark.parametrize("k", range(-10, 11))
    def test_shift(self, s: lattice_state.SelingerState, k: int):
        ns = s.shift(k, mc.NumpyConfig)
        np.testing.assert_allclose(ns.skew(mc.NumpyConfig), s.skew(mc.NumpyConfig))
        np.testing.assert_allclose(ns.bias(mc.NumpyConfig), s.bias(mc.NumpyConfig) + 2 * k)

    @pytest.mark.parametrize("state", tu.make_states(1000))
    @pytest.mark.parametrize("config", [mc.NumpyConfig, mc.with_dps(300)])
    def test_get_grid_operator(self, state: lattice_state.SelingerState, config):
        action = state.get_grid_operator(config)
        if state.skew(config) < 5:
            return
        new_state = state.apply(action, config)
        ratio = new_state.skew(config) / state.skew(config)
        err_msg = f"{ratio=} | b={state.m1.parametric_form(config).b} z={state.m1.parametric_form(config).z} zeta={state.m2.parametric_form(config).z}"
        assert ratio <= 0.9, err_msg

    @pytest.mark.parametrize("state", tu.make_states(100))
    def test_apply_r(self, state: lattice_state.SelingerState):
        ns = state.apply(lattice_state.GridOperatorAction(go.RSqrt2), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(new_p1.b, orig_p1.e * sinh(orig_p1.z))

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(new_p2.b, orig_p2.e * sinh(orig_p2.z))

    @pytest.mark.parametrize("state", tu.make_states(100))
    def test_apply_k(self, state: lattice_state.SelingerState):
        ns = state.apply(lattice_state.GridOperatorAction(go.KSqrt2), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p1.b, orig_p1.e * cosh(orig_p1.z + 1) - mc.NumpyConfig.sqrt2 * orig_p1.b
        )

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p2.b, mc.NumpyConfig.sqrt2 * orig_p2.b - orig_p2.e * cosh(orig_p2.z - 1)
        )

    @pytest.mark.parametrize("state", tu.make_states(100))
    def test_apply_k_conj(self, state: lattice_state.SelingerState):
        ns = state.apply(lattice_state.GridOperatorAction(go.KConjSqrt2), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p1.b, mc.NumpyConfig.sqrt2 * orig_p1.b - orig_p1.e * cosh(orig_p1.z - 1)
        )

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p2.b, orig_p2.e * cosh(orig_p2.z + 1) - mc.NumpyConfig.sqrt2 * orig_p2.b
        )

    @pytest.mark.parametrize("state", tu.make_states(10))
    @pytest.mark.parametrize("n", range(1, 10))
    def test_apply_a(self, state: lattice_state.SelingerState, n: int):
        ns = state.apply(lattice_state.GridOperatorAction(go.ASqrt2**n), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p1.b, orig_p1.b - 2 * n * orig_p1.e * _L_VALUE ** (-orig_p1.z)
        )

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p2.b, orig_p2.b - 2 * n * orig_p2.e * _L_VALUE ** (-orig_p2.z)
        )

    @pytest.mark.parametrize("state", tu.make_states(10))
    @pytest.mark.parametrize("n", range(1, 10))
    def test_apply_b(self, state: lattice_state.SelingerState, n: int):
        ns = state.apply(lattice_state.GridOperatorAction(go.BSqrt2**n), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p1.b, orig_p1.b + mc.NumpyConfig.sqrt2 * n * orig_p1.e * _L_VALUE ** (-orig_p1.z)
        )

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)
        np.testing.assert_allclose(
            new_p2.b, orig_p2.b - mc.NumpyConfig.sqrt2 * n * orig_p2.e * _L_VALUE ** (-orig_p2.z)
        )

    @pytest.mark.parametrize("state", tu.make_states(100))
    def test_apply_z(self, state: lattice_state.SelingerState):
        ns = state.apply(lattice_state.GridOperatorAction(go.ZSqrt2), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)

        for orig, new in (orig_p1, new_p1), (orig_p2, new_p2):
            np.testing.assert_almost_equal(new.e, orig.e)
            np.testing.assert_almost_equal(new.z, orig.z)
            np.testing.assert_almost_equal(new.b, -orig.b)

        np.testing.assert_almost_equal(ns.bias(mc.NumpyConfig), state.bias(mc.NumpyConfig))
        np.testing.assert_almost_equal(ns.skew(mc.NumpyConfig), state.skew(mc.NumpyConfig))

    @pytest.mark.parametrize("state", tu.make_states(100))
    def test_apply_x(self, state: lattice_state.SelingerState):
        ns = state.apply(lattice_state.GridOperatorAction(go.XSqrt2), mc.NumpyConfig)

        orig_p1 = state.m1.parametric_form(mc.NumpyConfig)
        new_p1 = ns.m1.parametric_form(mc.NumpyConfig)

        orig_p2 = state.m2.parametric_form(mc.NumpyConfig)
        new_p2 = ns.m2.parametric_form(mc.NumpyConfig)

        for orig, new in (orig_p1, new_p1), (orig_p2, new_p2):
            np.testing.assert_almost_equal(new.e, orig.e)
            np.testing.assert_almost_equal(new.z, -orig.z)
            np.testing.assert_almost_equal(new.b, orig.b)

        np.testing.assert_almost_equal(ns.bias(mc.NumpyConfig), -state.bias(mc.NumpyConfig))
        np.testing.assert_almost_equal(ns.skew(mc.NumpyConfig), state.skew(mc.NumpyConfig))

    @pytest.mark.parametrize("state", tu.make_states(10))
    @pytest.mark.parametrize("g", tu.random_grid_operator(10, 4))
    @pytest.mark.parametrize("k", range(-3, 4))
    def test_apply_shifted(self, state: lattice_state.SelingerState, g: go.GridOperator, k: int):
        desired = (
            state.shift(k, mc.NumpyConfig)
            .apply(lattice_state.GridOperatorAction(g), mc.NumpyConfig)
            .shift(k, mc.NumpyConfig)
        )
        got = state.apply(lattice_state.GridOperatorAction(g).shift(k), mc.NumpyConfig)
        np.testing.assert_allclose(got.m1.D, desired.m1.D)
        np.testing.assert_allclose(got.m2.D, desired.m2.D)
