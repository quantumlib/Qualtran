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

import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis.lattice.test_utils as tu
import qualtran.rotation_synthesis.math_config as mc


class TestEllipse:
    @pytest.mark.parametrize("ellipse", tu.make_ellipses(10))
    def test_parametric_form(self, ellipse: lattice.Ellipse):
        p = ellipse.parametric_form(config=mc.NumpyConfig)
        l_value = 1 + np.sqrt(2)

        np.testing.assert_allclose(ellipse.D, p.to_ellipse(l_value).D)

    @pytest.mark.parametrize("ellipse", tu.make_ellipses(10))
    def test_det(self, ellipse: lattice.Ellipse):
        np.testing.assert_allclose(ellipse._det, np.linalg.det(ellipse.D))

    @pytest.mark.parametrize("a", np.random.random(5))
    @pytest.mark.parametrize("b", np.random.random(5))
    @pytest.mark.parametrize("theta", np.random.random(5) * np.pi)
    def test_bounding_tilt(self, a: rst.Real, b: rst.Real, theta: np.ndarray):
        if a < b:
            # to make the choice of the 4 possiblities unique we make 0 < theta < pi and a >= b
            a, b = b, a
        e = lattice.Ellipse.from_axes(a, b, theta, np.zeros(2), mc.NumpyConfig)
        np.testing.assert_allclose(e.tilt(mc.NumpyConfig), np.pi - theta)

    @pytest.mark.parametrize("a", np.random.random(5))
    @pytest.mark.parametrize("b", np.random.random(5))
    @pytest.mark.parametrize("center", np.random.random((5, 2)))
    def test_bounding_box_no_rotation(self, a: rst.Real, b: rst.Real, center: np.ndarray):
        e = lattice.Ellipse(np.diagflat([1 / a, 1 / b]) ** 2, center)
        bbox = e.bounding_box(mc.NumpyConfig)
        x_max = a
        y_max = b
        assert bbox.x_bounds == lattice.Range(
            pytest.approx(center[0] - x_max), pytest.approx(center[0] + x_max)
        )
        assert bbox.y_bounds == lattice.Range(
            pytest.approx(center[1] - y_max), pytest.approx(center[1] + y_max)
        )

    @pytest.mark.parametrize("a", np.random.random(5) * 10)
    @pytest.mark.parametrize("b", np.random.random(5) * 10)
    @pytest.mark.parametrize("theta", (2 * np.random.random(5) - 1) * 2 * np.pi)
    def test_bounding_box_with_rotation(self, a: rst.Real, b: rst.Real, theta: np.ndarray):
        e = lattice.Ellipse.from_axes(a, b, theta, np.zeros(2), mc.NumpyConfig)
        bbox = e.bounding_box(mc.NumpyConfig)
        c, s = np.cos(theta), np.sin(theta)
        # Analytic bounds
        x_max = np.sqrt(a**2 * c**2 + b**2 * s**2)
        y_max = np.sqrt(a**2 * s**2 + b**2 * c**2)
        assert bbox.x_bounds == lattice.Range(pytest.approx(-x_max), pytest.approx(x_max))
        assert bbox.y_bounds == lattice.Range(pytest.approx(-y_max), pytest.approx(y_max))
