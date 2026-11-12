# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mpmath
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis.lattice._geometry as geometry
import qualtran.rotation_synthesis.lattice._test_utils as tu


class EllipseTestSuite:
    @pytest.mark.parametrize("ellipse", list(tu.make_ellipses(10)))
    def test_parametric_form(self, ellipse: lattice.Ellipse):
        p = ellipse.parametric_form(config=mc.NumpyConfig)
        l_value = 1 + np.sqrt(2)

        np.testing.assert_allclose(ellipse.D, p.to_ellipse(l_value).D)

    @pytest.mark.parametrize("ellipse", list(tu.make_ellipses(10)))
    def test_det(self, ellipse: lattice.Ellipse):
        np.testing.assert_allclose(ellipse._det, np.linalg.det(ellipse.D))

    @pytest.mark.parametrize("a", np.random.random(5))
    @pytest.mark.parametrize("b", np.random.random(5))
    @pytest.mark.parametrize("theta", np.random.random(5) * np.pi)
    def test_bounding_tilt(self, a: rst.Real, b: rst.Real, theta: np.ndarray):
        if a < b:
            # To make the choice of the 4 possibilities unique, we make 0 < theta < pi and a >= b.
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

        # Analytic bounds.
        x_max = np.sqrt(a**2 * c**2 + b**2 * s**2)
        y_max = np.sqrt(a**2 * s**2 + b**2 * c**2)
        assert bbox.x_bounds == lattice.Range(pytest.approx(-x_max), pytest.approx(x_max))
        assert bbox.y_bounds == lattice.Range(pytest.approx(-y_max), pytest.approx(y_max))


def test_ellipse_plot_creates_axes_and_labeled_patch(monkeypatch: pytest.MonkeyPatch):
    ellipse = lattice.Ellipse(np.array([[0.25, 0.05], [0.05, 0.11]]), np.array([1.0, 2.0]))
    shown_figures = []
    monkeypatch.setattr(Figure, "show", lambda fig: shown_figures.append(fig))  # noqa: PLW0108

    ax = ellipse.plot()
    try:
        assert shown_figures == [ax.figure]
        assert len(ax.patches) == 1

        patch = ax.patches[0]
        theta = float(ellipse.tilt(mc.NumpyConfig))
        rotated = ellipse.rotate(theta, mc.NumpyConfig)

        assert patch.center == pytest.approx(ellipse.center)
        assert patch.width == pytest.approx(2 / np.sqrt(rotated.D[0, 0]))
        assert patch.height == pytest.approx(2 / np.sqrt(rotated.D[1, 1]))
        assert patch.angle == pytest.approx(-theta * 180 / np.pi)
        assert patch.get_label().startswith("theta=")
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)


def test_ellipse_plot_uses_supplied_axes_without_label():
    ellipse = lattice.Ellipse(np.array([[0.25, 0.05], [0.05, 0.11]]), np.array([1.0, 2.0]))
    fig, ax = plt.subplots()
    try:
        result = ellipse.plot(ax=ax, add_label=False, facecolor="red", alpha=0.5)

        assert result is ax
        assert len(ax.patches) == 1

        patch = ax.patches[0]
        assert patch.get_label() is None
        assert patch.get_facecolor() == pytest.approx((1.0, 0.0, 0.0, 0.5))
        assert ax.get_legend() is None
    finally:
        plt.close(fig)


def test_ellipse_area():
    ellipse = lattice.Ellipse(np.array([[4.0, 1.0], [1.0, 2.0]]))

    assert ellipse.area(mc.NumpyConfig) == pytest.approx(np.pi / np.sqrt(7))


def test_ellipse_normalize_has_area_pi():
    ellipse = lattice.Ellipse(np.array([[4.0, 1.0], [1.0, 2.0]]), np.array([1.0, -2.0]))

    normalized = ellipse.normalize(mc.NumpyConfig)

    assert normalized.area(mc.NumpyConfig) == pytest.approx(np.pi)
    np.testing.assert_allclose(normalized.center, ellipse.center)


@pytest.mark.parametrize(
    ("x", "y", "expected"), [(2.0, -1.0, True), (6.0, -1.0, True), (6.01, -1.0, False)]
)
def test_ellipse_contains(x, y, expected):
    ellipse = lattice.Ellipse(np.diag([1 / 4**2, 1 / 3**2]), np.array([2.0, -1.0]))

    assert ellipse.contains(x, y, mc.NumpyConfig) == expected


def test_ellipse_rotate_preserves_shape_determinant_and_center():
    ellipse = lattice.Ellipse(np.array([[4.0, 1.0], [1.0, 2.0]]), np.array([1.0, -2.0]))
    theta = np.pi / 3
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotated = ellipse.rotate(theta, mc.NumpyConfig)

    np.testing.assert_allclose(rotated.D, rotation @ ellipse.D @ rotation.T)
    assert rotated._det == pytest.approx(ellipse._det)
    np.testing.assert_allclose(rotated.center, ellipse.center)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(1.0, 2.0, True), (0.0, 0.0, True), (-0.1, 2.0, False), (1.0, 3.1, False)],
)
def test_rectangle_contains(x, y, expected):
    rectangle = lattice.Rectangle(lattice.Range(0.0, 1.0), lattice.Range(0.0, 2.0))

    assert rectangle.contains(x, y, mc.NumpyConfig) == expected


def test_range_contains_boundaries_and_approximately_equal_values():
    interval = lattice.Range(1.0, 2.0)

    assert interval.contains(1.0, mc.NumpyConfig)
    assert interval.contains(2.0, mc.NumpyConfig)
    assert interval.contains(1.0 - 1e-9, mc.NumpyConfig)
    assert interval.contains(2.0 + 1e-9, mc.NumpyConfig)
    assert not interval.contains(1.0 - 1e-3, mc.NumpyConfig)
    assert not interval.contains(2.0 + 1e-3, mc.NumpyConfig)


def test_range_shift_changes_endpoints():
    shifted = lattice.Range(1.0, 2.0).shift(-3.5)

    assert shifted.start == pytest.approx(-2.5)
    assert shifted.end == pytest.approx(-1.5)


def test_ellipse_as_formula():
    ellipse = lattice.Ellipse(np.array([[2.14, 0.57], [0.57, 3.36]]), np.array([1.24, 2.36]))

    assert ellipse.as_formula() == (
        "2.14*(x-1.24)^2 + 1.14*(x-1.24)*(y-2.36) + 3.36*(y-2.36)^2 = 1"
    )

    assert ellipse.as_formula(".1f") == ("2.1*(x-1.2)^2 + 1.1*(x-1.2)*(y-2.4) + 3.4*(y-2.4)^2 = 1")


@pytest.mark.parametrize(
    ("bounds", "expected_type"),
    [
        ((mpmath.mpf("1.5"), 2), mpmath.mpf),
        ((1, 2), int),
        ((np.float64(1.5), 2), np.longdouble),
        ((1.5, 2.5), float),
    ],
)
def test_range_from_bounds_preserves_numeric_type(bounds, expected_type):
    interval = lattice.Range.from_bounds(*bounds)

    assert isinstance(interval.start, expected_type)
    assert isinstance(interval.end, expected_type)


def test_range_width_and_shift():
    interval = lattice.Range(2, 7)

    assert interval.width() == 5
    assert interval.shift(3) == lattice.Range(5, 10)


def test_rectangle_area_and_shift():
    rectangle = lattice.Rectangle(lattice.Range(1, 4), lattice.Range(2, 6))

    assert rectangle.area() == 12
    assert rectangle.shift(2, -1) == lattice.Rectangle(lattice.Range(3, 6), lattice.Range(1, 5))


def test_rectangle_make_square():
    bounds = lattice.Range(-2, 2)

    square = lattice.Rectangle.make_square(bounds)

    assert square.x_bounds == bounds
    assert square.y_bounds == bounds
    assert square.area() == 16


def test_ellipse_parametric_form_to_ellipse():
    parametric_form = geometry.EllipseParametricForm(e=2.0, z=1.0, b=0.5)

    ellipse = parametric_form.to_ellipse(l_value=3.0)

    np.testing.assert_allclose(ellipse.D, [[2 / 3, 0.5], [0.5, 6.0]])


def test_ellipse_accepts_mpmath_matrix_with_approximately_symmetric_entries():
    ellipse = lattice.Ellipse(
        np.array(
            [[mpmath.mpf(2), mpmath.mpf("0.5")], [mpmath.mpf("0.5000000001"), mpmath.mpf(2)]],
            dtype=object,
        )
    )

    assert isinstance(ellipse.D[0, 0], mpmath.mpf)


def test_ellipse_rectangleness_for_diagonal_matrix():
    ellipse = lattice.Ellipse(np.diag([1.0, 1.0]))

    assert ellipse.rectangleness(mc.NumpyConfig) == pytest.approx(np.pi / 4)
