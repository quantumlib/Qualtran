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

from __future__ import annotations

import functools
from typing import Callable, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import attrs
import numpy as np

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis.protocols._protocol as _protocol
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.rings import _zsqrt2


@functools.cache
def create_ellipse_for_error(eps: rst.Real, success_probability: rst.Real, config: mc.MathConfig):
    """Creates the smallest ellipse that contains an annular sector.

    The search area for the fallback protocol is an annular sector. The smallest ellipse that contains
    an annular sector is the smallest ellipse that contains its 5 extreme points.

    This method assumes that the annular sector has a center on the x-axis, with the bigger circle
    being the unit circle. An ellipse is defined using 5 parameters, because of the assumptions we
    know 2 of them, namely that both the y-coord of the center and the rotation angle of the ellipse
    are zeros.

    Given the x-coord of the center and one of the axis, the second axis can be computed from the
    ellipse equation. This means that we need to find two values x-coord of the center and one of
    the axis that minimize the area of the ellipse. The function that maps x-coords and axis length
    to ellipse area is convex which allows us to do ternary search on both parameters.

    Args:
        eps: The target error, equals the angle of the sector.
        success_probability: The target success probability, equals the square of the radius
            of the smaller circle.
        config: A math config.

    Returns:
        An ellipse.
    """
    sqrt_q = config.sqrt(success_probability)

    sin_delta = eps / 2
    cos_delta = config.sqrt(1 - sin_delta**2)

    points = np.array(
        [
            [config.one, config.zero],
            [cos_delta, sin_delta],
            [cos_delta, -sin_delta],
            [sqrt_q * cos_delta, sqrt_q * sin_delta],
            [sqrt_q * cos_delta, -sqrt_q * sin_delta],
        ]
    )

    def b_for_a(pivot: rst.Real, a: rst.Real) -> rst.Real:
        b = 1e-100
        for x, y in points:
            dx2 = (x - pivot) ** 2
            b = max(b, y * a / config.sqrt(a**2 - dx2))
        return b

    def sides_for_pivot(pivot: rst.Real) -> tuple[rst.Real, rst.Real]:
        s = np.abs(points[:, 0] - pivot).max()
        e = 3 * s
        while e - s > eps:
            m1 = s + (e - s) / 3
            m2 = m1 + (e - s) / 3
            if m1 * b_for_a(pivot, m1) < m2 * b_for_a(pivot, m2):
                e = m2
            else:
                s = m1
        return e, b_for_a(pivot, e)

    def area(pivot: rst.Real) -> rst.Real:
        a, b = sides_for_pivot(pivot)
        return a * b

    s, e = config.zero, config.one
    while e - s > eps:
        m1 = s + (e - s) / 3
        m2 = m1 + (e - s) / 3
        if area(m1) < area(m2):
            e = m2
        else:
            s = m1
    pos = s + (e - s) / 2
    a, b = sides_for_pivot(pos)
    return a, b, np.array([pos, config.zero])


@attrs.frozen
class Fallback(_protocol.ApproxProblem):
    r"""Approximate a Z-rotation with a fallback channel.

    Approximate the $Rz(2\theta)$ with the channel
    q: ─────────@───V───@───────Y───C───
                │       │       ║   ║
    ancilla: ───X───────X───M───╫───╫───
                            ║   ║   ║
    m: ═════════════════════@═══^═══^═══

    where a cheap (in terms of T gates) rotation `V` is applied and then depending on
    the result of the measurement we may need to apply an expensive correction. The problem
    is formulated so that we need to apply the correction in afew (<1%) occurences.

    Attributes:
        theta: the target angle.
        success_probability: The target success probability.
        eps: the target error budget.
        max_n: the maximum number of T gates to try.

    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        Section 3.3
    """

    theta: rst.Real
    success_probability: rst.Real
    eps: rst.Real
    max_n: int
    offset_angle: bool = True
    filter_by_dist: bool = True

    def make_state(self, n: int, config: mc.MathConfig, offset: bool) -> lattice.SelingerState:
        theta = self.theta
        if offset:
            theta += config.pi

        r0 = _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, config)
        r1 = _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV_CONJ, n, config)

        cos_t, sin_t = config.cos(theta), config.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        x_axis, y_axis, center = create_ellipse_for_error(
            self.eps, self.success_probability, config
        )
        e1 = lattice.Ellipse.from_axes(
            r0 * x_axis, r0 * y_axis, theta, r0 * (rotation_matrix @ center), config
        )
        e2 = lattice.Ellipse.from_axes(
            r1, r1, config.zero, np.array([config.zero, config.zero]), config
        )
        return lattice.SelingerState(e1, e2)

    def make_real_bound_fn(self, n: int, config: mc.MathConfig) -> Callable[[rings.ZW], bool]:
        def fn(p):
            abs_p2 = (p * p.conj()).to_zsqrt2()[0]
            target = 2 * _zsqrt2.LAMBDA_KLIUCHNIKOV**n
            if abs_p2 > target:
                return False
            u = p.value(config.sqrt2)
            u = u / _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, config)
            abs_u2 = u.real**2 + u.imag**2
            if abs_u2 < self.success_probability and not config.isclose(
                abs_u2, self.success_probability
            ):
                return False
            arg_u = config.arctan2(u.imag, u.real)
            if not self.filter_by_dist:
                return True
            options = [self.theta]
            if self.offset_angle:
                options.append(self.theta + config.pi)
            for t in options:
                if 2 * abs(config.sin(abs(arg_u - t))) <= self.eps:
                    return True
            return False

        return fn

    def get_points(
        self, config: mc.MathConfig, verbose: bool = False
    ) -> Iterator[tuple[int, rings.ZW]]:
        n = 0
        while True:
            if verbose:
                print(f"{n=}")
            options = [False]
            if self.offset_angle:
                options.append(True)
            for offset in options:
                os = self.make_state(n, config, offset=offset)
                overall_action = lattice.get_overall_action(os, config)
                fin_state = os.apply(overall_action, config)
                for p in lattice.get_points_from_state(fin_state, config):
                    q = overall_action.g.apply(p)
                    yield n, q
            n += 1
            if n > self.max_n:
                break

    def plot(
        self,
        n: int,
        ax: Optional[plt.Axes] = None,
        *,
        offset: bool = False,
        show_legend: bool = True,
    ) -> plt.Axes:
        import matplotlib.pyplot as plt
        from matplotlib import patches

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))

        state = self.make_state(n, mc.NumpyConfig, offset)
        r = float(_zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, mc.NumpyConfig))

        state.m1.plot(ax, add_label=False, fill=False, alpha=0)
        ax.relim()
        ax.autoscale_view()

        delta = np.arcsin(float(self.eps) / 2)
        sqrt_q = np.sqrt(float(self.success_probability))
        ax.add_patch(
            patches.Wedge(
                (0, 0),
                r,
                theta1=(self.theta - delta) * 180 / np.pi,
                theta2=(self.theta + delta) * 180 / np.pi,
                width=r * (1 - sqrt_q),
                color="green",
                label="target region",
            )
        )

        ax.add_patch(patches.Circle((0, 0), r, fill=False, color="black"))
        ax.add_patch(patches.Circle((0, 0), r * sqrt_q, fill=False, color="black"))

        state.m1.plot(ax, add_label=show_legend, fill=False)
        if show_legend:
            ax.legend()
        if fig is not None:
            fig.show()
        return ax
