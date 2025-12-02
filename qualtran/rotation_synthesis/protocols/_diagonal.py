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

from typing import Callable, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import attrs
import numpy as np
from matplotlib import patches

import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.protocols._protocol as _protocol
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.rings import _zsqrt2


def make_ellipse_for_circular_segment(
    chord_length: rst.Real, radius: rst.Real, theta: rst.Real, config: mc.MathConfig
) -> lattice.Ellipse:
    r"""Constructs the smallest ellipse that contains the given circular segement.

    Given a circular segment on one side of the chord $\overset{\huge\frown}{AB}$ with point $p$
    at the center of $\overline{AB}$, the smallest ellipse that contains it has center $p$ and axes
    $\overline{AB}$ and $\overline{CD}$ where $C$ and $D$ lie $\overrightarrow{Op}$ with $D$ at
    intersection of the circle perimeter with the line and $\overline{Cp} = \overline{pD}$

    Args:
        chord_length: The length of the chord bounding the segement.
        radius: The radius of the circle.
        theta: The angle that the point $p$ makes with the x-axis.
        config: A math config.

    Returns:
        The smallest ellipse that contains the circular segment.
    """
    r = config.sqrt(1 - (chord_length / 2) ** 2)
    a = 1 - r
    b = chord_length / 2

    c, s = config.cos(theta), config.sin(theta)
    center = np.array([c, s]) * r * radius
    e1 = lattice.Ellipse.from_axes(radius * a, radius * b, theta, center, config)
    return e1


@attrs.frozen
class Diagonal(_protocol.ApproxProblem):
    r"""Approximate a Z-rotation with a string of Clifford+T gates.

    The inequality $D(U, e^{i\theta Z}) < \epsilon$ where $D$ is a distance measure (e.g. diamond)
    puts a geometric constraint on the upper left element of $U$ so that it belongs to a circular
    segment.

    Attributes:
        theta: the target angle.
        eps: the target error budget.
        max_n: the maximum number of T gates to try.

    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        Section 3.2

    """

    theta: rst.Real
    eps: rst.Real
    max_n: int

    def make_state(self, n: int, config: mc.MathConfig, offset: int) -> lattice.SelingerState:
        theta = self.theta
        if offset:
            theta += config.pi

        r0 = _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, config)
        r1 = _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV_CONJ, n, config)

        e1 = make_ellipse_for_circular_segment(self.eps, r0, theta, config)
        e2 = lattice.Ellipse.from_axes(
            r1, r1, config.zero, np.array([config.zero, config.zero]), config
        )
        # phi = 2*np.arcsin(float(eps/2))
        # target_area = 0.5*r0**2*(phi - config.sin(phi))
        # print(f'{phi=} {target_area=} actual area={e1.area(config)} ratio={e1.area(config) / target_area}')
        return lattice.SelingerState(e1, e2)

    def make_real_bound_fn(self, n: int, config: mc.MathConfig) -> Callable[[rings.ZW], bool]:
        neg_rot = config.cos(self.theta) - config.sin(self.theta) * 1j

        def fn(p):
            u = p.value(config.sqrt2)
            u = u / _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, config)
            abs_p2, _, _ = (p * p.conj()).to_zsqrt2()
            target = 2 * _zsqrt2.LAMBDA_KLIUCHNIKOV**n
            if abs_p2 > target:
                return False
            diamond_distance_squared = 4 * (1 - (u * neg_rot).real ** 2)
            return diamond_distance_squared <= self.eps**2

        return fn

    def get_points(
        self, config: mc.MathConfig, verbose: bool = False
    ) -> Iterator[tuple[int, rings.ZW]]:
        n = 0
        while True:
            if verbose:
                print(f"{n=}", flush=True)
            for offset in False, True:
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
        show_legend: bool = True,
        *,
        offset: bool = False,
    ) -> plt.Axes:
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))

        state = self.make_state(n, mc.NumpyConfig, offset)
        r = float(_zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, mc.NumpyConfig))

        state.m1.plot(ax, add_label=False, fill=False, alpha=0)
        ax.relim()
        ax.autoscale_view()

        delta = np.arcsin(float(self.eps) / 2)
        ax.add_patch(
            patches.Wedge(
                (0, 0),
                r,
                theta1=(self.theta - delta) * 180 / np.pi,
                theta2=(self.theta + delta) * 180 / np.pi,
                color="green",
                label="target region",
            )
        )

        ax.add_patch(patches.Circle((0, 0), r, fill=False, color="black"))

        sin_delta = self.eps / 2
        cos_delta = np.sqrt(1 - sin_delta**2)
        line = np.array([[cos_delta, sin_delta], [cos_delta, -sin_delta]])
        cos_theta, sin_theta = np.cos(float(self.theta)), np.sin(float(self.theta))
        rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        line = line @ rot_matrix.T
        tri = np.array([line[0], line[1], [0, 0]])
        ax.add_patch(patches.Polygon(r * tri, closed=True, fill=True, color="white"))
        state.m1.plot(ax, add_label=show_legend, fill=False)
        if show_legend:
            ax.legend()
        if fig is not None:
            fig.show()
        return ax
