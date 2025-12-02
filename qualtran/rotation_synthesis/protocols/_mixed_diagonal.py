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

import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.protocols._diagonal as _diagonal
import qualtran.rotation_synthesis.protocols._protocol as _protocol
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.rings import _zsqrt2


@attrs.frozen
class MixedDiagonal(_protocol.ApproxProblem):
    r"""Approximate a Z-rotation with a twirled string of Clifford+T gates.

    Attributes:
        theta: the target angle.
        eps: the target error budget.
        max_n: the maximum number of T gates to try.

    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        Section 3.4

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

        e1 = _diagonal.make_ellipse_for_circular_segment(
            config.sqrt(self.eps / 2) * 2, r0, theta, config
        )
        e2 = lattice.Ellipse.from_axes(
            r1, r1, config.zero, np.array([config.zero, config.zero]), config
        )
        return lattice.SelingerState(e1, e2)

    def make_real_bound_fn(self, n: int, config: mc.MathConfig) -> Callable[[rings.ZW], bool]:
        neg_rot = config.cos(self.theta) - config.sin(self.theta) * 1j

        def fn(p):
            abs_p2, _, _ = (p * p.conj()).to_zsqrt2()
            target = 2 * _zsqrt2.LAMBDA_KLIUCHNIKOV**n
            if abs_p2 > target:
                return False
            u = p.value(config.sqrt2)
            u = u / _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, n, config)
            return (u * neg_rot).real ** 2 >= 1 - self.eps / 2

        return fn

    def get_points(
        self, config: mc.MathConfig, verbose: bool = False
    ) -> Iterator[tuple[int, rings.ZW]]:
        n = 0
        while True:
            if verbose:
                print(f"{n=}")
            os = self.make_state(n, config, offset=False)
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
        from matplotlib import patches

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))

        state = self.make_state(n, mc.NumpyConfig, offset)
        r = float(
            mc.NumpyConfig.sqrt((2 * _zsqrt2.LAMBDA_KLIUCHNIKOV**n).value(mc.NumpyConfig.sqrt2))
        )

        state.m1.plot(ax, add_label=False, fill=False, alpha=0)
        ax.relim()
        ax.autoscale_view()

        delta = np.arcsin(np.sqrt(float(self.eps) / 2))
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

        sin_delta = np.sin(delta)
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
