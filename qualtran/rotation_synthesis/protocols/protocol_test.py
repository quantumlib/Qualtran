# Copyright 2026 Google LLC
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

from typing import Callable, Iterator

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.channels as channels
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis.protocols as protocols
import qualtran.rotation_synthesis.rings as rings


@pytest.mark.parametrize(
    "problem",
    [
        protocols.Diagonal(theta=0.3, eps=0.2, max_n=1),
        protocols.Fallback(theta=0.3, success_probability=0.9, eps=0.2, max_n=1),
        protocols.MixedDiagonal(theta=0.3, eps=0.2, max_n=1),
    ],
)
def test_protocol_plot_uses_supplied_axes_and_honors_legend(problem):
    fig, ax = plt.subplots()
    try:
        result = problem.plot(1, ax=ax, show_legend=False, offset=True)

        assert result is ax
        assert ax.patches
        assert ax.get_legend() is None
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "problem",
    [
        protocols.Diagonal(theta=0.3, eps=0.2, max_n=1),
        protocols.Fallback(theta=0.3, success_probability=0.9, eps=0.2, max_n=1),
        protocols.MixedDiagonal(theta=0.3, eps=0.2, max_n=1),
    ],
)
def test_protocol_plot_creates_figure_and_legend(problem, monkeypatch: pytest.MonkeyPatch):
    shown_figures = []
    monkeypatch.setattr(Figure, "show", lambda fig: shown_figures.append(fig))  # noqa: PLW0108

    ax = problem.plot(1)
    try:
        assert shown_figures == [ax.figure]
        assert ax.patches
        assert ax.get_legend() is not None
    finally:
        plt.close(ax.figure)


class _ConcreteApproxProblem(protocols.ApproxProblem):
    def make_state(self, n: int, config: mc.MathConfig, **kwargs) -> lattice.SelingerState:
        raise NotImplementedError

    def make_real_bound_fn(self, n: int, config: mc.MathConfig) -> Callable[[rings.ZW], bool]:
        raise NotImplementedError

    def get_points(
        self, config: mc.MathConfig, verbose: bool
    ) -> Iterator[tuple[int, rings.ZW]]:
        return iter(())


def test_approx_problem_plot_is_not_implemented():
    with pytest.raises(NotImplementedError):
        _ConcreteApproxProblem().plot(0)


def test_simple_point_collector_lifecycle():
    collector = protocols.SimplePointCollector(target_count=2)
    p = rings.ZW((1, 0, 0, 0))
    q = rings.ZW((0, 1, 0, 0))

    assert collector.status() == "have=0 need=2"
    assert not collector.is_done()

    collector.add_point(p, q, 1, mc.NumpyConfig)
    assert collector.status() == "have=1 need=2"
    assert not collector.is_done()

    collector.add_point(q, p, 2, mc.NumpyConfig)
    assert collector.status() == "have=2 need=2"
    assert collector.is_done()
    assert collector.result() == [
        channels.UnitaryChannel(p, q, 1),
        channels.UnitaryChannel(q, p, 2),
    ]


def test_split_region_collector_lifecycle_and_boundary_assignment():
    collector = protocols.SplitRegionCollector((1, 0, 0), target_count=2)
    positive = rings.ZW((1, 0, 0, 0))
    negative = rings.ZW((-1, 0, 0, 0))
    boundary = rings.ZW((0, 0, 0, 0))
    q = rings.ZW((1, 0, 0, 0))

    assert collector._evaluate(positive, 0, mc.NumpyConfig) == pytest.approx(1)
    assert collector._evaluate(negative, 0, mc.NumpyConfig) == pytest.approx(-1)
    assert collector._evaluate(boundary, 0, mc.NumpyConfig) == pytest.approx(0)
    assert collector.status() == "empty"
    assert not collector.is_done()

    collector.add_point(positive, q, 1, mc.NumpyConfig)
    assert collector.status() == "empty"
    collector.add_point(boundary, q, 2, mc.NumpyConfig)
    assert collector.status() == "empty"
    collector.add_point(positive, q, 3, mc.NumpyConfig)
    assert collector.status() == "waiting right"
    assert not collector.is_done()
    collector.add_point(negative, q, 4, mc.NumpyConfig)
    assert collector.status() == "ready"
    assert collector.is_done()

    left, right = collector.result()
    assert left == (
        channels.UnitaryChannel(positive, q, 1),
        channels.UnitaryChannel(positive, q, 3),
    )
    assert right == (
        channels.UnitaryChannel(boundary, q, 2),
        channels.UnitaryChannel(negative, q, 4),
    )


def test_split_region_collector_waiting_left_and_boundary_fallback():
    collector = protocols.SplitRegionCollector((1, 0, 0), target_count=2)
    positive = rings.ZW((1, 0, 0, 0))
    negative = rings.ZW((-1, 0, 0, 0))
    boundary = rings.ZW((0, 0, 0, 0))
    q = rings.ZW((1, 0, 0, 0))

    collector.add_point(negative, q, 1, mc.NumpyConfig)
    collector.add_point(negative, q, 2, mc.NumpyConfig)
    assert collector.status() == "waiting left"
    assert not collector.is_done()

    collector.add_point(boundary, q, 3, mc.NumpyConfig)
    collector.add_point(positive, q, 4, mc.NumpyConfig)
    assert collector.status() == "ready"
    assert collector.is_done()

    left, right = collector.result()
    assert left == (
        channels.UnitaryChannel(boundary, q, 3),
        channels.UnitaryChannel(positive, q, 4),
    )
    assert right == (
        channels.UnitaryChannel(negative, q, 1),
        channels.UnitaryChannel(negative, q, 2),
    )
