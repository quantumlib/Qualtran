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

import abc
from typing import Callable, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import qualtran.rotation_synthesis.channels as channels
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.rings as rings


class ApproxProblem(abc.ABC):
    """An abstract class representing an approximation problem."""

    @abc.abstractmethod
    def make_state(self, n: int, config: mc.MathConfig, **kwargs) -> lattice.SelingerState:
        """Create a State corresponding to the geometric problem defined by the protocol at `n`."""

    @abc.abstractmethod
    def make_real_bound_fn(self, n: int, config: mc.MathConfig) -> Callable[[rings.ZW], bool]:
        """Create a function that determines whether a lattice point is inside the target region or not."""

    @abc.abstractmethod
    def get_points(self, config: mc.MathConfig, verbose: bool) -> Iterator[tuple[int, rings.ZW]]:
        """Yields the lattice points inside the region."""

    def plot(self, n: int, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plots the target region"""
        raise NotImplementedError()


class PointCollector(abc.ABC):
    """A point collector accepts valid points and decides whether to terminate the search.

    A collector receives valid points (i.e. points that belong to the search region) during integer
    point enumeration, decides whether the termination condition is reached and constructs the
    resulting quantum channel(s).
    """

    @abc.abstractmethod
    def add_point(self, p: rings.ZW, q: rings.ZW, n: int, config: mc.MathConfig) -> None:
        """Adds a valid point to the collector."""

    @abc.abstractmethod
    def is_done(self) -> bool:
        """Determines if the collector has recieved points that satisfy its objective."""

    @abc.abstractmethod
    def result(self):
        """Returns channels created using the accepted points."""

    @abc.abstractmethod
    def status(self) -> str:
        """Returns a human readable string represnting the current state of the collector"""


class SimplePointCollector(PointCollector):
    """Accepts the first `k` valid points."""

    def __init__(self, target_count: int = 1):
        super().__init__()
        self._points: list[tuple[rings.ZW, rings.ZW, int]] = []
        self._target_count = target_count

    def add_point(self, p: rings.ZW, q: rings.ZW, n: int, config: mc.MathConfig) -> None:
        self._points.append((p, q, n))

    def is_done(self) -> bool:
        return len(self._points) >= self._target_count

    def result(self) -> list[channels.Channel]:
        return [channels.UnitaryChannel(*p) for p in self._points]

    def status(self) -> str:
        return f"have={len(self._points)} need={self._target_count}"


class SplitRegionCollector(PointCollector):
    """splits recieved into the two spaces on either side of a line.

    A SplitRegionCollector terminates when at least `k` points are recieved for each side
    of the splitting line. where the splitting regions are determined by the sign of
    $ax+by-c$ where $x, y$ are the cooredinates of the point and $a,b,c$ are the parameters
    of the line.
    """

    def __init__(self, splitter_line, target_count: int = 1):
        super().__init__()
        self._left: list[tuple[rings.ZW, rings.ZW, int]] = []
        self._right: list[tuple[rings.ZW, rings.ZW, int]] = []
        self._splitter = splitter_line
        self._target_count = target_count

    def _evaluate(self, p: rings.ZW, n: int, config: mc.MathConfig):
        u = p.value(config.sqrt2)
        x, y = u.real, u.imag
        a, b, c = self._splitter
        return a * x + b * y - c

    def add_point(self, p: rings.ZW, q: rings.ZW, n: int, config: mc.MathConfig) -> None:
        side = self._evaluate(p, n, config)
        if side > 0:
            target = self._left
        elif side < 0:
            target = self._right
        elif len(self._left) < len(self._right):
            target = self._left
        else:
            target = self._right
        target.append((p, q, n))

    def is_done(self):
        return min(len(self._left), len(self._right)) >= self._target_count

    def result(
        self,
    ) -> tuple[tuple[channels.UnitaryChannel, ...], tuple[channels.UnitaryChannel, ...]]:
        return tuple(channels.UnitaryChannel(*x) for x in self._left), tuple(
            channels.UnitaryChannel(*x) for x in self._right
        )

    def status(self) -> str:
        left_incomplete = len(self._left) < self._target_count
        right_incomplete = len(self._right) < self._target_count
        if left_incomplete and right_incomplete:
            return "empty"
        if left_incomplete:
            return "waiting left"
        if right_incomplete:
            return "waiting right"
        return "ready"
