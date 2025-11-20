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

"""Contains methods for integer point enumeration."""

from typing import Iterator

import numpy as np

import qualtran.rotation_synthesis.lattice.geometry as geometry
import qualtran.rotation_synthesis.lattice.grid_operators as go
import qualtran.rotation_synthesis.math_config as mc
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.lattice import state

LAMBDA = rings.ZSqrt2(1, 1)
LAMBDA_INV = rings.ZSqrt2(-1, 1)
NEG_LAMBDA = rings.ZSqrt2(-1, -1)
NEG_LAMBDA_INV = -LAMBDA_INV


def enumerate_1d(
    inter: geometry.Range, comp_inter: geometry.Range, config: mc.MathConfig
) -> Iterator[rings.ZSqrt2]:
    r"""Yield points $p \in \mathbb{Z}[\sqrt{2}]$ contained in the region.

    Point $p$ belongs to the region iff $p \in inter$ and $p^sbullet \in \texit{comp_inter}$

    Follows section 4 of https://arxiv.org/abs/1403.2975

    Args:
        inter: first interval.
        comp_inter: the second (complementary) interval.
        config: the math config to use.
    """
    l_value = 1 + config.sqrt2
    inv_l = 1 / l_value
    width = inter.width()
    n = int(config.ceil(config.log(width) / config.log(l_value)))

    if n >= 0:
        lambda_inv = LAMBDA_INV
        neg_lambda = NEG_LAMBDA
    else:
        lambda_inv = LAMBDA
        neg_lambda = NEG_LAMBDA_INV

    x0, x1 = np.array([inter.start, inter.end]) * (lambda_inv ** abs(n)).value(config.sqrt2)
    y0, y1 = np.array([comp_inter.start, comp_inter.end]) * (neg_lambda ** abs(n)).value(
        config.sqrt2
    )
    if n % 2 == 1:
        y0, y1 = y1, y0
    assert x1 - x0 >= inv_l

    if n >= 0:
        ln = LAMBDA**n
    else:
        ln = LAMBDA_INV ** (-n)

    scale = 2 * config.sqrt2
    low, high = (x0 - y1) / scale, (x1 - y0) / scale
    for b in range(config.floor(low) - 1, config.ceil(high) + 2):
        left, right = x0 - b * config.sqrt2, x1 - b * config.sqrt2
        for a in range(config.floor(left) - 1, config.ceil(right) + 2):
            v = rings.ZSqrt2(a, b) * ln
            if inter.contains(v.value(config.sqrt2), config) and comp_inter.contains(
                v.conjugate().value(config.sqrt2), config
            ):
                yield v


def enumerate_upright(
    r: geometry.Rectangle, comp_r: geometry.Rectangle, config: mc.MathConfig
) -> Iterator[rings.ZW]:
    r"""Yield $p \in \mathbb{Z}[e^{i \pi/4}]$ such that $p \in r$ and $p^\sbullet \in \textit{comp_r}$

    Follows section 5 of https://arxiv.org/abs/1403.2975

    Args:
        r: A rectangle such that $p \in r$.
        comp_r: A rectangle such that $p^\bullet \in comp_r$.
        config: The MathConfig to use.

    Yields:
        Points $p$ such that $p \in r$ and $p^\sbullet \in \textit{comp_r}$
    """
    second_part = tuple(enumerate_1d(r.y_bounds, comp_r.y_bounds, config))
    for a in enumerate_1d(r.x_bounds, comp_r.x_bounds, config):
        for b in second_part:
            yield rings.ZW.from_pair(a, b, include_w=False)

    r_prime = r.shift(-1 / config.sqrt2, -1 / config.sqrt2)
    comp_r_prime = comp_r.shift(1 / config.sqrt2, 1 / config.sqrt2)
    second_part = tuple(enumerate_1d(r_prime.y_bounds, comp_r_prime.y_bounds, config))
    for a in enumerate_1d(r_prime.x_bounds, comp_r_prime.x_bounds, config):
        for b in second_part:
            yield rings.ZW.from_pair(a, b, include_w=True)


def get_overall_action(
    s: state.SelingerState, config: mc.MathConfig, verbose: bool = False
) -> state.GridOperatorAction:
    """Returns GridOperatorAction whose effect on the state reduces its skew to be below 15

    Args:
        s: The state.
        config: The MathConfig to use.
        verbose: Whether to print debug statements or not.

    Returns:
        A GridOperatorAction whose effect reduces the skew of the state to be less that 15.
    """
    s = state.SelingerState(s.m1.normalize(config), s.m2.normalize(config))
    overall_action = state.GridOperatorAction(go.ISqrt2)
    i = 0
    if verbose:
        skew, bias = s.skew(config), s.bias(config)
        print(f"{i}. {skew=} {bias=}")
    while s.skew(config) > 15:
        old = s.skew(config)
        action = s.get_grid_operator(config)
        overall_action = overall_action.followed_by(action)
        s = s.apply(action, config)
        i += 1
        if verbose:
            skew, bias = s.skew(config), s.bias(config)
            print(f"{i}. {skew=} {bias=}")
        assert s.skew(config) <= 0.9 * old
    return overall_action


def get_points_from_state(s: state.SelingerState, config: mc.MathConfig) -> Iterator[rings.ZW]:
    r"""Yields the points $p \in \mathbb{Z}[\omega]$ contained in the given state."""
    for p in enumerate_upright(s.m1.bounding_box(config), s.m2.bounding_box(config), config):
        if s.contains(p, config):
            yield p
