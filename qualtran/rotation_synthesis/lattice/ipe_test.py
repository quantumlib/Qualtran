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
import math

import numpy as np
import pytest

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.lattice as lattice
import qualtran.rotation_synthesis.lattice._test_utils as tu
import qualtran.rotation_synthesis.rings as rings


def test_enumerate1d_requiring_swapping():
    A = lattice.Range(start=-5.535533905932738, end=4.121320343559642)
    B = lattice.Range(start=0.2928932188134524, end=1.1213203435596426)
    assert set(lattice.enumerate_1d(A, B, mc.NumpyConfig)) == {
        rings.ZSqrt2(a=-2, b=-2),
        rings.ZSqrt2(a=-1, b=-1),
        rings.ZSqrt2(a=1, b=0),
        rings.ZSqrt2(a=2, b=1),
    }


def bf(A, B, bounds):
    left, right = bounds
    for p in itertools.product(range(left, right + 1), repeat=2):
        x = rings.ZSqrt2(*p)
        if A.contains(float(x), mc.NumpyConfig) and B.contains(
            float(x.conjugate()), mc.NumpyConfig
        ):
            yield x


@pytest.mark.parametrize("w", 1 / (1 + mc.NumpyConfig.sqrt2) * np.random.default_rng(0).random(10))
def test_enumerate1d_requiring_scaling_up(w):
    A = lattice.Range(start=0, end=w)
    B = lattice.Range(start=0, end=10)
    assert set(bf(A, B, (-10, 10))) == set(lattice.enumerate_1d(A, B, mc.NumpyConfig))


@pytest.mark.parametrize("n", range(8))
def test_enumerate1d_against_brute_force(n):
    r0 = np.sqrt(2 * (2 + np.sqrt(2)) ** n)
    r1 = np.sqrt(2 / (2 + np.sqrt(2)) ** n)
    A, B = lattice.Range.from_bounds(-r0, r0), lattice.Range.from_bounds(-r1, r1)
    bf_sol = set(bf(A, B, (-int(r0) - 3, int(r0) + 3)))
    enum_sol = set(lattice.enumerate_1d(A, B, mc.NumpyConfig))
    assert bf_sol == enum_sol


def bf_rect(r1, r2, b):
    w = np.exp(1j * np.pi / 4)
    W = [w**i for i in range(4)]
    W_conj = W * np.array([1, -1, 1, -1])
    for m in itertools.product(range(math.floor(-b), math.ceil(b) + 1), repeat=4):
        v = np.dot(m, W)
        v_conj = np.dot(m, W_conj)
        if r1.contains(v.real, v.imag, mc.NumpyConfig) and r2.contains(
            v_conj.real, v_conj.imag, mc.NumpyConfig
        ):
            yield rings.ZW(m)


@pytest.mark.parametrize("n", range(4))
def test_enumerate_upright_against_bf(n):
    r0 = np.sqrt(2 * (2 + np.sqrt(2)) ** n)
    r1 = np.sqrt(2 * (2 - np.sqrt(2)) ** n)
    A, B = lattice.Range.from_bounds(-r0, r0), lattice.Range.from_bounds(-r1, r1)
    got = set(
        lattice.enumerate_upright(lattice.Rectangle(A, A), lattice.Rectangle(B, B), mc.NumpyConfig)
    )
    want = set(bf_rect(lattice.Rectangle(A, A), lattice.Rectangle(B, B), r0))
    assert got == want


@pytest.mark.parametrize("state", tu.make_states(5))
def test_get_overall_action(state: lattice.SelingerState):
    action = lattice.get_overall_action(state, mc.NumpyConfig)
    assert state.apply(action, mc.NumpyConfig).skew(mc.NumpyConfig) <= 15


@pytest.mark.parametrize("state", tu.make_states(5))
def test_get_points_from_state(state: lattice.SelingerState):
    for p in lattice.get_points_from_state(state, mc.NumpyConfig):
        assert state.contains(p, mc.NumpyConfig)
