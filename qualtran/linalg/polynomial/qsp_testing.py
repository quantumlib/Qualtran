#  Copyright 2024 Google LLC
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
from typing import Sequence, Union

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran.symbolics import Shaped


def check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
    P: Union[Sequence[complex], Polynomial, Shaped],
    Q: Union[Sequence[complex], Polynomial, Shaped],
    *,
    random_state: np.random.RandomState,
    rtol: float = 1e-7,
    n_points: int = 1000,
):
    r"""Checks that two GQSP polynomials are consistent on random points.

    Given a pair of GQSP polynomials $P, Q$, this function checks that $|P(z)|^2 + |Q(z)|^2 = 1$
    for random $z$ on the complex unit circle (i.e. $|z| = 1$).
    """
    P = Polynomial(P)
    Q = Polynomial(Q)

    z = np.exp(random_state.random(size=n_points) * np.pi * 2j)
    np.testing.assert_allclose(np.abs(P(z)) ** 2 + np.abs(Q(z)) ** 2, 1, rtol=rtol)


def random_qsp_polynomial(
    degree: int, *, random_state: np.random.RandomState, only_real_coeffs: bool = False
) -> Sequence[complex]:
    r"""Generates a random complex polynomial $P$ s.t. $|P(e^{ix})| \le 1$ for every $x$.

    Args:
        degree: the degree of the generated polynomial
        random_state: np.random.RandomState
        only_real_coeffs: if True, generate polynomial with real coefficients.
    """
    poly = random_state.random(size=degree) / degree
    if not only_real_coeffs:
        poly = poly * np.exp(random_state.random(size=degree) * np.pi * 2j)
    return list(poly)


def _polynomial_max_abs_value_on_unit_circle(
    P: Union[NDArray[np.number], Sequence[complex], Shaped], *, n_points=2**17
):
    r"""Find the maximum absolute value of $P$ on $N$ uniform points on the complex unit circle.

    For a polynomial $P$, this function computes

    $$
        \max_{k = 0}^{N - 1} |P(e^{2 \pi i k/N})|
    $$

    TODO(#860) Figure out a more efficient and always correct way to do this.

    Args:
        P: complex polynomial
        n_points: number of points $N$ to evaluate
    """
    from scipy.fft import fft

    P = np.asarray(P)
    poly = np.zeros(n_points, dtype=P.dtype)
    poly[: len(P)] = P

    values = fft(poly)

    return np.max(np.abs(values))


def scale_down_to_qsp_polynomial(
    P: Sequence[complex], *, n_points: int = 2**17
) -> NDArray[np.complex128]:
    r"""Scale down the polynomial to be a valid QSP Polynomial

    $P$ is a QSP polynomial if $|P(e^{i\theta})| \le 1$ for every $\theta \in [0, 2\pi]$.

    If the input polynomial is not a valid QSP polynomial, this function attempts to compute
    the maximum absolute value on the unit circle, and scale it down by that factor.
    Otherwise returns the input as-is.

    Args:
        P: input polynomial to scale if needed
        n_points: number of points to sample on the unit circle to evaluate the polynomial
    """
    P = np.asarray(P)
    max_value = _polynomial_max_abs_value_on_unit_circle(list(P), n_points=n_points)
    if max_value > 1:
        P = P / max_value
    return P


def assert_is_qsp_polynomial(
    P: Union[NDArray[np.number], Sequence[complex], Shaped], *, n_points: int = 2**17
):
    r"""Check if the given polynomial is a valid QSP polynomial.

    $P$ is a QSP polynomial if $|P(e^{i\theta})| \le 1$ for every $\theta \in [0, 2\pi]$.
    """
    max_value = _polynomial_max_abs_value_on_unit_circle(P, n_points=n_points)
    assert (
        max_value <= 1
    ), f"Not a QSP polynomial! maximum absolute value {max_value} is greater than 1."
