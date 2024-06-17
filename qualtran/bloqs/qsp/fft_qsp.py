#  Copyright 2023 Google LLC
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
import math
from typing import Sequence, Union

import numpy as np


def _get_r(delta: float, d: int) -> float:
    return (1 / (1 - delta)) ** (1 / d)


def _get_N0(epsilon: float, delta: float, d: int) -> int:
    r = _get_r(delta, d)
    first_term = 2 / (np.log(r))

    second_term_num = 8 * np.log(1 / delta)
    second_term_den = epsilon * (r - 1)

    return math.ceil(first_term * np.log(second_term_num / second_term_den))


def _get_N(epsilon0: float, d: int) -> int:

    epsilon = epsilon0 / 4
    delta = epsilon0 / (5 * (d + 1))

    result = _get_N0(epsilon, delta, d)
    if result % 2 == 0:
        return result
    return result + 1


def _get_scale_factor(epsilon: float) -> float:
    return 1 - (epsilon / 4)


def _get_modes(the_log: np.ndarray, N: int) -> np.ndarray:
    modes = np.fft.fft(the_log, norm="forward")
    modes[0] *= 1 / 2  # Note modes are ordered differently in the text
    modes[N // 2 + 1 :] = 0
    return modes


def fft_complementary_polynomial(
    P: Union[Sequence[float], Sequence[complex]], tolerance: float = 1e-4
):
    """
    Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying
        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    Args:
          P: Co-efficients of a complex QSP polynomial
          tolerance: The maximum allowable amount that the sum of the squares can be off from the unit circle. Note that
            a high tolerance will require more memory and computation time. This scales roughly as O(1/tolerance).
    Returns:
        The complementary polynomial, Q.

    References:
    [Complementary polynomials in quantum signal processing](https://arxiv.org/abs/2406.04246)
        Berntson and Sunderhauf. (2024). Figure 1.
    """
    # Scale P
    P = np.array(P)
    scaled_P = (1 - tolerance / 4) * P

    d = P.shape[0]
    N = _get_N(tolerance, d)

    # Pad P to FFT dimension N
    padded_poly = lambda x: np.pad(scaled_P, (0, N - 1))
    # Evaluate P(omega) at roots of unity omega
    p_eval = lambda x: np.fft.ifft(padded_poly(x), norm="forward")
    # Compute log(1-|P(omega)|^2) at roots of unity omega
    the_log = lambda x: np.log(1 - (np.abs(p_eval(x))) ** 2)
    # Apply Fourier multiplier in Fourier space
    modes = lambda x: np.fft.ifft(_get_modes(the_log(x), N), norm="forward")
    # Compute coefficients of Q
    calculate_coeff = lambda x: np.fft.fft(np.exp(modes(x)), norm="forward")[: P.shape[0]]

    result = calculate_coeff(scaled_P)
    return result
