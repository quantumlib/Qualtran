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
from typing import Sequence, Union

import numpy as np


def fft_complementary_polynomial(
    P: Union[Sequence[float], Sequence[complex]], tolerance: float = 1e-4, num_modes: int = 500
):
    """Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying
        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    Note that this function uses several private local functions to process intermediate calculations. Writing each
    step as a callable function prevents the output of these steps from being saved as a variable and thus
    reduces the memory used.

    Args:
          P: Co-efficients of a complex QSP polynomial
          num_modes: The number of modes used in the FFT operation. The more modes, the more accurate the result.
          tolerance: The maximum allowable amount that the sum of the squares can be off from the unit circle. This is
            mainly used for scaling and does not directly affect the speed of the calculation.

    Returns:
        The complementary polynomial, Q.

    References:
        [Complementary polynomials in quantum signal processing](https://arxiv.org/abs/2406.04246)
        Berntson and Sunderhauf. (2024). Figure 1.
    """
    P = np.array(P)
    N = num_modes

    def _scale(x):
        """Scale input according to tolerance."""
        return (1 - tolerance / 4) * x

    def _pad_poly(x):
        """Pad P to FFT dimension N"""
        return np.pad(_scale(x), (0, N - 1))

    def _p_eval(x):
        """Evaluate P(omega) at roots of unity omega"""
        return np.fft.ifft(_pad_poly(x), norm="forward")

    def _get_log(x):
        """Compute log(1-|P(omega)|^2) at roots of unity omega"""
        return np.log(1 - (np.abs(_p_eval(x))) ** 2)

    def _fourier_multiplier(the_log: np.ndarray, N: int) -> np.ndarray:
        """
        Applies the Fourier multiplier after applying the fft (Eq. 1.7)
        """
        modes = np.fft.fft(the_log, norm="forward")
        modes[0] *= 1 / 2  # Note modes are ordered differently in the text
        modes[N // 2 + 1 :] = 0
        return modes

    def _get_modes(x):
        """Apply Fourier multiplier in Fourier space"""
        return np.fft.ifft(_fourier_multiplier(_get_log(x), N), norm="forward")

    def calculate_coeff(poly: np.ndarray) -> np.ndarray:
        """Compute coefficients of Q

        Calculates the coefficients and truncates them to the proper degree.

        Args:
            poly: The input polynomial, P.
        Returns:
            The complementary polynomial, Q.
        """
        return np.fft.fft(np.exp(_get_modes(poly)), norm="forward")[: poly.shape[0]]

    return calculate_coeff(P)
