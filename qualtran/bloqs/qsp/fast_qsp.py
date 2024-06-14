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
from numpy.typing import NDArray
from scipy.optimize import minimize


class FastComplementaryQSPHelper:
    """
    A helper class to obtain the complimentary polynomial given a QSP polynomial.

    Attributes:
        P: Co-efficients of a complex QSP polynomial.
        only_reals: If `true`, then only real polynomial values will be returned.
    """

    def __init__(self, poly: NDArray, only_reals: bool = False):
        self.only_reals = only_reals
        if self.only_reals:
            assert poly.dtype == np.float64
            self.conv_p_negative = self.conv_by_flip_conj(poly) * -1
        else:
            assert poly.dtype == np.complex128
            self.conv_p_negative = self.complex_conv_by_flip_conj(poly.real, poly.imag) * -1
        self.conv_p_negative[poly.shape[0] - 1] = 1 - np.linalg.norm(poly) ** 2

    def loss_function(self, x: NDArray):
        if self.only_reals:
            conv_result = self.conv_by_flip_conj(x)
        else:
            real_part = x[: len(x) // 2]
            imag_part = x[len(x) // 2 :]
            conv_result = self.complex_conv_by_flip_conj(real_part, imag_part)

        # Compute loss using squared distance function
        loss = np.linalg.norm(self.conv_p_negative - conv_result) ** 2
        return loss

    @staticmethod
    def array_to_complex(x: NDArray) -> NDArray:
        """
        Converts a real array into a complex array.

        This method assumes that the real array is 1d and is twice the size of the desired complex array. The first half
        of the array is understood to be the real part, and the second half, the imaginary part.

        Args:
             A real nd array twice as long as the desired complex array
        Returns:
            A complex nd array built from the real array
        """
        real_part = x[: len(x) // 2]
        imag_part = x[len(x) // 2 :]
        return real_part + 1.0j * imag_part

    @staticmethod
    def conv_by_flip_conj(poly: NDArray) -> NDArray:
        return np.convolve(poly, np.flip(poly, axis=[0]), mode="full")

    @staticmethod
    def complex_conv_by_flip_conj(real_part: NDArray, imag_part: NDArray):
        """
        Performs the flip convolution.

        This method is used in sveral parts of the complementary polynomial
        calculation. Due to a limitation of the scipy optimizer, the
        input array must be split into its real and imaginary components first.
        """
        real_flip = np.flip(real_part, axis=[0])
        imag_flip = np.flip(-1 * imag_part, axis=[0])

        conv_real_part = np.convolve(real_part, real_flip, mode="full")
        conv_imag_part = np.convolve(imag_part, imag_flip, mode="full")

        conv_real_imag = np.convolve(real_part, imag_flip, mode="full")
        conv_imag_real = np.convolve(imag_part, real_flip, mode="full")

        # Compute real and imaginary part of the convolution
        real_conv = conv_real_part - conv_imag_part
        imag_conv = conv_real_imag + conv_imag_real

        # Combine to form the complex result
        return real_conv + 1j * imag_conv


def fast_complementary_polynomial(
    P: Union[Sequence[float], Sequence[complex]],
    random_state: np.random.RandomState,
    only_reals: bool = False,
    tolerance: float = 1e-10,
):
    """
    Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying

        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    using the flip convolution method described in Eq(60).  This
    an alternative for the complementary_polynomial in the
    generalized_qsp module.

    Note that by default, this method will take a complex input and
    return a complex output. If only real-valued results are desired,
    this must be explicitly set by setting "only_reals" to True.
    Since there are many possible complimentary polynomials given an
    input P, setting "only_reals" will run a slightly different method
    than the default to insure the complementary polynomial is real.
    This method, however, is significantly less accurate than
    the default method. If a real valued complementary polynomial
    is desired, it is recommended to use the complementary_polynomial
    method from the generalized_qsp module instead.

    Args:
        P: Co-efficients of a complex polynomial.
        random_state: The random state to use to generate the initial guess of the
            complementary polynomial Q.
        only_reals: If true, performs the calculation to only use and return real
            valued coefficients. Note that if this is set to "true", and P is
            complex, an error will be thrown.
        tolerance: The allowable tolerance for finding the minimum of the
            qsp loss function. In general, this number should be at least 1/10 of
            the desired tolerance used by the code that calls this method.

    References:
    [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
        Motlagh and Wiebe. (2023). Equation 60.
    """
    if only_reals:
        poly = np.array(P, dtype=np.float64)
        q_initial = random_state.randn(poly.shape[0])
    else:
        poly = np.array(P, dtype=np.complex128)
        q_initial = random_state.randn(poly.shape[0] * 2)
    q_initial_normalized = q_initial / np.linalg.norm(q_initial)

    qsp = FastComplementaryQSPHelper(poly, only_reals=only_reals)

    minimizer = minimize(qsp.loss_function, q_initial_normalized, jac="3-point", tol=tolerance)
    if only_reals:
        return minimizer.x

    return qsp.array_to_complex(minimizer.x)
