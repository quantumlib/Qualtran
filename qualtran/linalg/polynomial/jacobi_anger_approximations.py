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
import bisect

import numpy as np
import scipy
import sympy
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt


def degree_jacobi_anger_approximation(t: SymbolicFloat, *, precision: SymbolicFloat) -> SymbolicInt:
    r"""Degree of the Jacobi-Anger expansion of $e^{it\sin(\theta)}$ or $e^{it\cos(\theta)}$.

    The Jacobi-Anger expansions are given by:

    $$
        C(e^{i\theta}) = e^{it\cos\theta} = \sum_{n = -\infty}^\infty i^n J_n(t) e^{in\theta}
        S(e^{i\theta}) = e^{it\sin\theta} = \sum_{n = -\infty}^\infty J_n(t) e^{in\theta}
    $$
    where $J_n$ is the $n$-th Bessel function of the first kind.

    We truncate the above series to the range $n \in [-d, d]$ such that $|J_{d+1}(t)| \le \epsilon$.

    If any parameter is symbolic, this returns an asymptotic result given by
    $$
        d = \mathcal{O}(t + \frac{\log(1/\epsilon)}{\log\log(1/\epsilon)})
    $$

    It returns `d` up to a symbolic constant. To ignore the constant, please use `big_O` on the final expression.

    Args:
        t: scale of the exponent in the function to approximate.
        precision: $\epsilon$ in the above polynomial approximation

    Returns:
        Truncation degree $d$ as defined above.
    """
    if is_symbolic(t, precision):
        # use a symbol for the constant.
        c_ja = sympy.Symbol("C_{JA}", positive=True)
        return c_ja * (t + sympy.log(1 / precision) / sympy.log(sympy.log(1 / precision)))

    def term_too_small(n: int) -> bool:
        return bool(np.isclose(scipy.special.jv(n, t), 0, atol=float(precision)))

    d = 1
    while not term_too_small(d):
        d *= 2

    # find the smallest `n` such that J_n(z) is too small
    d = bisect.bisect_left(range(d), True, key=term_too_small) - 1
    assert not term_too_small(d) and term_too_small(d + 1)
    return d


def approx_exp_cos_by_jacobi_anger(t: float, *, degree: int) -> NDArray[np.complex128]:
    r"""Laurent Polynomial approximation for $e^{i\theta} \mapsto e^{it\cos\theta}$.

    The approximation is given by
    $$
        e^{it\cos\theta} = \sum_{n = -\infty}^\infty i^n J_n(t) e^{in\theta}
    $$
    where $J_n$ is the $n$-th Bessel function of the first kind.

    For a given approximation degree $d$, we truncate it by restricting $n \in [-d, d]$.

    Args:
        t: scale in the exponent
        degree: value of $d$ to truncate the polynomial to $n \in [-d, d]$
    """
    coeff_indices = np.arange(-degree, degree + 1)
    return 1j**coeff_indices * scipy.special.jv(coeff_indices, t)


def approx_exp_sin_by_jacobi_anger(t: float, *, degree: int) -> NDArray[np.complex128]:
    r"""Laurent Polynomial approximation for $e^{i\theta} \mapsto e^{it\cos\theta}$.

    The approximation is given by
    $$
        e^{it\sin\theta} = \sum_{n = -\infty}^\infty J_n(t) e^{in\theta}
    $$
    where $J_n$ is the $n$-th Bessel function of the first kind.

    For a given approximation degree $d$, we truncate it by restricting $n \in [-d, d]$.

    Args:
        t: scale in the exponent
        degree: value of $d$ to truncate the polynomial to $n \in [-d, d]$
    """
    coeff_indices = np.arange(-degree, degree + 1)
    return scipy.special.jv(coeff_indices, t)
