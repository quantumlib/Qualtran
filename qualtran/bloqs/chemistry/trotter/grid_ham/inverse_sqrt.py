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
"""Bloqs for computing the inverse Square root of a fixed point number."""
from functools import cached_property
from typing import Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, QFxp, QInt, Register, Signature
from qualtran.bloqs.arithmetic import Add, MultiplyTwoReals, ScaleIntByReal, SquareRealNumber

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def get_inverse_square_root_poly_coeffs() -> Tuple[NDArray, NDArray]:
    """Polynomial coefficients for approximating inverse square root.

    This function returns the coefficients of a piecewise cubic polynomial
    interpolation to the inverse square root, defined over two intervals [1, 3/2] (a) and
    [3/2, 2] (b). The coefficients were provided by the reference below in the
    context of computing the Coulomb potential.

    References:
        [Quantum computation of stopping power for inertial fusion target design]
        (https://arxiv.org/abs/2308.12352) pg. 12 / 13.
    """
    poly_coeffs_a = np.array(
        [
            0.99994132489119882162,
            0.49609891915903542303,
            0.33261112772430493331,
            0.14876762006038398086,
        ]
    )
    poly_coeffs_b = np.array(
        [
            0.81648515205385221995,
            0.27136515484240234115,
            0.12756148214815175348,
            0.044753028579153842218,
        ]
    )
    return poly_coeffs_a, poly_coeffs_b


def build_qrom_data_for_poly_fit(
    selection_bitsize: int, target_bitsize: int, poly_coeffs: Tuple[NDArray, NDArray]
) -> NDArray:
    """Build QROM data from polynomial coefficients from the referenence.

    Args:
        selection_bitsize: Number of bits for QROM selection register. This is
            determined in practice by the number of bits required to store
            r_{ij}^2.
        target_bitsize: Number of bits of precision for polynomial coefficients.
        poly_coeffs: Coefficients for piecewise polynomial approximation to
            inverse square root. These are provided by the function
            get_inverse_square_root_poly_coeffs.

    Returns:
        qrom_data: An array of integers representing the appropriately
            repeated scaled fixed-point representation of the polynomial
            coefficients required for the variable spaced QROM.

    References:
        [Quantum computation of stopping power for inertial fusion target design]
        (https://browse.arxiv.org/pdf/2308.12352.pdf) pg. 12.
    """
    poly_coeffs_a, poly_coeffs_b = poly_coeffs
    # We compute the inverse square root of x^2 using variable spaced QROM,
    # interpolation and newton-raphson Build data so QROM recognizes repeated
    # entries so as to use the variable spaced QROM implementation.  The
    # repeated ranges occur for l :-> l + 2^k, and are repeated twice for
    # coeffs_a and coeffs_b. We need to scale the coefficients by 2^{-(k-1)} to
    # correctly account for the selection range (r_{ij}^2). Our coefficients are
    # initially defined in the range [1, 3/2] for "_a" and [3/2, 2] for "_b".
    data = np.zeros((4, 2 ** (selection_bitsize)), dtype=np.int_)
    for i, (a, b) in enumerate(zip(poly_coeffs_a, poly_coeffs_b)):
        # In practice we should set x = 0 to some large constant, but we will just skip for now.
        # x = 1
        coeff = QFxp(target_bitsize, target_bitsize).to_fixed_width_int(a)
        data[i, 1] = coeff
        # x = 2
        coeff = QFxp(target_bitsize, target_bitsize).to_fixed_width_int(b)
        data[i, 2] = coeff
        # x = 3
        coeff = QFxp(target_bitsize, target_bitsize).to_fixed_width_int(a / 2 ** (1 / 2))
        data[i, 3] = coeff
        start = 4
        for k in range(2, selection_bitsize):
            coeff = QFxp(target_bitsize, target_bitsize).to_fixed_width_int(a / 2 ** (k / 2))
            # Number of time to repeat the data.
            data_size = max(1, 2 ** (k - 1))
            end = start + data_size
            data[i, start:end] = coeff
            coeff = QFxp(target_bitsize, target_bitsize).to_fixed_width_int(b / 2 ** (k / 2))
            start += data_size
            end += data_size
            data[i, start:end] = coeff
            start = end
    return data


@frozen
class NewtonRaphsonApproxInverseSquareRoot(Bloq):
    r"""Bloq implementing a single Newton-Raphson step to approximate the inverse square root.

    Given a (polynomial) approximation for $1/\sqrt{x}$ (which will be $y_0$)
    below we can approximate the inverse square root by

    $$
        y_{n+1} = \frac{1}{2}y_n\left(3-y_n^2 x\right)
    $$

    For the case of computing the Coulomb potential we want

    $$
        \frac{1}{|r_i-r_j|} = \frac{1}{\sqrt{\sum_k^3 (x^{k}_i-x^{k}_j)^2}}
    $$
    where $x^{k}_i$ is the $i$-th electron's coordinate in 3D and $k \in \{x,y,z\}$.
    Thus the input register should store $\sum_{k=x,y,z} (x^{k}_i-x^{k}_j)^2$.

    Args:
        x_sq_bitsize: The number of bits encoding the input (integer) register holding (x^2).
        poly_bitsize: The number of bits encoding the input (fp-real) register
            holding y0 (the output of PolynomialEvaluation).
        output_bitsize: The number of bits to store the output of the NewtonRaphson step.

    Registers:
        x_sq: an input_bitsize size register storing the value x^2.
        poly: an poly_bitsize size register storing the value x^2.
        target: a target_bitsize size register storing the output of the newton raphson step.

    References:
        [Faster quantum chemistry simulation on fault-tolerant quantum
            computers](https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta)
    """

    x_sq_bitsize: int
    poly_bitsize: int
    target_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_sq', QAny(bitsize=self.x_sq_bitsize)),
                Register('poly', QAny(bitsize=self.poly_bitsize)),
                Register('target', QAny(self.target_bitsize)),
            ]
        )

    def pretty_name(self) -> str:
        return 'y = x^{-1/2}'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # y * ((2 + b^2 + delta) + y^2 x)
        # 1. square y
        # 2. scale y^2 by x
        # 3. multiply y (2 + b^2 + delta)
        # 4. multiply y^2 x by y
        # 5. add 3. and 4.
        return {
            SquareRealNumber(self.poly_bitsize): 1,
            # TODO: When decomposing we will potentially need to cast into a larger register.
            # See: https://github.com/quantumlib/Qualtran/issues/655
            ScaleIntByReal(self.poly_bitsize, self.x_sq_bitsize): 1,
            MultiplyTwoReals(self.target_bitsize): 2,
            Add(QInt(self.target_bitsize)): 1,
        }


@frozen
class PolynmomialEvaluationInverseSquareRoot(Bloq):
    r"""Bloq to evaluate a polynomial approximation to inverse Square root from QROM.

    Args:
        in_bitsize: The number of bits encoding the input registers.
        out_bitsize: The number of bits encoding the input registers.

    Registers:
        in_c{0,1,2,3}: QROM input containing the 4 polynomial coefficients.
        out: Output register to store polynomial approximation to inverse square root.

    References:
        [Quantum computation of stopping power for inertial fusion target design](
            https://arxiv.org/pdf/2308.12352.pdf)
    """

    x_sq_bitsize: int
    poly_bitsize: int
    out_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_sq', QAny(bitsize=self.x_sq_bitsize)),
                Register('in_coeff', QAny(bitsize=self.poly_bitsize), shape=(4,)),
                Register('out', QAny(bitsize=self.out_bitsize)),
            ]
        )

    def pretty_name(self) -> str:
        return 'y ~ x^{-1/2}'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # This should probably be scale int by float rather than 3 real
        # multiplications as x in Eq. 49 of the reference is an integer.
        return {MultiplyTwoReals(self.poly_bitsize): 3, Add(QInt(self.poly_bitsize)): 3}


@bloq_example
def _nr_inv_sqrt() -> NewtonRaphsonApproxInverseSquareRoot:
    nr_inv_sqrt = NewtonRaphsonApproxInverseSquareRoot(7, 8, 12)
    return nr_inv_sqrt


@bloq_example
def _poly_inv_sqrt() -> PolynmomialEvaluationInverseSquareRoot:
    poly_inv_sqrt = PolynmomialEvaluationInverseSquareRoot(7, 8, 12)
    return poly_inv_sqrt


_NR_INV_SQRT = BloqDocSpec(
    bloq_cls=NewtonRaphsonApproxInverseSquareRoot,
    import_line='from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import NewtonRaphsonApproxInverseSquareRoot',
    examples=(_nr_inv_sqrt,),
)

_POLY_INV_SQRT = BloqDocSpec(
    bloq_cls=PolynmomialEvaluationInverseSquareRoot,
    import_line='from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import PolynmomialEvaluationInverseSquareRoot',
    examples=(_poly_inv_sqrt,),
)
