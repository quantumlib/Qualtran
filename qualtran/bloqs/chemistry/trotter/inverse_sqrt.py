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
from typing import Optional, Set, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator

from qualtran.bloqs.arithmetic import Add, MultiplyTwoReals, ScaleIntByReal, SquareRealNumber
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


@frozen
class NewtonRaphsonApproxInverseSquareRoot(Bloq):
    r"""Bloq implementing a single Newton-Raphson step to approximate the inverse square root.

    Given a (polynomial) approximation for $y_n = 1/sqrt{x}$ we can approximate
    the inverse square root by

    $$
        y_{n+1} = \frac{1}{2}y_n\left(3-y_n^2 x\right)
    $$

    For the case of computing the Coulomb potential we want

    $$
        \frac{1}{|r_i-r_j|} = \frac{1}{\sqrt{\sum_k^3 (x^{k}_i-x^{k})^2}}
    $$
    where $x^k_i \in \{x, y, z}$. Thus the input register should store $\sum_k^3
    (x^{k}_i-x^{k}_j)^2$.

    Args:
        x_sq_bitsize: The number of bits encoding the input (integer) register holding (x^2).
        poly_bitsize: The number of bits encoding the input (fp-real) register
            holding y0 (the output of PolynomialEvaluation).
        output_bitsize: The number of bits to store the output of the NewtonRaphson step.

    Register:
        x_sq: an input_bitsize size register storing the value x^2.
        poly: an poly_bitsize size register storing the value x^2.
        target: a target_bitsize size register storing the output of the newton raphson step.

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """
    x_sq_bitsize: int
    poly_bitsize: int
    target_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_sq', bitsize=self.x_sq_bitsize),
                Register('poly', bitsize=self.poly_bitsize),
                Register('target', self.target_bitsize),
            ]
        )

    def short_name(self) -> str:
        return 'y = x^{-1/2}'

    def t_complexity(self) -> 'TComplexity':
        return (
            SquareRealNumber(self.poly_bitsize).t_complexity()
            + ScaleIntByReal(self.x_sq_bitsize, self.poly_bitsize).t_complexity()
            + 2 * MultiplyTwoReals(self.target_bitsize).t_complexity()
            + Add(self.target_bitsize).t_complexity()
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # y * ((2 + b^2 + delta) + y^2 x)
        # 1. square y
        # 2. scale y^2 by x
        # 3. multiply y (2 + b^2 + delta)
        # 4. multiply y^2 x by y
        # 5. add 3. and 4.
        return {
            (1, SquareRealNumber(self.poly_bitsize)),
            (1, ScaleIntByReal(self.target_bitsize, self.x_sq_bitsize)),
            (2, MultiplyTwoReals(self.target_bitsize)),
            (1, Add(self.target_bitsize)),
        }


@frozen
class PolynmomialEvaluationInverseSquareRoot(Bloq):
    r"""Bloq to evaluate a polynomial approximation to inverse Square root from QROM.

    Args:
        in_bitsize: The number of bits encoding the input registers.
        out_bitsize: The number of bits encoding the input registers.

    Register:
        in_c{0,1,2,3}: QROM input containing the 4 polynomial coefficients.
     - out: Output register to store polynomial approximation to inverse square root.

    References:
        (Quantum computation of stopping power for inertial fusion target design
    )[https://arxiv.org/pdf/2308.12352.pdf]
    """
    x_sq_bitsize: int
    poly_bitsize: int
    out_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_sq', bitsize=self.x_sq_bitsize),
                Register('in_coeff', bitsize=self.poly_bitsize, shape=(4,)),
                Register('out', bitsize=self.out_bitsize),
            ]
        )

    def short_name(self) -> str:
        return 'y ~ x^{-1/2}'

    def t_complexity(self) -> 'TComplexity':
        # There are 3 multiplications and subtractions, the shifts (-1, -3/2)
        # are not included in Fusion estimates as these can be achieved with
        # Clifford gates only.
        return 3 * (
            Add(self.poly_bitsize).t_complexity()
            + MultiplyTwoReals(self.poly_bitsize).t_complexity()
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # This should probably be scale int by float rather than 3 real
        # multiplications as x in Eq. 49 of the reference is an integer.
        return {(3, MultiplyTwoReals(self.poly_bitsize)), (3, Add(self.poly_bitsize))}
