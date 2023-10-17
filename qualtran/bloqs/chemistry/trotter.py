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

from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen
from cirq_ft import TComplexity
from cirq_ft.algos.qrom import QROM
from cirq_ft.infra.bit_tools import float_as_fixed_width_int
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import (
    Add,
    MultiplyTwoReals,
    OutOfPlaceAdder,
    ScaleIntByReal,
    SquareRealNumber,
    SumOfSquares,
)
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates.rotation import RotationBloq
from qualtran.cirq_interop import CirqGateAsBloq

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


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
        _, coeff = float_as_fixed_width_int(a, target_bitsize)
        data[i, 1] = coeff
        # x = 2
        _, coeff = float_as_fixed_width_int(b, target_bitsize)
        data[i, 2] = coeff
        # x = 3
        _, coeff = float_as_fixed_width_int(a / 2 ** (1 / 2), target_bitsize)
        data[i, 3] = coeff
        start = 4
        for k in range(2, selection_bitsize):
            _, coeff = float_as_fixed_width_int(a / 2 ** (k / 2), target_bitsize)
            # Number of time to repeat the data.
            data_size = max(1, 2 ** (k - 1))
            end = start + data_size
            data[i, start:end] = coeff
            _, coeff = float_as_fixed_width_int(b / 2 ** (k / 2), target_bitsize)
            start += data_size
            end += data_size
            data[i, start:end] = coeff
            start = end
    return data


@frozen
class QuantumVariableRotation(Bloq):
    r"""Bloq implementing Quantum Variable Rotation

    $$
        \sum_j c_j|\phi_j\rangle \rightarrow \sum_j e^{i \xi \phi_j}  c_j | \phi_j\rangle
    $$

    This is the basic implementation in Fig. 14 of the reference.

    Args:
        bitsize: The number of bits encoding the phase angle $\phi_j$.

    Register:
        phi: a bitsize size register storing the angle $\phi_j$.

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
            Fig 14.
    """
    phi_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('phi', bitsize=self.phi_bitsize)])

    def short_name(self) -> str:
        return 'e^{i*phi}'

    def t_complexity(self) -> 'TComplexity':
        # Upper bounding for the moment with just phi_bitsize * Rz rotation gates.
        return self.phi_bitsize * Rz(0.0).t_complexity()

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        theta = ssa.new_symbol('theta')
        # need to update rotation bloq.
        return {(self.phi_bitsize, RotationBloq(theta))}


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


@frozen
class KineticEnergy(Bloq):
    """Bloq for the Kinetic energy unitary defined in the reference.

    Args:
        num_elec: The number of electrons.
        num_grid: The number of grid points in each of the x, y and z
            directions. In total, for a cubic grid, there are N = num_grid**3
            grid points. The number of bits required (in each spatial dimension)
            is thus log N + 1, where the + 1 is for the sign bit.

    Registers:
     - system: The system register of size eta * 3 * nb

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """

    num_elec: int
    num_grid: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register(
                    'system',
                    shape=(self.num_elec, 3),
                    bitsize=((self.num_grid - 1).bit_length() + 1),
                )
            ]
        )

    def short_name(self) -> str:
        return 'U_T(dt)'

    def build_composite_bloq(self, bb: BloqBuilder, *, system: SoquetT) -> Dict[str, SoquetT]:
        bitsize = (self.num_grid - 1).bit_length() + 1
        for i in range(self.num_elec):
            system[i], sos = bb.add(SumOfSquares(bitsize=bitsize, k=3), input=system[i])
            sos = bb.add(QuantumVariableRotation(phi_bitsize=(2 * bitsize + 2)), phi=sos)
            bb.free(sos)
        return {'system': system}


@frozen
class PairPotential(Bloq):
    """Potential Energy bloq for single pair of particles i and j.

    Args:
        bitsize: The number of bits for a single component of the system register.
        qrom_data: The polynomial coefficients to load by QROM.
        poly_bitsize: The number of bits of precision for the polynomial coefficients.
        label: A label for the bloqs short name. The potential bloq can encode
            any sort of Coulomb interaction (electron-electron, election-ion,
            ion-ion,...) so can be reused. This label is to distinguish these
            different cases.

    Registers:
     - system_i: The ith electron's register.
     - system_j: The jth electron's register.

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """

    bitsize: int
    qrom_data: Tuple[Tuple[int], ...] = field(repr=False)
    poly_bitsize: int = 15
    inv_sqrt_bitsize: int = 24
    label: str = "V"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('system_i', shape=(3,), bitsize=self.bitsize),
                Register('system_j', shape=(3,), bitsize=self.bitsize),
            ]
        )

    def pretty_name(self) -> str:
        return "PairPotential"

    def short_name(self) -> str:
        return f'U_{self.label}(dt)_ij'

    def build_composite_bloq(
        self, bb: BloqBuilder, *, system_i: SoquetT, system_j: SoquetT
    ) -> Dict[str, SoquetT]:
        # compute r_i - r_j
        # r_i + (-r_j), in practice we need to flip the sign bit, but this is just 3 cliffords.
        diff_ij = np.array([bb.allocate(self.bitsize) for _ in range(3)])
        for xyz in range(3):
            system_i[xyz], system_j[xyz], diff_ij[xyz] = bb.add(
                OutOfPlaceAdder(self.bitsize), a=system_i[xyz], b=system_j[xyz], c=diff_ij[xyz]
            )
        # Compute r_{ij}^2 = (x_i-x_j)^2 + ...
        bitsize_rij_sq = 2 * self.bitsize + 2
        diff_ij, sos = bb.add(SumOfSquares(bitsize=self.bitsize, k=3), input=diff_ij)
        # Use rij^2 as the selection register for QROM to output a polynomial approximation to r_{ij}^{-1}.
        qrom_anc_c0 = bb.allocate(self.poly_bitsize)
        qrom_anc_c1 = bb.allocate(self.poly_bitsize)
        qrom_anc_c2 = bb.allocate(self.poly_bitsize)
        qrom_anc_c3 = bb.allocate(self.poly_bitsize)
        qrom = QROM(
            [np.array(d) for d in self.qrom_data],
            selection_bitsizes=(bitsize_rij_sq,),
            target_bitsizes=(self.poly_bitsize,) * 4,
        )
        qrom_bloq = CirqGateAsBloq(qrom)
        sos, qrom_anc_c0, qrom_anc_c1, qrom_anc_c2, qrom_anc_c3 = bb.add(
            qrom_bloq,
            selection=sos,
            target0=qrom_anc_c0,
            target1=qrom_anc_c1,
            target2=qrom_anc_c2,
            target3=qrom_anc_c3,
        )

        # Compute the polynomial from the polynomial coefficients stored in QROM
        poly_out = bb.allocate(self.poly_bitsize)
        sos, [qrom_anc_c0, qrom_anc_c1, qrom_anc_c2, qrom_anc_c3], poly_out = bb.add(
            PolynmomialEvaluationInverseSquareRoot(
                x_sq_bitsize=bitsize_rij_sq,
                poly_bitsize=self.poly_bitsize,
                out_bitsize=self.poly_bitsize,
            ),
            x_sq=sos,
            in_coeff=np.array([qrom_anc_c0, qrom_anc_c1, qrom_anc_c2, qrom_anc_c3]),
            out=poly_out,
        )
        # Do a Newton-Raphson step to obtain a more accurate estimate of r_{ij}^{-1}
        inv_sqrt_sos = bb.allocate(self.inv_sqrt_bitsize)
        sos, poly_out, inv_sqrt_sos = bb.add(
            NewtonRaphsonApproxInverseSquareRoot(
                x_sq_bitsize=bitsize_rij_sq,
                poly_bitsize=self.poly_bitsize,
                target_bitsize=self.inv_sqrt_bitsize,
            ),
            x_sq=sos,
            poly=poly_out,
            target=inv_sqrt_sos,
        )
        inv_sqrt_sos = bb.add(
            QuantumVariableRotation(phi_bitsize=self.inv_sqrt_bitsize), phi=inv_sqrt_sos
        )
        bb.free(sos)
        bb.free(inv_sqrt_sos)
        bb.free(qrom_anc_c0)
        bb.free(qrom_anc_c1)
        bb.free(qrom_anc_c2)
        bb.free(qrom_anc_c3)
        bb.free(poly_out)
        for x in diff_ij:
            bb.free(x)
        return {'system_i': system_i, "system_j": system_j}


@frozen
class PotentialEnergy(Bloq):
    """Bloq for a Coulombic Unitary.

    This is a basic implementation which just iterates through num_elec *
    (num_elec - 1) electron pairs.

    Args:
        num_elec: The number of electrons.
        num_grid: The number of grid points in each of the x, y and z
            directions. In total, for a cubic grid there are N = num_grid**3
            grid points. The number of bits required (in each spatial dimension)
            is thus log N + 1, where the + 1 is for the sign bit.
        label: A label for the bloqs short name. The potential bloq can encode
            any sort of Coulomb interaction (electron-electron, election-ion,
            ion-ion,...) so can be reused. This label is to distinguish these
            different cases.

    Registers:
     - system: The system register of size eta * 3 * nb

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """

    num_elec: int
    num_grid: int
    poly_bitsize: int = 15
    label: str = "V"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register(
                    'system',
                    shape=(self.num_elec, 3),
                    bitsize=((self.num_grid - 1).bit_length() + 1),
                )
            ]
        )

    def pretty_name(self) -> str:
        return "PotentialEnergy"

    def short_name(self) -> str:
        return f'U_{self.label}(dt)'

    def build_composite_bloq(self, bb: BloqBuilder, *, system: SoquetT) -> Dict[str, SoquetT]:
        bitsize = (self.num_grid - 1).bit_length() + 1
        ij_pairs = np.triu_indices(self.num_elec, k=1)
        poly_coeffs = get_inverse_square_root_poly_coeffs()
        qrom_data = build_qrom_data_for_poly_fit(2 * bitsize + 2, self.poly_bitsize, poly_coeffs)
        # Make hashable
        qrom_data = tuple(tuple(int(k) for k in d) for d in qrom_data)
        for i, j in zip(*ij_pairs):
            system[i], system[j] = bb.add(
                PairPotential(bitsize, qrom_data, poly_bitsize=self.poly_bitsize, label=self.label),
                system_i=system[i],
                system_j=system[j],
            )
        return {'system': system}
