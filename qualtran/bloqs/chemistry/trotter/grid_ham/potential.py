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
"""Bloqs for the Potential energy of a 3D grid based Hamiltonian."""

from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.data_types import BQUInt
from qualtran.bloqs.arithmetic import OutOfPlaceAdder, SumOfSquares
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
    build_qrom_data_for_poly_fit,
    get_inverse_square_root_poly_coeffs,
    NewtonRaphsonApproxInverseSquareRoot,
    PolynmomialEvaluationInverseSquareRoot,
)
from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.drawing import Text, WireSymbol


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
        system_i: The ith electron's register.
        system_j: The jth electron's register.

    References:
        [Faster quantum chemistry simulation on fault-tolerant quantum
            computers](https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta)
    """

    bitsize: int
    qrom_data: Tuple[Tuple[int], ...] = field(
        repr=False, converter=lambda d: tuple(tuple(x) for x in d)
    )
    poly_bitsize: int = 15
    inv_sqrt_bitsize: int = 24
    label: str = "V"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('system_i', dtype=QAny(self.bitsize), shape=(3,)),
                Register('system_j', dtype=QAny(self.bitsize), shape=(3,)),
            ]
        )

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'U_{self.label}(dt)_ij')
        return super().wire_symbol(reg, idx)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, system_i: SoquetT, system_j: SoquetT
    ) -> Dict[str, SoquetT]:
        if isinstance(system_i, Soquet) or isinstance(system_j, Soquet):
            raise ValueError("system_i and system_j must be numpy arrays of Soquet")
        # compute r_i - r_j
        # r_i + (-r_j), in practice we need to flip the sign bit, but this is just 3 cliffords.
        diff_ij = [0, 0, 0]
        for xyz in range(3):
            system_i[xyz], system_j[xyz], diff_ij[xyz] = bb.add(
                OutOfPlaceAdder(self.bitsize), a=system_i[xyz], b=system_j[xyz]
            )
        # Compute r_{ij}^2 = (x_i-x_j)^2 + ...
        bitsize_rij_sq = 2 * (self.bitsize + 1) + 2
        diff_ij, sos = bb.add(
            SumOfSquares(bitsize=self.bitsize + 1, k=3), input=np.asarray(diff_ij)
        )
        for xyz in range(3):
            system_i[xyz], system_j[xyz] = bb.add(
                OutOfPlaceAdder(self.bitsize, is_adjoint=True),
                a=system_i[xyz],
                b=system_j[xyz],
                c=diff_ij[xyz],
            )
        # Use rij^2 as the selection register for QROM to output a polynomial approximation to r_{ij}^{-1}.
        qrom_anc_c0 = bb.allocate(self.poly_bitsize)
        qrom_anc_c1 = bb.allocate(self.poly_bitsize)
        qrom_anc_c2 = bb.allocate(self.poly_bitsize)
        qrom_anc_c3 = bb.allocate(self.poly_bitsize)
        cast = Cast(
            inp_dtype=sos.reg.dtype,
            out_dtype=BQUInt(sos.reg.dtype.bitsize, iteration_length=len(self.qrom_data[0])),
        )
        sos = bb.add(cast, reg=sos)
        qrom_bloq = QROM(
            [np.array(d) for d in self.qrom_data],
            selection_bitsizes=(bitsize_rij_sq,),
            target_bitsizes=(self.poly_bitsize,) * 4,
        )
        sos, qrom_anc_c0, qrom_anc_c1, qrom_anc_c2, qrom_anc_c3 = bb.add(
            qrom_bloq,
            selection=sos,
            target0_=qrom_anc_c0,
            target1_=qrom_anc_c1,
            target2_=qrom_anc_c2,
            target3_=qrom_anc_c3,
        )
        sos = bb.add(cast.adjoint(), reg=sos)

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
        system: The system register of size eta * 3 * nb

    References:
        [Faster quantum chemistry simulation on fault-tolerant quantum
            computers](https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta)
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
                    dtype=QAny(((self.num_grid - 1).bit_length() + 1)),
                    shape=(self.num_elec, 3),
                )
            ]
        )

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'U_{self.label}(dt)')
        return super().wire_symbol(reg, idx)

    def build_composite_bloq(self, bb: BloqBuilder, *, system: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(system, Soquet):
            raise ValueError("system must be a numpy array of Soquet")
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


@bloq_example
def _pair_potential() -> PairPotential:
    bitsize = 7
    poly_bitsize = 15
    poly_coeffs = get_inverse_square_root_poly_coeffs()
    qrom_data = build_qrom_data_for_poly_fit(2 * bitsize + 2, poly_bitsize, poly_coeffs)
    qrom_data = tuple(tuple(int(k) for k in d) for d in qrom_data)
    pair_potential = PairPotential(bitsize=bitsize, qrom_data=qrom_data, poly_bitsize=poly_bitsize)
    return pair_potential


@bloq_example
def _potential_energy() -> PotentialEnergy:
    nelec = 12
    ngrid_x = 2 * 8 + 1
    potential_energy = PotentialEnergy(nelec, ngrid_x)
    return potential_energy


_POTENTIAL_ENERGY = BloqDocSpec(bloq_cls=PotentialEnergy, examples=(_potential_energy,))

_PAIR_POTENTIAL = BloqDocSpec(
    bloq_cls=PairPotential,
    import_line=(
        'from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential, build_qrom_data_for_poly_fit\n'
        'from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import get_inverse_square_root_poly_coeffs'
    ),
    examples=(_pair_potential,),
)
