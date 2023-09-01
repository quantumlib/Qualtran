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
from typing import Dict

import numpy as np
from attrs import frozen
from cirq_ft import TComplexity

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import OutOfPlaceAdder, SumOfSquares


@frozen
class QuantumVariableRotation(Bloq):
    r"""Bloq implementing Quantum Variable Rotation

    $$
        \sum_j c_j|\phi_j\rangle \rightarrow \sum_j e^{i \xi \phi_j}  c_j | \phi_j\rangle
    $$

    Args:
        bitsize: The number of bits encoding the phase angle $\phi_j$.

    Register:
     - phi: a bitsize size register storing the angle $\phi_j$.

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """
    phi_bitsize: int
    target_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('phi', bitsize=self.phi_bitsize),
                Register('target', bitsize=self.target_bitsize),
            ]
        )

    def short_name(self) -> str:
        return 'e^{i*phi}'


@frozen
class NewtonRaphson(Bloq):
    r"""Bloq implementing a single Newton-Raphson step

    Given a (polynomial) approximation for $y_n = 1/sqrt{x}$ we can approximate
    the inverse squareroot by

    $$
        y_{n+1} = \frac{1}{2}y_n\left(3-y_n^2 x\right)
    $$

    For the case of computing the coulomb potential we want

    $$
        \frac{1}{|r_i-r_j|} = \frac{1}{\sqrt{\sum_k^3 (x^{k}_i-x^{k})^2}}
    $$
    where $x^k_i \in \{x, y, z}$. Thus the input register should store $\sum_k^3
    (x^{k}_i-x^{k}_j)^2$.

    Args:
        bitsize: The number of bits encoding the input registers.

    Register:
     - x: a bitsize size register storing the value x^2.
     - y: a bitsize size register storing the output of the newton raphson step. Should approximate $1/

    References:
        (Faster quantum chemistry simulation on fault-tolerant quantum
            computers)[https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta]
    """
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', bitsize=self.bitsize), Register('y', bitsize=self.bitsize)])

    def short_name(self) -> str:
        return 'y = x^{-1/2}'

    # def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
    #     if ssa is None:
    #         raise ValueError(f"{self} requires a SympySymbolAllocator")
    #     k = ssa.new_symbol('k')
    #     return {(self.bitsize, CtrlModAddK(k=k, bitsize=self.bitsize, mod=self.mod))}

    def t_complexity(self) -> 'TComplexity':
        # Rough estimate from fusion paper, not sure where the 24 bits comes from
        return TComplexity(t=3 * self.bitsize**2 + n)


@frozen
class KineticEnergy(Bloq):
    """Bloq for Kinetic energy unitary.

    Args:
        num_elec: The number of electrons.
        num_grid: The number of grid points in each of the x, y and z
            directions. In total, for a cubic grid there are N = num_grid**3
            grid points. The number of bits required (in each spatial dimension)
            is thus log N + 1, where the + 1 is for the sign bit.

    Registers:
     -
     -

    References:
        (Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial
            Optimization)[https://arxiv.org/pdf/2007.07391.pdf].
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
        # TODO: discrepency here with target bitsize of 2*bitsize + 1 vs 2*bitsize + 2 listed in fusion paper.
        for i in range(self.num_elec):
            # temporary register to store output of sum of momenta
            sos = bb.allocate(2 * bitsize + 1)
            system[i], sos = bb.add(SumOfSquares(bitsize=bitsize, k=3), input=system[i], result=sos)
            sys_i = bb.join(np.array([bb.split(soq) for soq in system[i]]).ravel())
            sos, sys_i = bb.add(
                QuantumVariableRotation(phi_bitsize=(2 * bitsize + 1), target_bitsize=3 * bitsize),
                phi=sos,
                target=sys_i,
            )
            # Need to reshape this back into 3 registers of size bitsize
            system[i] = [bb.join(d) for d in np.array(bb.split(sys_i)).reshape(3, bitsize)]
            bb.free(sos)
        return {'system': system}


@frozen
class PairPotential(Bloq):
    """Bloq for single pair of particles i and j."""

    bitsize: int
    label: str = "V"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('system_i', shape=(3,), bitsize=self.bitsize),
                Register('system_j', shape=(3,), bitsize=self.bitsize),
            ]
        )

    def short_name(self) -> str:
        return f'{self.label}(dt)_ij'

    def build_composite_bloq(
        self, bb: BloqBuilder, *, system_i: SoquetT, system_j
    ) -> Dict[str, SoquetT]:
        # compute r_i - r_j
        # r_i + (-r_j), in practice we need to flip the sign bit, but this is just 3 cliffords.
        # We're potentially abusing this adder here with the result stored in register b.
        diff_ij = np.array([bb.allocate(self.bitsize) for _ in range(3)])
        for xyz in range(3):
            system_i[xyz], system_j[xyz], diff_ij[xyz] = bb.add(
                OutOfPlaceAdder(self.bitsize), a=system_i[xyz], b=system_j[xyz], c=diff_ij[xyz]
            )
        # temporary register to store output r_{ij}^2 = (x_i-x_j)^2 + ...
        sos = bb.allocate(2 * self.bitsize + 1)
        diff_ij, sos = bb.add(SumOfSquares(bitsize=self.bitsize, k=3), input=diff_ij, result=sos)
        # Compute inverse squareroot of x^2 using variable spaced QROM, interpolation and newton-raphson
        # Function interpolation QROM
        inv_sqrt_sos = bb.allocate(2 * self.bitsize + 1)
        sos, inv_sqrt_sos = bb.add(NewtonRaphson(2 * self.bitsize + 1), x=sos, y=inv_sqrt_sos)
        sys_ij = bb.join(
            np.array(
                [bb.split(soq) for soq in system_i] + [bb.split(soq) for soq in system_j]
            ).ravel()
        )
        inv_sqrt_sos, sys_ij = bb.add(
            QuantumVariableRotation(
                phi_bitsize=(2 * self.bitsize + 1), target_bitsize=6 * self.bitsize
            ),
            phi=inv_sqrt_sos,
            target=sys_ij,
        )
        # Need to reshape this back into 3 registers of size self.self.bitsize
        sys_ij = bb.split(sys_ij)
        # print([bb.join(d) for d in np.array(sys_ij[: 3 * self.self.bitsize]).reshape(3, self.self.bitsize)])
        system_i = [
            bb.join(d) for d in np.array(sys_ij[: 3 * self.bitsize]).reshape(3, self.bitsize)
        ]
        system_j = [
            bb.join(d) for d in np.array(sys_ij[3 * self.bitsize :]).reshape(3, self.bitsize)
        ]
        bb.free(sos)
        bb.free(inv_sqrt_sos)
        for x in diff_ij:
            bb.free(x)
        return {'system_i': system_i, "system_j": system_j}


@frozen
class PotentialEnergy(Bloq):
    """Bloq for a Coulombic Unitary.

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
     -
     -

    References:
        (Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial
            Optimization)[https://arxiv.org/pdf/2007.07391.pdf].
    """

    num_elec: int
    num_grid: int
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

    def short_name(self) -> str:
        return f'{self.label}(dt)'

    def build_composite_bloq(self, bb: BloqBuilder, *, system: SoquetT) -> Dict[str, SoquetT]:
        bitsize = (self.num_grid - 1).bit_length() + 1
        ij_pairs = np.triu_indices(self.num_elec, k=1)
        for i, j in zip(*ij_pairs):
            system[i], system[j] = bb.add(
                PairPotential(bitsize), system_i=system[i], system_j=system[j]
            )
        return {'system': system}
