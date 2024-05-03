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
"""Bloqs implementing unitary evolution under the one-body hopping Hamiltonian in 2D."""
from functools import cached_property
from typing import Set, TYPE_CHECKING, Union

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class HoppingPlaquette(Bloq):
    r"""A bloq implementing a single plaquette unitary.

    The bloq implements
    $$
        e^{i \kappa R_\mathrm{plaq}}
    $$
    where $\tau R^{k\sigma}_\mathrm{plaq} = H_h^{x(k,\sigma)}$, i.e. R is
    non-zero only in the subploq relevant for the particular indexed plaquette.

    The plaquette operator is given by
    $$
        \sum_{i,j} [R_{\mathrm{plaq}}]_{i,j} a_{i\sigma}^\dagger a_{j\sigma}
    $$
    where the non-zero sub-bloq of R_{\mathrm{plaq}} is

    $$
        R_{\mathrm{plaq}} = 
        \begin{bmatrix}
            0 & 1 & 0 & 1 \\
            1 & 0 & 1 & 0 \\
            0 & 1 & 0 & 1 \\
            1 & 0 & 1 & 0
        \end{bmatrix}
    $$

    Args:
        kappa: The scalar prefactor appearing in the definition of the unitary.
            Usually a combination of the timestep and the hopping parameter $\tau$. 
        eps: The precision of the single qubit rotations.

    Registers:
        qubits: A register of four qubits this unitary should act on.

    References:
        [Early fault-tolerant simulations of the Hubbard model](https://arxiv.org/abs/2012.09238)
        page 13 Eq. E4 and E5 (Appendix E)
    """

    kappa: Union[float, sympy.Expr]
    eps: Union[float, sympy.Expr] = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('qubits', QBit(), shape=(4,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # The TwoBitFFFT in the reference is F(k=0, n=arbitrary)
        # page 14, discussion after E13
        # There are 4 flanking f-gates and a e^{iXX}e^{iYY} rotation, which can
        # be rotated to single rotation + cliffords.
        return {(TwoBitFFFT(0, 1), 4), (Rz(self.kappa, eps=self.eps), 2)}


@frozen
class HoppingTile(Bloq):
    r"""Bloq implementing a "tile" of the one-body hopping unitary.

    Implements the unitary
    $$
    e^{i H_h^{x}} = \prod_{k\sigma} e^{i t H_h^{x(k,\sigma)}}
    $$
    for a particular choise of of plaquette hamiltonian $H_h^x$, where $x$ = pink or gold.

    Args:
        length: Lattice side length L.
        angle: The prefactor scaling the Hopping hamiltonian in the unitary (`t` above).
            This should contain any relevant prefactors including the time step
            and any splitting coefficients.
        tau: The Hopping hamiltonian parameter. Typically the hubbard model is
            defined relative to $\tau$ so it's defaulted to 1.
        eps: The precision of the single qubit rotations.
        pink: The colour of the plaquette.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](https://arxiv.org/abs/2012.09238)
        see Eq. 21 and App E.
    """

    length: Union[int, sympy.Expr]
    angle: Union[float, sympy.Expr]
    tau: float = 1.0
    eps: Union[float, sympy.Expr] = 1e-9
    pink: bool = True

    def __attrs_post_init__(self):
        if self.length % 2 != 0:
            raise ValueError('Only even length lattices are supported')

    def short_name(self) -> str:
        l = 'p' if self.pink else 'g'
        return f'H_h^{l}'

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Page 5, text after Eq. 22. There are L^2 / 4 plaquettes of a given colour and x2 for spin.
        return {
            (HoppingPlaquette(kappa=self.tau * self.angle, eps=self.eps), self.length**2 // 2)
        }


@bloq_example
def _hopping_tile() -> HoppingTile:
    length = 8
    angle = 0.5
    hopping_tile = HoppingTile(length, angle)
    return hopping_tile


_HOPPING_DOC = BloqDocSpec(
    bloq_cls=HoppingTile,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingTile',
    examples=(_hopping_tile,),
)


@bloq_example
def _plaquette() -> HoppingPlaquette:
    length = 8
    angle = 0.15
    plaquette = HoppingPlaquette(length, angle)
    return plaquette


_PLAQUETTE_DOC = BloqDocSpec(
    bloq_cls=HoppingPlaquette,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingPlaquette',
    examples=(_plaquette,),
)
