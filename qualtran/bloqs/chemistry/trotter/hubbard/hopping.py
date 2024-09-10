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
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT
from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
from qualtran.symbolics import SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


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
    where the non-zero sub-bloq of $R_{\mathrm{plaq}}$ is

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

    kappa: SymbolicFloat
    eps: SymbolicFloat = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('qubits', QBit(), shape=(4,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # The TwoBitFFFT in the reference is F(k=0, n=arbitrary)
        # page 14, discussion after E13
        # There are 4 flanking f-gates and a e^{iXX}e^{iYY} rotation, which can
        # be rotated to single rotation + cliffords.
        return {TwoBitFFFT(0, 1, eps=self.eps): 4, Rz(self.kappa, eps=self.eps): 2}


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

    length: SymbolicInt
    angle: SymbolicFloat
    tau: float = 1.0
    eps: SymbolicFloat = 1e-9
    pink: bool = True

    def __attrs_post_init__(self):
        if isinstance(self.length, int) and self.length % 2 != 0:
            raise ValueError('Only even length lattices are supported')

    def pretty_name(self) -> str:
        l = 'p' if self.pink else 'g'
        return f'H_h^{l}'

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Page 5, text after Eq. 22. There are L^2 / 4 plaquettes of a given colour and x2 for spin.
        return {HoppingPlaquette(kappa=self.tau * self.angle, eps=self.eps): self.length**2 // 2}


@frozen
class HoppingTileHWP(HoppingTile):
    r"""Bloq implementing a "tile" of the one-body hopping unitary using Hamming weight phasing.

    Implements the unitary
    $$
    e^{i H_h^{x}} = \prod_{k\sigma} e^{i t H_h^{x(k,\sigma)}}
    $$
    for a particular choise of of plaquette hamiltonian $H_h^x$, where $x$ = pink or gold.

    Each plaquette Hamiltonian can be split into $L^2/4$ commuting terms. Each
    term can be implemented using the 4-qubit HoppingPlaquette above. The
    HoppingPlaquette bloq contains 2 arbitrary rotations which are flanked by Clifford operations.
    All of the rotations within a HoppingTile have the same angle so we can use
    HammingWeightPhaseing to reduce the number of T gates that need to be
    synthesized. Accounting for spin there are then $2 \times 2 \times L^2/4$
    arbitrary rotations in each Tile, but only  $L^2/2$ of them can be applied
    at the same time due to the $e^{iXX} e^{iYY}$ circuit not permitting parallel $R_z$ gates.

    Unlike in the HoppingTile implementation where we can neatly factor
    everything into sub-bloqs, here we would need to apply any clifford and F
    gates first in parallel then bulk apply the rotations in parallel using
    HammingWeightPhasing and then apply another layer of clifford and F gates.

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
        [Early fault-tolerant simulations of the Hubbard model](
            https://arxiv.org/abs/2012.09238) see Eq. 21 and App E.
    """

    def short_name(self) -> str:
        l = 'p' if self.pink else 'g'
        return f'H_h^{l}(HWP)'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Page 5, text after Eq. 22. There are L^2 / 4 plaquettes of a given colour and x2 for spin.
        # Each plaquette contributes 4 TwoBitFFFT gates and two arbitrary rotations.
        # We use Hamming weight phasing to apply all 2 * L^2/4 (two for spin
        # here) for both of these rotations.
        return {
            TwoBitFFFT(0, 1, self.eps): 4 * self.length**2 // 2,
            HammingWeightPhasing(2 * self.length**2 // 4, self.tau * self.angle, eps=self.eps): 2,
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


@bloq_example
def _hopping_tile_hwp() -> HoppingTileHWP:
    length = 8
    angle = 0.15
    hopping_tile_hwp = HoppingTileHWP(length, angle)
    return hopping_tile_hwp


_HOPPING_TILE_HWP_DOC = BloqDocSpec(
    bloq_cls=HoppingTileHWP,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingTileHWP',
    examples=(_hopping_tile_hwp,),
)
