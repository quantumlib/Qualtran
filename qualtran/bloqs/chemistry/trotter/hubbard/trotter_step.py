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
r"""Bloqs implementing unitary evolution under the interacting part of the Hubbard Hamiltonian.

The approximate unitary is given by:

$$
U \approx e^{i \frac{t}{2} H_I} e^{i \frac{t}{2} H_h^p} e^{i t H_h^g}
          e^{i \frac{t}{2} H_h^p} e^{i \frac{t}{2} H_I}
$$

"""
from typing import Sequence

from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingTile, HoppingTileHWP
from qualtran.bloqs.chemistry.trotter.hubbard.interaction import Interaction, InteractionHWP
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary


def build_plaq_unitary_second_order_suzuki(
    length: int,
    hubb_u: float,
    timestep: float,
    hubb_t: float = 1.0,
    eps: float = 1e-9,
    strip_layer: bool = False,
) -> TrotterizedUnitary:
    """Build second order Suzuki-Trotter unitary for the square lattice Hubbard model.

    Args:
        length: box length
        hubb_u: Hubbard u.
        timestep: The time step for the unitary.
        hubb_t: Hubbard t. Default = 1.
        eps: The precision for single-qubit rotations.
        strip_layer: Whether to strip one application of the interaction term
            which is a common optimization if multiple trotter step are merged.

    Returns:
        unitary: The trotterized approximation to the unitary e^{-i t H}.
    """
    # Build the basic bloqs which make up the 2nd order PlAQ unitary.
    # The pink and gold "tiles".
    pink = HoppingTile(length=length, angle=0, eps=eps, pink=True, tau=hubb_t)
    gold = HoppingTile(length=length, angle=0, eps=eps, pink=False, tau=hubb_t)
    interaction = Interaction(length=length, angle=0, eps=eps, hubb_u=hubb_u)
    indices: Sequence[int] = ()
    coeffs: Sequence[float] = ()
    if strip_layer:
        # H_p H_g H_p H_I
        indices = (1, 2, 1, 0)
        coeffs = (0.5, 1, 0.5, 1)
    else:
        # Trotter splitting parameters when H = H_I + H_h^p + H_h^g
        indices = (0, 1, 2, 1, 0)
        coeffs = (0.5, 0.5, 1.0, 0.5, 0.5)
    unitary = TrotterizedUnitary(
        (interaction, pink, gold), indices=indices, coeffs=coeffs, timestep=timestep
    )
    return unitary


def build_plaq_hwp_unitary_second_order_suzuki(
    length: int,
    hubb_u: float,
    timestep: float,
    hubb_t: float = 1.0,
    eps: float = 1e-9,
    strip_layer: bool = False,
) -> TrotterizedUnitary:
    """Build second order Suzuki-Trotter unitary for the square lattice Hubbard model.

    This variant uses Hamming weight phasing for the rotations.

    Args:
        length: box length
        hubb_u: Hubbard u.
        timestep: The time step for the unitary.
        hubb_t: Hubbard t. Default = 1.
        eps: The precision for single-qubit rotations.
        strip_layer: Whether to strip one application of the interaction term
            which is a common optimization if multiple trotter step are merged.

    Returns:
        unitary: The trotterized approximation to the unitary e^{-i t H}.
    """
    # Build the basic bloqs which make up the 2nd order PlAQ unitary.
    # The pink and gold "tiles".
    pink = HoppingTileHWP(length=length, angle=0, eps=eps, pink=True, tau=hubb_t)
    gold = HoppingTileHWP(length=length, angle=0, eps=eps, pink=False, tau=hubb_t)
    interaction = InteractionHWP(length=length, angle=0, eps=eps, hubb_u=hubb_u)
    indices: Sequence[int] = ()
    coeffs: Sequence[float] = ()
    if strip_layer:
        # H_p H_g H_p H_I
        indices = (1, 2, 1, 0)
        coeffs = (0.5, 1, 0.5, 1)
    else:
        # Trotter splitting parameters when H = H_I + H_h^p + H_h^g
        indices = (0, 1, 2, 1, 0)
        coeffs = (0.5, 0.5, 1.0, 0.5, 0.5)
    unitary = TrotterizedUnitary(
        (interaction, pink, gold), indices=indices, coeffs=coeffs, timestep=timestep
    )
    return unitary
