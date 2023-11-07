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
r"""Bloqs for SELECT T for the first quantized chemistry Hamiltonian with a quantum projectile."""
from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SelectTFirstQuantizationWithProj(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_n: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.

    Registers:
        flag_T: a flag to control on the success of the $T$ state preparation.
        flag_mean: a flag for whether to select the mean part of the
            projectile's Hamiltonian or not.
        plus: A $|+\rangle$ state.
        w: A register for selecting x, y and z components of the momentum register.
        w_mean: A register for selecting x, y and z components of the momentum
            register. This is for the mean part of the projectile's kinetic energy.
        r: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        s: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        p: An ancilla register which should contained the (swapped) electron's
            state OR the projectiles state.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 20, section B
    """
    num_bits_n: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("flag_T", bitsize=1),
                Register("flag_mean", bitsize=1),
                Register("plus", bitsize=1),
                Register("w", bitsize=3),
                Register("w_mean", bitsize=3),
                Register("r", bitsize=self.num_bits_n),
                Register("s", bitsize=self.num_bits_n),
                Register("p", bitsize=self.num_bits_n, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'SEL $T$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Modification of the SEL T costs from the first quantized bloq with n_p replace with n_n.
        # The + 1 is from an additional Toffoli for the selection between the
        # square and the product of the momentum offset of the projectile.
        return {(Toffoli(), (5 * (self.num_bits_n - 1) + 2 + 1))}


@bloq_example
def _sel_t_proj() -> SelectTFirstQuantizationWithProj:
    num_bits_n = 8
    eta = 32
    sel_t_proj = SelectTFirstQuantizationWithProj(num_bits_n, eta)
    return sel_t_proj
