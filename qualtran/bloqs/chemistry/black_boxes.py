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
"""Common Chemistry bloqs which have costs that differ from those currently implemented in qualtran.

These are for temporary convenience to lock-in the quoted literature costs
before they can be safely replaced by actual implementations. 
"""
from functools import cached_property
from typing import Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
from sympy import factorint

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareUniformSuperposition(Bloq):
    r"""Prepare a uniform superposition over $d$ basis states.

    Uses quoted literature costs which relies on phase gradient for rotations.

    Args:
        d: The number of coefficients to prepare.
        num_bits_rot_aa: The number of bits of precision for the single-qubit
            rotation for amplitude amplification during the uniform state
            preparataion. Default 8.

    Registers:
        d: the register to prepare the uniform superposition on.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Page 39.
    """
    d: int
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('d', (self.num_non_zero - 1).bit_length())]
        return Signature(regs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        factors = factorint(self.num_non_zero)
        eta = factors[min(list(sorted(factors.keys())))]
        if self.d % 2 == 1:
            eta = 0
        uniform_prep = 3 * (self.d - 1).bit_length() - 3 * eta + 2 * self.num_bits_rot_aa - 9
        return {(Toffoli(), uniform_prep)}


def get_qroam_cost(data_size: int, bitsize: int, adjoint: bool = False) -> Tuple[int, int]:
    """This gives the optimal k and minimum cost for a QROM over L values of size M.

    Adapted from openfermion and accounts for quoted inverse cost.

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.
        adjoint: whether to get costs from inverse qrom (true) or not (false).

    Returns:
       val_opt: minimal (optimal) cost of QROM
    """
    if adjoint:
        k = 0.5 * np.log2(data_size)
        value = lambda k: data_size / 2**k + 2**k
    else:
        k = 0.5 * np.log2(data_size / bitsize)
        assert k >= 0
        value = lambda k: data_size / 2**k + bitsize * (2**k - 1)
    k_int = np.array([np.floor(k), np.ceil(k)])
    k_opt = k_int[np.argmin(value(k_int))]
    val_opt = np.ceil(value(k_opt))
    return int(val_opt)


@frozen
class QROAM(Bloq):
    """Placeholder bloq for QROAM with costs matching literature values.

    https://github.com/quantumlib/Qualtran/issues/368

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.
        adjoint: whether to get costs from inverse qrom (true) or not (false).

    Returns:
       val_opt: minimal (optimal) cost of QROM
    """

    data_size: int
    selection_bitsize: int
    target_bitsize: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"QROAM{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(sel=self.data_size, trg=self.target_bitsize)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        cost = get_qroam_cost(self.data_size, self.target_bitsize, adjoint=self.adjoint)
        return {(Toffoli(), cost)}
