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

These are for temporary convenience to lock-in the quoted literature costs.
"""
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import field, frozen

from qualtran import Bloq, BloqBuilder, QAny, QBit, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.mcmt import MultiControlZ
from qualtran.drawing import Circle, Text, TextBox, WireSymbol

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def qroam_cost(x, data_size: int, bitsize: int, adjoint: bool = False):
    # See appendix B of https://arxiv.org/pdf/1902.02134
    if adjoint:
        return data_size / x + x
    else:
        return data_size / x + bitsize * (x - 1)


def qroam_cost_dirty(x, data_size: int, bitsize: int, adjoint: bool = False):
    # See appendix A of https://arxiv.org/pdf/1902.02134
    if adjoint:
        return data_size / x + x
    else:
        return 2 * (data_size / x - 1) + 4 * bitsize * (x - 1)


def get_optimal_log_block_size_clean_ancilla(
    data_size: int, bitsize: int, adjoint: bool = False, qroam_block_size: Optional[int] = None
) -> int:
    if qroam_block_size is None:
        if adjoint:
            log_blk = 0.5 * np.log2(data_size)
            qroam_block_size = 2**log_blk
        else:
            log_blk = 0.5 * np.log2(data_size / bitsize)
            assert log_blk >= 0
            qroam_block_size = 2**log_blk
    k = np.log2(qroam_block_size)
    k_int = np.array([np.floor(k), np.ceil(k)])
    return int(k_int[np.argmin(qroam_cost(2**k_int, data_size, bitsize, adjoint))])


def get_qroam_cost_clean_ancilla(
    data_size: int, bitsize: int, adjoint: bool = False, qroam_block_size: Optional[int] = None
) -> int:
    """This gives the optimal k and minimum cost for a QROM over L values of size M.

    Adapted from openfermion and accounts for quoted inverse cost.

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.
        adjoint: whether to get costs from inverse qrom (true) or not (false).
        qroam_block_size: The block size for QROAM. Default find the optimal
            value given the data size.

    Returns:
       val_opt: minimal (optimal) cost of QROM
    """
    if qroam_block_size == 1:
        return data_size - 1
    k_opt = get_optimal_log_block_size_clean_ancilla(data_size, bitsize, adjoint, qroam_block_size)
    return int(np.ceil(qroam_cost(2**k_opt, data_size, bitsize, adjoint)))


@frozen
class QROAM(Bloq):
    """Placeholder bloq for QROAM with costs matching literature values.

    https://github.com/quantumlib/Qualtran/issues/368

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.
        adjoint: whether to get costs from inverse qrom (true) or not (false).
        qroam_blocking_factor: Block size for qroam. Default None (tries to find optimal block size)
    """

    data_size: int
    target_bitsize: int
    is_adjoint: bool = False
    qroam_block_size: Optional[int] = None

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(sel=(self.data_size - 1).bit_length(), trg=self.target_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cost = get_qroam_cost_clean_ancilla(
            self.data_size,
            self.target_bitsize,
            adjoint=self.is_adjoint,
            qroam_block_size=self.qroam_block_size,
        )
        return {Toffoli(): cost}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        dag = '†' if self.is_adjoint else ''
        return f"QROAM{dag}"


@frozen
class QROAMTwoRegs(Bloq):
    """Placeholder bloq for QROAM on two registers.

    Args:
        data_a_size: Amount of data we want to load from first index.
        data_b_size: Amount of data we want to load from second index.
        data_a_block_size: Blocking factor for first index.
        data_b_block_size: Blocking factor for second index.
        target_bitsize: the amount of bits of output we need.
        is_adjoint: whether to get costs from inverse qrom (true) or not (false).

    Returns:
       val_opt: minimal (optimal) cost of QROM

    References:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Appendix G, Eq. G3 and G6.
    """

    data_a_size: int
    data_b_size: int
    data_a_block_size: int
    data_b_block_size: int
    target_bitsize: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(sel=(self.data_a_size - 1).bit_length(), trg=self.target_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cost = int(np.ceil(self.data_a_size / self.data_a_block_size))
        cost *= int(np.ceil(self.data_b_size / self.data_b_block_size))
        if self.is_adjoint:
            cost += self.data_a_block_size * self.data_b_block_size
        else:
            cost += self.target_bitsize * (self.data_a_block_size * self.data_b_block_size - 1)
        return {Toffoli(): cost}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        dag = '†' if self.is_adjoint else ''
        return f"QROAM{dag}"


@frozen
class ApplyControlledZs(Bloq):
    """Apply controlled Z operation to a single qubit in a big system register.

    Used in THC / DF Select.

    This is either a CCZ or CCCZ operation. Wrap it as a bloq to hide the split / joins.

    Args:
        cvs: The control values of the controls.
        bitsize: The system bitsize.

    Registers:
        ctrls: control registers
        system: system register
    """

    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("ctrls", QBit(), shape=(len(self.cvs),)),
                Register("system", QAny(bitsize=self.bitsize)),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text("C" * len(self.cvs) + "Z")
        if reg.name == 'system':
            return TextBox('Z')

        (c_idx,) = idx
        filled = bool(self.cvs[c_idx])
        return Circle(filled)

    def build_composite_bloq(self, bb: 'BloqBuilder', ctrls: SoquetT, system: Soquet):
        split_sys = bb.split(system)
        ctrls, split_sys[0] = bb.add(MultiControlZ(self.cvs), controls=ctrls, target=split_sys[0])
        system = bb.join(split_sys)
        return {'ctrls': ctrls, 'system': system}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # remove this method once https://github.com/quantumlib/Qualtran/issues/528 is resolved.
        return {Toffoli(): len(self.cvs) - 1}
