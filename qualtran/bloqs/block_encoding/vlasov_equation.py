#  Copyright 2024 Google LLC
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

from typing import cast, Dict, Set

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    DecomposeTypeError,
    QFxp,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.controlled import CtrlSpec
from qualtran.bloqs.arithmetic.comparison import GreaterThan
from qualtran.bloqs.arithmetic.multiplication import InvertRealNumber
from qualtran.bloqs.arithmetic.trigonometric import ArcSin
from qualtran.bloqs.basic_gates import Ry, Swap, Toffoli
from qualtran.bloqs.basic_gates.x_basis import XGate
from qualtran.bloqs.block_encoding.sparse_matrix import EntryOracle, SparseMatrix
from qualtran.bloqs.mcmt import MultiControlPauli
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.rotations.phase_gradient import _fxp
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import ceil, HasLength, is_symbolic, SymbolicInt


@frozen
class InverseSquareRoot(Bloq):
    r"""Compute the inverse square root of a fixed-point number.

    Args:
        bitsize: Number of bits used to represent the number.
        num_frac: Number of fraction bits in the number.
        num_iters: Number of Newton-Raphson iterations.
    Registers:
        x: `bitsize`-sized input register.
        result: `bitsize`-sized output register.
    References:
        [Optimizing Quantum Circuits for Arithmetic](https://arxiv.org/abs/1805.12445). Appendix C.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    num_iters: SymbolicInt = 4  # reference studies 3, 4, or 5 iterations

    def __attrs_post_init__(self):
        if (
            not is_symbolic(self.num_frac)
            and not is_symbolic(self.bitsize)
            and self.num_frac > self.bitsize
        ):
            raise ValueError("num_frac must be < bitsize.")

    @property
    def signature(self):
        return Signature(
            [
                Register("x", QFxp(self.bitsize, self.num_frac)),
                Register("result", QFxp(self.bitsize, self.num_frac)),
            ]
        )

    def pretty_name(self) -> str:
        return "1/sqrt(x)"

    def on_classical_vals(
        self, x: ClassicalValT, result: ClassicalValT
    ) -> Dict[str, ClassicalValT]:
        if is_symbolic(self.bitsize):
            raise ValueError(f"Symbolic bitsize {self.bitsize} not supported")
        x_fxp: float = _fxp(x / 2**self.bitsize, self.bitsize).astype(float)
        result ^= int(1 / np.sqrt(x_fxp) * 2**self.bitsize)
        return {'x': x, 'result': result}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.bitsize
        p = self.bitsize - self.num_frac
        m = self.num_iters
        # directly copied from T_invsqrt on page 10 of reference
        ts = ceil(
            n**2 * (15 * m / 2 + 3)
            + 15 * n * p * m
            + n * (23 * m / 2 + 5)
            - 15 * p**2 * m
            + 15 * p * m
            - 2 * m
        )
        return {(Toffoli(), ts)}


@frozen
class VlasovEntryOracle(EntryOracle):
    r"""Oracle specifying the entries of the Hamiltonian for the Vlasov equation.

    Args:
        system_bitsize: The number of bits used to represent an index. The value of `M` in the
            referenced equation must be equal to `2**system_bitsize - 1`.
        entry_bitsize: The number of bits of precision to represent the arcsin of each entry.
        alpha: The physical parameter $\alpha$ in the referenced equation.

    Registers:
        q: The flag qubit that is rotated proportionally to the value of the entry.
        i: The row index.
        j: The column index.

    References:
        [A quantum algorithm for the linear Vlasov equation with collisions](https://arxiv.org/pdf/2303.03450) (2022). Eq. (27).
    """

    system_bitsize: SymbolicInt
    entry_bitsize: SymbolicInt
    alpha: float

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        mcx = MultiControlPauli(cvs=HasLength(self.system_bitsize), target_gate=cirq.X)
        gt = GreaterThan(self.system_bitsize, self.system_bitsize)
        num_frac = self.entry_bitsize - (self.system_bitsize - 1)
        inv = InvertRealNumber(QFxp(self.system_bitsize, 1), QFxp(self.entry_bitsize, num_frac))
        invsqrt = InverseSquareRoot(self.entry_bitsize, num_frac)
        arcsin = ArcSin(self.entry_bitsize, num_frac)
        swap = Swap(1).controlled(CtrlSpec(cvs=(1, 0)))
        x = XGate().controlled(CtrlSpec(cvs=(1, 0)))
        ry = Ry(2**-self.entry_bitsize).controlled()
        return {
            (mcx, 1),
            (And(), 1),
            (Ry(2 * np.arccos(np.sqrt((1 + self.alpha) / 2))).controlled(), 1),
            (gt, 2),
            (Swap(self.system_bitsize).controlled(), 1),
            (inv, 1),
            (invsqrt, 1),
            (arcsin, 1),
            (And(cv1=0, cv2=0), 1),
            (XGate(), 1),
            (swap, 2 * self.entry_bitsize),
            (x, self.entry_bitsize),
            (ry, self.entry_bitsize),
            (arcsin.adjoint(), 1),
            (invsqrt.adjoint(), 1),
            (inv.adjoint(), 1),
            (Swap(self.system_bitsize).controlled(), 1),
            (XGate(), 1),
            (And(cv1=0, cv2=0, uncompute=True), 1),
            (gt.adjoint(), 2),
            (And().adjoint(), 1),
            (mcx.adjoint(), 2),
        }

    def build_composite_bloq(
        self, bb: BloqBuilder, q: SoquetT, i: SoquetT, j: SoquetT
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.system_bitsize) or is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        mcx = MultiControlPauli(cvs=(0,) * self.system_bitsize, target_gate=cirq.X)
        i_bits, i_zero = bb.add(mcx, controls=bb.split(cast(Soquet, i)), target=bb.allocate(1))
        j_bits, j_zero = bb.add(mcx, controls=bb.split(cast(Soquet, j)), target=bb.allocate(1))
        i = bb.join(cast(NDArray, i_bits))
        j = bb.join(cast(NDArray, j_bits))
        i_zero_j_zero, i_or_j_zero = bb.add(And(), ctrl=np.array([i_zero, j_zero]))

        # case 1: i = 0 or j = 0, entry is sqrt((1 + alpha) / 2)
        i_or_j_zero, q = bb.add(
            Ry(2 * np.arccos(np.sqrt((1 + self.alpha) / 2))).controlled(), ctrl=i_or_j_zero, q=q
        )

        gt = GreaterThan(self.system_bitsize, self.system_bitsize)
        i_gt_j = bb.allocate(1)
        j_gt_i = bb.allocate(1)
        i, j, i_gt_j = bb.add(gt, a=i, b=j, target=i_gt_j)
        i, j, j_gt_i = bb.add(gt, a=j, b=i, target=j_gt_i)

        # case 2: i > j, entry is sqrt(i / 2)
        # swap i and j such that we fall into case 3
        i_gt_j, i, j = bb.add(Swap(self.system_bitsize).controlled(), ctrl=i_gt_j, x=i, y=j)

        # case 3: i < j, entry is sqrt(j / 2)
        num_frac = self.entry_bitsize - (self.system_bitsize - 1)
        entry0 = bb.allocate(dtype=QFxp(self.entry_bitsize, num_frac))
        # compute entry = 1 / (j / 2)
        inv = InvertRealNumber(QFxp(self.system_bitsize, 1), QFxp(self.entry_bitsize, num_frac))
        j, entry0 = bb.add(inv, a=j, result=entry0)
        # compute entry' = 1 / sqrt(1 / (j / 2)) = sqrt(j / 2)
        invsqrt = InverseSquareRoot(self.entry_bitsize, num_frac)
        entry1 = bb.allocate(dtype=QFxp(self.entry_bitsize, num_frac))
        entry0, entry1 = bb.add(invsqrt, x=entry0, result=entry1)
        # compute entry' = arcsin(entry)
        arcsin = ArcSin(self.entry_bitsize, num_frac)
        entry = bb.allocate(dtype=QFxp(self.entry_bitsize, num_frac))
        entry1, entry = bb.add(arcsin, x=entry1, result=entry)

        i_gt_j_j_gt_i, i_neq_j = bb.add(And(cv1=0, cv2=0), ctrl=np.array([i_gt_j, j_gt_i]))
        i_gt_j, j_gt_i = cast(NDArray, i_gt_j_j_gt_i)
        i_neq_j = cast(Soquet, bb.add(XGate(), q=i_neq_j))

        # if i_neq_j and not i_or_j_zero, rotate by entry
        entry_bits = bb.split(cast(Soquet, entry))
        zero = bb.allocate(1)
        ctrls = np.array([i_neq_j, i_or_j_zero])
        swap = Swap(1).controlled(CtrlSpec(cvs=(1, 0)))
        for k in range(len(entry_bits)):
            # selectively swap entry bit with 0 so we only rotate if i_neq_j and not i_or_j_zero
            ctrls, entry_bits[k], zero = bb.add(swap, ctrl=ctrls, x=entry_bits[k], y=zero)
            # flip q because entry is arcsin, not arccos
            ctrls, q = bb.add(XGate().controlled(CtrlSpec(cvs=(1, 0))), ctrl=ctrls, q=q)
            entry_bits[k], q = bb.add(Ry(2**-k).controlled(), ctrl=entry_bits[k], q=q)
            ctrls, entry_bits[k], zero = bb.add(swap, ctrl=ctrls, x=entry_bits[k], y=zero)
        i_neq_j, i_or_j_zero = cast(NDArray, ctrls)
        bb.free(cast(Soquet, zero))
        entry = bb.join(entry_bits)

        entry1, entry = bb.add(arcsin.adjoint(), x=entry1, result=entry)
        bb.free(entry)
        entry0, entry1 = bb.add(invsqrt.adjoint(), x=entry0, result=entry1)
        bb.free(entry1)
        j, entry0 = bb.add(inv.adjoint(), a=j, result=entry0)
        bb.free(cast(Soquet, entry0))

        i_gt_j, i, j = bb.add(Swap(self.system_bitsize).controlled(), ctrl=i_gt_j, x=i, y=j)

        i_neq_j = bb.add(XGate(), q=i_neq_j)
        i_gt_j, j_gt_i = cast(
            NDArray,
            bb.add(
                And(cv1=0, cv2=0, uncompute=True), ctrl=np.array([i_gt_j, j_gt_i]), target=i_neq_j
            ),
        )

        i, j, j_gt_i = bb.add(gt.adjoint(), a=j, b=i, target=j_gt_i)
        i, j, i_gt_j = bb.add(gt.adjoint(), a=i, b=j, target=i_gt_j)
        bb.free(cast(Soquet, i_gt_j))
        bb.free(cast(Soquet, j_gt_i))

        i_zero, j_zero = cast(
            NDArray, bb.add(And().adjoint(), ctrl=i_zero_j_zero, target=i_or_j_zero)
        )
        j_bits, j_zero = bb.add(mcx.adjoint(), controls=bb.split(cast(Soquet, j)), target=j_zero)
        i_bits, i_zero = bb.add(mcx.adjoint(), controls=bb.split(cast(Soquet, i)), target=i_zero)
        bb.free(cast(Soquet, j_zero))
        bb.free(cast(Soquet, i_zero))

        return {"q": q, "i": bb.join(cast(NDArray, i_bits)), "j": bb.join(cast(NDArray, j_bits))}


@bloq_example
def _vlasov_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import SymmetricBandedRowColumnOracle

    row_oracle = SymmetricBandedRowColumnOracle(3, bandsize=1)
    col_oracle = SymmetricBandedRowColumnOracle(3, bandsize=1)
    entry_oracle = VlasovEntryOracle(3, 7, alpha=0.2)
    symmetric_banded_matrix_block_encoding = SparseMatrix(
        row_oracle, col_oracle, entry_oracle, eps=0
    )
    return symmetric_banded_matrix_block_encoding
