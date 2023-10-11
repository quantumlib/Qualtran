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

import abc
from functools import cached_property
from typing import Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.algos import MultiControlPauli

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop import CirqGateAsBloq

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class BlockEncoding(Bloq):
    r"""Abstract base class that defines the API for a Block encoding."""

    @property
    @abc.abstractmethod
    def control_registers(self) -> Iterable[Register]:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Iterable[Register]:
        ...

    @property
    @abc.abstractmethod
    def junk_registers(self) -> Iterable[Register]:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Iterable[Register]:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.junk_registers,
                *self.target_registers,
                *self.selection_registers,
            ]
        )


@frozen
class Reflection(Bloq):
    """Implement a reflection about zero using a "controled" multi-controlled pauli-Z gate.

    Args:
        bitsize: the number of bits to reflect about.
        num_controls: the number of (on) controls.

    Registers:
        refl: Registers to reflect about.
        ctrl: Control registers.
    """

    bitsize: int
    num_controls: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(refl=self.bitsize, ctrl=self.num_controls)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', refl: SoquetT, ctrl: SoquetT
    ) -> Dict[str, 'SoquetT']:
        """ """
        # Apply the Z to one of the on controls
        # need to merge controls and junk for reflection
        # assuming controls is a list of single qubit controls
        controls = bb.split(ctrl)
        if self.num_controls > 1:
            merged_controls_for_mcp = bb.join(np.concatenate([controls[1:], bb.split(refl)]))
        else:
            merged_controls_for_mcp = refl
        mcp = CirqGateAsBloq(
            MultiControlPauli(
                [1] * (self.num_controls - 1) + [0] * self.bitsize, target_gate=cirq.Z
            )
        )
        # apply the Z to the first control
        merged_controls_for_mcp, controls[0] = bb.add(
            mcp, controls=merged_controls_for_mcp, target=controls[0]
        )
        split_controls = bb.split(merged_controls_for_mcp)
        if len(controls) > 1:
            controls[1:] = split_controls[: self.num_controls - 1]
        refl = split_controls[self.num_controls - 1 :]
        return {"refl": bb.join(refl), "ctrl": bb.join(controls)}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        cvs = (1,) * (self.num_controls - 1) + (0,) * self.bitsize
        return {
            (1, MultiAnd(cvs)),
            # (n - 1, MultiAnd(cvs, adjoint=True)), Not implemented error.
            (1, ArbitraryClifford(n=2)),  # CZ gate
        }


@frozen
class BlockEncodeChebyshevPolynomial(Bloq):
    r"""Block encoding of T_j[H] where T_j is the jth Chebyshev polynomial.

    Here H is a Hamiltonian with spectral norm $|H| \le 1$ and we assume we have
    an M qubit ancilla register.

    We assume j > 0 to avoid block encoding the identity operator.

    Args:
        block_encoding: Block encoding of a Hamiltonian H, B[H]. Must specify
            system and junk registers.
        num_bits_ancilla: Number of ancilla qubits about which to reflect.
        num_bits_system: Number of system qubits.
        order: order of Chebychev polynomial.

    References:
        Page 45; Theorem 1.
        [Quantum computing enhanced computational catalysis]
        (https://arxiv.org/abs/2007.14460).
            Burg, Low et. al. 2021.
    """

    block_encoding: BlockEncoding
    order: int

    def pretty_name(self) -> str:
        return f"T_{self.order}[{self.block_encoding.pretty_name()}]"

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.block_encoding.control_registers,
                *self.block_encoding.selection_registers,
                *self.block_encoding.junk_registers,
                *self.block_encoding.target_registers,
            ]
        )

    def __attrs_post_init__(self):
        if self.order < 1:
            raise ValueError(f"order must be greater >= 1. Found {self.order}.")

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n = self.order
        num_junk = sum(
            [r.bitsize * int(np.prod(r.shape)) for r in self.block_encoding.junk_registers]
        )
        num_ctrl = len(self.block_encoding.control_registers)
        # there are n - 1 reflections and n B[H]'s
        return {
            (n - 1, Reflection(bitsize=num_junk, num_controls=num_ctrl)),
            (n, self.block_encoding),
        }
