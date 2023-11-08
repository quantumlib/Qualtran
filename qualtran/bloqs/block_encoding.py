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
"""High level bloqs for defining bloq encodings and operations on block encodings."""

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
class BlockEncoding(Bloq, metaclass=abc.ABCMeta):
    r"""Abstract base class that defines the API for a block encoding.

    Given an operator $V$ which can be expressed as a linear combination of unitaries

    $$
        V = \sum_l w_l U_l,
    $$
    where $w_l \ge 0$, $w_l \in \mathbb{R}$, then the block encoding $\mathcal{B}[V]$ satisifies
    $$
        a_\langle 0| \mathcal{B}[V] |0\rangle_a |\psi\rangle_s = V|\psi\rangle_s
    $$
    where the subscripts $a$ and $s$ signify ancilla and system registers respectively.

    Registers:
    """

    @property
    @abc.abstractmethod
    def junk_registers(self) -> infra.Registers:
        ...

    @property
    @abc.abstractmethod
    def system_registers(self) -> infra.Registers:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.junk_registers, *self.system_registers])


@frozen
class BlackBoxBlockEncoding(Bloq):
    """Standard block encoding using SELECT and PREPARE to a LCU"""

    select: Bloq
    prepare: Bloq
