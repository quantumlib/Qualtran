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

import attrs

from qualtran import BloqBuilder, Signature, SoquetT
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat, SymbolicInt


@attrs.frozen
class QubitizedWalkOperator(BlockEncoding):
    r"""Construct a Szegedy Quantum Walk operator of a block encoding.

    Args:
        block_encoding: The input block-encoding.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    block_encoding: BlockEncoding

    @property
    def alpha(self) -> SymbolicFloat:
        return self.block_encoding.alpha

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.block_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return self.block_encoding.resource_bitsize

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.block_encoding.epsilon

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self.block_encoding.signal_state

    @cached_property
    def signature(self) -> Signature:
        return self.block_encoding.signature

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(self.block_encoding.signal_state, global_phase=-1)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        soqs |= bb.add_d(self.block_encoding, **soqs)
        soqs |= bb.add_d(
            self.reflect, **{reg.name: soqs[reg.name] for reg in self.reflect.signature}
        )
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {self.block_encoding: 1, self.reflect: 1}

    def __str__(self):
        return f'Walk[{self.block_encoding}]'
