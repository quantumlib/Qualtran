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
from functools import cached_property
from typing import Iterable, Sequence, Union

import attrs
import cirq
import numpy as np

from qualtran import GateWithRegisters, QFxp, Signature
from qualtran.bloqs.arithmetic import PlusEqualProduct
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import is_symbolic, SymbolicInt


@attrs.frozen
class ArcTan(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    r"""Applies U|x>|0>|0000...0> = |x>|sign>|abs(-2 arctan(x) / pi)>.

    This is computed by a series of fused-multiply-add (`PlusEqualProduct`) instructions.
    See Ref. 2 for classical implementations, and Ref. 3 for quantum multiplication circuits.

    Args:
        selection_bitsize: The bitsize of input register |x>.
        target_bitsize: The bitsize of output register. The computed quantity,
            $\abs(-2 * \arctan(x) / \pi)$ is stored as a fixed-length binary approximation
            in the output register of size `target_bitsize`.

    References:
        [Mean estimation when you have the source code; or, quantum Monte Carlo methods](https://arxiv.org/abs/2208.07544)
        Kothari and Donnell, 2022. Appendix A.

        [StackOverflow: FPATAN](https://stackoverflow.com/questions/23047978/how-is-arctan-implemented/23097989#23097989)
        Accessed 29.07.2022.

        [Fast quantum integer multiplication with zero ancillas](https://arxiv.org/abs/2403.18006)
        Kahanamoku-Meyer and Yao, 2024.
    """

    selection_bitsize: SymbolicInt
    target_bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(select=self.selection_bitsize, sign=1, target=self.target_bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        if is_symbolic(self.selection_bitsize) or is_symbolic(self.target_bitsize):
            raise TypeError(
                "Cannot build registers for symbolic bitsizes "
                f"{self.selection_bitsize=} {self.target_bitsize=}"
            )
        return (2,) * self.selection_bitsize, (2,), (2,) * self.target_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "ArcTan":
        raise NotImplementedError()

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        input_val, target_sign, target_val = register_values
        output_val = -2 * np.arctan(input_val, dtype=np.double) / np.pi
        assert -1 <= output_val <= 1
        output_sign = int(output_val < 0)
        output_bin = QFxp(self.target_bitsize, self.target_bitsize).to_fixed_width_int(
            abs(output_val)
        )
        return input_val, target_sign ^ output_sign, target_val ^ output_bin

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        # Hack to propagate bigO(...). Here `c` is some constant.
        c = ssa.new_symbol("c")

        return {
            PlusEqualProduct(self.target_bitsize, self.target_bitsize, 2 * self.target_bitsize): c
        }

    def adjoint(self) -> 'ArcTan':
        return self
