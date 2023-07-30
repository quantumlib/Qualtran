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
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, Union

import sympy
from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import TGate
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class ContiguousRegister(Bloq):
    r"""Build contiguous register s from mu and nu.

    $$
        s = \nu (\nu - 1) / 2 + \mu
    $$

    Note the THC contiguous register is rather specific, as to save some space
    we write the one body data into the $|mu=M+1\rangle|nu\rangle$ register.

    Args:

    Registers

    References:
        (Even more efficient quantum computations of chemistry through
        tensor hypercontraction)[https://arxiv.org/pdf/2011.03494.pdf] Eq. 29.
    """

    bitsize: int
    s_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", bitsize=self.bitsize),
                Register("nu", bitsize=self.bitsize),
                Register("s", bitsize=self.s_bitsize),
            ]
        )

    def on_classical_vals(
        self, mu: 'ClassicalValT', nu: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'mu': mu, 'nu': nu, 's': nu * (nu - 1)}

    def bloq_counts(
        self, ssa: Optional['SympySymbolAllocator'] = None
    ) -> Set[Tuple[Union[int, sympy.Expr], Bloq]]:
        return {(self.bitsize**2 + self.bitsize - 1, TGate())}
