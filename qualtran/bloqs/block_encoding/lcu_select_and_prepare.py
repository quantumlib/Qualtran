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
from typing import Optional, Tuple, TYPE_CHECKING

from qualtran import BloqDocSpec, GateWithRegisters, Register, Signature

if TYPE_CHECKING:
    from qualtran.symbolics import SymbolicFloat


class SelectOracle(GateWithRegisters):
    r"""Abstract base class that defines the interface for a SELECT Oracle.

    The action of a SELECT oracle on a selection register $|l\rangle$ and target register
    $|\Psi\rangle$ can be defined as:

    $$
        \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    In other words, the `SELECT` oracle applies $l$'th unitary $U_l$ on the target register
    $|\Psi\rangle$ when the selection register stores integer $l$.

    $$
        \mathrm{SELECT}|l\rangle |\Psi\rangle = |l\rangle U_{l}|\Psi\rangle
    $$
    """

    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[Register, ...]:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[Register, ...]:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[Register, ...]:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )


_SELECT_ORACLE_DOC = BloqDocSpec(
    bloq_cls=SelectOracle,
    import_line='from qualtran.bloqs.block_encoding.lcu_select_and_prepare import SelectOracle',
    examples=[],
)


class PrepareOracle(GateWithRegisters):
    r"""Abstract base class that defines the API for a PREPARE Oracle.

    Given a set of coefficients $\{c_0, c_1, ..., c_{N - 1}\}$, the PREPARE oracle is used to encode
    the coefficients as amplitudes of a state $|\Psi\rangle = \sum_{l=0}^{N-1} \sqrt{\frac{c_l}{\lambda}} |l\rangle$
    where $\lambda = \sum_l |c_l|$, using a selection register $|l\rangle$. In order to prepare such
    a state, the PREPARE circuit is also allowed to use a junk register that is entangled with
    selection register.

    Thus, the action of a PREPARE circuit on an input state $|0\rangle$ can be defined as:

    $$
        \mathrm{PREPARE} |0\rangle = \sum_{l=0}^{N-1} \sqrt{ \frac{c_l}{\lambda} } |l\rangle |\mathrm{junk}_l\rangle
    $$
    """

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[Register, ...]:
        ...

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return ()

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])

    @property
    def l1_norm_of_coeffs(self) -> Optional['SymbolicFloat']:
        r"""Sum of the absolute values of coefficients $c_i$.

        For LCU Hamiltonians, this is usually referred to as $\lambda$ in texts.
        """
        return None


_PREPARE_ORACLE_DOC = BloqDocSpec(
    bloq_cls=PrepareOracle,
    import_line='from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle',
    examples=[],
)
