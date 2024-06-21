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
from typing import Tuple

from qualtran import Bloq, BloqDocSpec, QDType, Register
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.symbolics import SymbolicFloat


class BlockEncoding(Bloq):
    r"""Abstract interface for an arbitrary block encoding.

    In general, given an $s$-qubit operator $H$ then the $(s+a)$-qubit unitary $B[H] = U$ is
    a $(\alpha, a, \epsilon)$-block encoding of $H$ if it satisfies:

    $$
        \lVert H - \alpha (\langle G|_a\otimes I_s U |G\rangle_a \otimes I_s) \rVert
        \le \epsilon,
    $$

    where $a$ is an ancilla register and $s$ is the system register, $U$ is a unitary sometimes
    called a signal oracle, $\alpha$ is a normalization constant chosen such
    that  $\alpha \ge \lVert H\rVert$ (where $\lVert \cdot \rVert$ denotes the
    spectral norm), and $\epsilon$ is the precision to which the block encoding
    is prepared. The state $|G\rangle_a$ is sometimes called the signal state,
    and its form depends on the details of the block encoding.

    For LCU based block encodings with $H = \sum_l w_l U_l$
    we have
    $$
    U = \sum_l |l\rangle\langle l| \otimes U_l
    $$
    and $|G\rangle = \sum_l \sqrt{\frac{w_l}{\alpha}}|l\rangle_a$, which define the
    usual SELECT and PREPARE oracles.

    Other ways of building block encodings exist so we define the abstract base
    class `BlockEncoding` bloq, which expects values for $\alpha$, $\epsilon$,
    system and ancilla registers and a bloq which prepares the state $|G\rangle$.

    Users must specify:
    1. the normalization constant $\alpha \ge \lVert A \rVert$, where
        $\lVert \cdot \rVert$ denotes the spectral norm.
    2. the precision to which the block encoding is to be prepared ($\epsilon$).

    Developers must provide a method to return a bloq to prepare $|G\rangle$.

    References:
        [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163/)
            Low et al. 2019. Sec 2 and 3 for introduction and definition of terms.

        [The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation](https://arxiv.org/abs/1804.01973)
            Chakraborty et al. 2018. Definition 3 page 8.

    """

    def pretty_name(self) -> str:
        return 'B[H]'

    @property
    def dtype(self) -> QDType:
        """The type of the system being block encoded."""
        raise NotImplementedError

    @property
    def alpha(self) -> SymbolicFloat:
        """The normalization constant."""
        raise NotImplementedError

    @property
    def num_ancillas(self) -> int:
        """The number of ancilla qubits."""
        raise NotImplementedError

    @property
    def num_resource(self) -> int:
        """The number of resource qubits not counted in ancillas."""
        raise NotImplementedError

    @property
    def epsilon(self) -> SymbolicFloat:
        """The precision to which the block encoding is to be prepared."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[Register, ...]:
        """The ancilla registers `a` above."""

    @property
    @abc.abstractmethod
    def junk_registers(self) -> Tuple[Register, ...]:
        """Any additional junk registers not included in selection registers."""

    @property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[Register, ...]:
        """The system registers of combined size `s`."""

    @property
    @abc.abstractmethod
    def signal_state(self) -> PrepareOracle:
        r"""Returns the signal / ancilla flag state $|G\rangle."""


_BLOCK_ENCODING_DOC = BloqDocSpec(
    bloq_cls=BlockEncoding,  # type: ignore[type-abstract]
    import_line="from qualtran.bloqs.block_encoding import BlockEncoding",
    examples=[],
)
