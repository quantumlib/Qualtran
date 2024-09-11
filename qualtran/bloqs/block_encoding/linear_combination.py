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
from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np
from attrs import evolve, field, frozen, validators
from typing_extensions import Self

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    QAny,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.bloq import DecomposeTypeError
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import BlackBoxPrepare, BlackBoxSelect
from qualtran.bloqs.block_encoding.phase import Phase
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling
from qualtran.symbolics import smax, ssum, SymbolicFloat, SymbolicInt
from qualtran.symbolics.types import is_symbolic


@frozen
class LinearCombination(BlockEncoding):
    r"""Linear combination of a sequence of block encodings.

    Builds the block encoding $B[\lambda_1 U_1 + \lambda_2 U_2 + \cdots + \lambda_n U_n]$ given
    block encodings $B[U_1], \ldots, B[U_n]$ and coefficients $\lambda_i \in \mathbb{R}$.

    When each $B[U_i]$ is a $(\alpha_i, a_i, \epsilon_i)$-block encoding of $U_i$, we have that
    $B[\lambda_1 U_1 + \cdots + \lambda_n U_n]$ is a $(\alpha, a, \epsilon)$-block encoding
    of $\lambda_1 U_1 + \cdots + \lambda_n U_n$ where the normalization constant
    $\alpha = \sum_i \lvert\lambda_i\rvert\alpha_i$, number of ancillas
    $a = \lceil \log_2 n \rceil + \max_i a_i$, and precision
    $\epsilon = (\sum_i \lvert\lambda_i\rvert)\max_i \epsilon_i$.

    Under the hood, this bloq uses LCU Prepare and Select oracles to build the block encoding.
    These oracles will be automatically instantiated if not specified by the user.

    Args:
        block_encodings: A sequence of block encodings.
        lambd: Corresponding coefficients.
        lambd_bits: Number of bits needed to represent coefficients precisely.
        prepare: If specified, oracle preparing $\sum_i \sqrt{|\lambda_i|} |i\rangle$
            (state should be normalized and can have junk).
        select: If specified, oracle taking
            $|i\rangle|\psi\rangle \mapsto \text{sgn}(\lambda_i) |i\rangle U_i|\psi\rangle$.

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register (present only if bitsize > 0).

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](
        https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    _block_encodings: Tuple[BlockEncoding, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(2)
    )
    _lambd: Tuple[float, ...] = field(converter=lambda x: x if isinstance(x, tuple) else tuple(x))
    lambd_bits: SymbolicInt

    _prepare: Optional[BlackBoxPrepare] = None
    _select: Optional[BlackBoxSelect] = None

    def __attrs_post_init__(self):
        if len(self._block_encodings) != len(self._lambd):
            raise ValueError("Must provide the same number of block encodings and coefficients.")
        if sum(abs(x) for x in self._lambd) == 0:
            raise ValueError("Coefficients must not sum to zero.")
        if not all(be.system_bitsize == self.system_bitsize for be in self._block_encodings):
            raise ValueError("All block encodings must have the same dtype.")
        if (
            self._prepare is not None or self._select is not None
        ) and self.prepare.selection_registers != self.select.selection_registers:
            raise ValueError(
                "If given, prepare and select oracles must have same selection registers."
            )
        if self._select is not None and self._select.target_registers != (
            self.signature.get_left("system"),
        ):
            raise ValueError(
                "If given, select oracle must have block encoding `system` register as target."
            )
        if not all(
            isinstance(u.signal_state.prepare, PrepareIdentity) for u in self._block_encodings
        ):
            raise ValueError(
                "Cannot take linear combination of block encodings with non-zero signal state."
            )

    @classmethod
    def of_terms(cls, *terms: Tuple[float, BlockEncoding], lambd_bits: SymbolicInt = 1) -> Self:
        """Construct a `LinearCombination` from pairs of (coefficient, block encoding)."""
        return cls(tuple(t[1] for t in terms), tuple(t[0] for t in terms), lambd_bits)

    @cached_property
    def signed_block_encodings(self):
        """Appropriately negated constituent block encodings."""
        return tuple(
            Phase(be, phi=1, eps=0) if l < 0 else be
            for l, be in zip(self._lambd, self._block_encodings)
        )

    @cached_property
    def rescaled_lambd(self):
        """Rescaled and padded array of coefficients."""
        x = np.abs(np.array(self._lambd) * np.array([be.alpha for be in self._block_encodings]))
        x /= np.linalg.norm(x, ord=1)
        x.resize(2 ** int(np.ceil(np.log2(len(x)))), refcheck=False)
        return x

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),
            resource=QAny(self.resource_bitsize),
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.signed_block_encodings[0].system_bitsize

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return ssum(abs(l) * be.alpha for be, l in zip(self._block_encodings, self._lambd))

    @cached_property
    def be_ancilla_bitsize(self) -> SymbolicInt:
        return smax(be.ancilla_bitsize for be in self.signed_block_encodings)

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.be_ancilla_bitsize + self.prepare.selection_bitsize

    @cached_property
    def be_resource_bitsize(self) -> SymbolicInt:
        return smax(be.resource_bitsize for be in self.signed_block_encodings)

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.be_resource_bitsize + self.prepare.junk_bitsize

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return ssum(abs(l) for l in self.rescaled_lambd) * smax(
            be.epsilon for be in self.signed_block_encodings
        )

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @cached_property
    def prepare(self) -> BlackBoxPrepare:
        if self._prepare is not None:
            return self._prepare
        if is_symbolic(self.lambd_bits):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        alt, keep, mu = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=tuple(self.rescaled_lambd), sub_bit_precision=self.lambd_bits
        )
        N = len(self.rescaled_lambd)

        # import here to avoid circular dependency of StatePreparationAliasSampling
        # on PrepareOracle in qualtran.bloq.block_encoding
        from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
            StatePreparationAliasSampling,
        )

        # disable spurious pylint
        # pylint: disable=abstract-class-instantiated
        prep = StatePreparationAliasSampling(
            selection_registers=Register('selection', BQUInt((N - 1).bit_length(), N)),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            sum_of_unnormalized_probabilities=1,
        )
        return BlackBoxPrepare(prep)

    @cached_property
    def select(self) -> BlackBoxSelect:
        if self._select is not None:
            return self._select
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        assert not is_symbolic(self.be_ancilla_bitsize)
        assert not is_symbolic(self.be_resource_bitsize)

        # make all bloqs have same ancilla and resource registers
        bloqs = []
        for be in self.signed_block_encodings:
            assert not is_symbolic(be.ancilla_bitsize)
            assert not is_symbolic(be.resource_bitsize)

            partitions: List[Tuple[Register, List[Union[str, Unused]]]] = [
                (Register("system", QAny(self.system_bitsize)), ["system"])
            ]
            if self.be_ancilla_bitsize > 0:
                regs: List[Union[str, Unused]] = []
                if be.ancilla_bitsize > 0:
                    regs.append("ancilla")
                if self.be_ancilla_bitsize > be.ancilla_bitsize:
                    regs.append(Unused(self.be_ancilla_bitsize - be.ancilla_bitsize))
                partitions.append((Register("ancilla", QAny(self.be_ancilla_bitsize)), regs))
            if self.be_resource_bitsize > 0:
                regs = []
                if be.resource_bitsize > 0:
                    regs.append("resource")
                if self.be_resource_bitsize > be.resource_bitsize:
                    regs.append(Unused(self.be_resource_bitsize - be.resource_bitsize))
                partitions.append((Register("resource", QAny(self.be_resource_bitsize)), regs))
            bloqs.append(AutoPartition(be, partitions, left_only=False))

        # import here to avoid circular dependency of ApplyLthBloq
        # on SelectOracle in qualtran.bloqs.block_encoding
        from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

        return BlackBoxSelect(ApplyLthBloq(np.array(bloqs)))

    def build_composite_bloq(
        self, bb: BloqBuilder, system: Soquet, ancilla: Soquet, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        assert not is_symbolic(self.be_ancilla_bitsize)
        assert not is_symbolic(self.be_resource_bitsize)
        assert not is_symbolic(self.prepare.junk_bitsize)
        assert not is_symbolic(self.select.system_bitsize)

        # partition ancilla register
        be_system_soqs: Dict[str, SoquetT] = {"system": system}
        anc_regs = [Register("selection", QAny(self.prepare.selection_bitsize))]
        if self.be_ancilla_bitsize > 0:
            anc_regs.append(Register("ancilla", QAny(self.be_ancilla_bitsize)))
        anc_part = Partition(self.ancilla_bitsize, tuple(anc_regs))
        anc_soqs = bb.add_d(anc_part, x=ancilla)
        if self.be_ancilla_bitsize > 0:
            be_system_soqs["ancilla"] = anc_soqs.pop("ancilla")
        prepare_in_soqs = {"selection": anc_soqs.pop("selection")}

        # partition resource register if necessary
        if self.resource_bitsize > 0:
            res_regs = []
            if self.be_resource_bitsize > 0:
                res_regs.append(Register("resource", QAny(self.be_resource_bitsize)))
            if self.prepare.junk_bitsize > 0:
                res_regs.append(Register("prepare_junk", QAny(self.prepare.junk_bitsize)))
            res_part = Partition(self.resource_bitsize, tuple(res_regs))
            res_soqs = bb.add_d(res_part, x=soqs.pop("resource"))
            if self.be_resource_bitsize > 0:
                be_system_soqs["resource"] = res_soqs.pop("resource")
            if self.prepare.junk_bitsize > 0:
                prepare_in_soqs["junk"] = res_soqs.pop("prepare_junk")

        # merge system, ancilla, resource of block encoding into system register of Select oracle
        be_regs = [Register("system", QAny(self.system_bitsize))]
        if self.be_ancilla_bitsize > 0:
            be_regs.append(Register("ancilla", QAny(self.be_ancilla_bitsize)))
        if self.be_resource_bitsize > 0:
            be_regs.append(Register("resource", QAny(self.be_resource_bitsize)))
        be_part = Partition(self.select.system_bitsize, tuple(be_regs))

        prepare_soqs = bb.add_d(self.prepare, **prepare_in_soqs)
        select_out_soqs = bb.add_d(
            self.select,
            selection=prepare_soqs.pop("selection"),
            system=cast(Soquet, bb.add(evolve(be_part, partition=False), **be_system_soqs)),
        )
        prep_adj_soqs = bb.add_d(
            self.prepare.adjoint(), selection=select_out_soqs.pop("selection"), **prepare_soqs
        )

        # partition system register of Select into system, ancilla, resource of block encoding
        be_soqs = bb.add_d(be_part, x=select_out_soqs.pop("system"))
        out: Dict[str, SoquetT] = {"system": be_soqs.pop("system")}

        # merge ancilla registers of block encoding and Prepare oracle
        anc_soqs = {"selection": prep_adj_soqs.pop("selection")}
        if self.be_ancilla_bitsize > 0:
            anc_soqs["ancilla"] = be_soqs.pop("ancilla")
        out["ancilla"] = cast(Soquet, bb.add(evolve(anc_part, partition=False), **anc_soqs))

        # merge resource registers of block encoding and Prepare oracle
        if self.resource_bitsize > 0:
            res_soqs = dict()
            if self.be_resource_bitsize > 0:
                res_soqs["resource"] = be_soqs.pop("resource")
            if self.prepare.junk_bitsize > 0:
                res_soqs["prepare_junk"] = prep_adj_soqs.pop("junk")
            out["resource"] = cast(Soquet, bb.add(evolve(res_part, partition=False), **res_soqs))

        return out

    def __str__(self) -> str:
        return f"B[{'+'.join(str(be)[2:-1] for be in self.signed_block_encodings)}]"


@bloq_example
def _linear_combination_block_encoding() -> LinearCombination:
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    linear_combination_block_encoding = LinearCombination(
        (Unitary(TGate()), Unitary(Hadamard()), Unitary(XGate()), Unitary(ZGate())),
        lambd=(0.25, -0.25, 0.25, -0.25),
        lambd_bits=1,
    )
    return linear_combination_block_encoding


_LINEAR_COMBINATION_DOC = BloqDocSpec(
    bloq_cls=LinearCombination, examples=[_linear_combination_block_encoding]
)
