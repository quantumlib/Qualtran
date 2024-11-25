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

from collections import Counter
from functools import cached_property
from typing import Dict, Tuple

from attrs import evolve, field, frozen, validators
from typing_extensions import Self

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import is_symbolic, prod, ssum, SymbolicFloat, SymbolicInt


@frozen
class TensorProduct(BlockEncoding):
    r"""Tensor product of a sequence of block encodings.

    Builds the block encoding as
    $$
        B[U_1 ⊗ U_2 ⊗ \cdots ⊗ U_n] = B[U_1] ⊗ B[U_2] ⊗ \cdots ⊗ B[U_n]
    $$

    When each $B[U_i]$ is a $(\alpha_i, a_i, \epsilon_i)$-block encoding of $U_i$, we have that
    $B[U_1 ⊗ \cdots ⊗ U_n]$ is a $(\prod_i \alpha_i, \sum_i a_i, \sum_i \alpha_i \epsilon_i)$-block
    encoding of $U_1 ⊗ \cdots ⊗ U_n$.

    Args:
        block_encodings: A sequence of block encodings.

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register (present only if bitsize > 0).

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    block_encodings: Tuple[BlockEncoding, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )

    @classmethod
    def of(cls, *block_encodings: BlockEncoding) -> Self:
        """Construct a `TensorProduct` from block encodings."""
        return cls(block_encodings)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),  # if ancilla_bitsize is 0, not present
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return ssum(u.system_bitsize for u in self.block_encodings)

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return prod(u.alpha for u in self.block_encodings)

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return ssum(u.ancilla_bitsize for u in self.block_encodings)

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return ssum(u.resource_bitsize for u in self.block_encodings)

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return ssum(u.alpha * u.epsilon for u in self.block_encodings)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        if all(isinstance(u.signal_state.prepare, PrepareIdentity) for u in self.block_encodings):
            return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))
        else:
            # TODO: implement by taking tensor product of component signal states
            raise NotImplementedError

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return Counter(self.block_encodings)

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        sys_regs = tuple(
            evolve(u.signature.get_left("system"), name=f"system{i}_")
            for i, u in enumerate(self.block_encodings)
        )
        anc_regs = tuple(
            evolve(u.signature.get_left("ancilla"), name=f"ancilla{i}_")
            for i, u in enumerate(self.block_encodings)
            if "ancilla" in u.signature._lefts
        )
        res_regs = tuple(
            evolve(u.signature.get_left("resource"), name=f"resource{i}_")
            for i, u in enumerate(self.block_encodings)
            if "resource" in u.signature._lefts
        )

        sys_part = Partition(self.system_bitsize, regs=sys_regs)
        sys_out_regs = list(bb.add_t(sys_part, x=system))
        if len(anc_regs) > 0:
            anc_part = Partition(self.ancilla_bitsize, regs=anc_regs)
            anc_out_regs = list(bb.add_t(anc_part, x=soqs["ancilla"]))
        if len(res_regs) > 0:
            res_part = Partition(self.resource_bitsize, regs=res_regs)
            res_out_regs = list(bb.add_t(res_part, x=soqs["resource"]))
        sys_i = 0
        anc_i = 0
        res_i = 0
        for u in self.block_encodings:
            u_soqs = dict()
            u_soqs["system"] = sys_out_regs[sys_i]
            if "ancilla" in u.signature._lefts:
                u_soqs["ancilla"] = anc_out_regs[anc_i]
            if "resource" in u.signature._lefts:
                u_soqs["resource"] = res_out_regs[res_i]
            u_soqs_out = bb.add_d(u, **u_soqs)
            sys_out_regs[sys_i] = u_soqs_out["system"]
            sys_i += 1
            if "ancilla" in u.signature._lefts:
                anc_out_regs[anc_i] = u_soqs_out["ancilla"]
                anc_i += 1
            if "resource" in u.signature._lefts:
                res_out_regs[res_i] = u_soqs_out["resource"]
                res_i += 1
        soqs_out = dict()
        soqs_out["system"] = bb.add(
            sys_part.adjoint(), **{r.name: sp for r, sp in zip(sys_regs, sys_out_regs)}
        )
        if len(anc_regs) > 0:
            soqs_out["ancilla"] = bb.add(
                anc_part.adjoint(), **{r.name: ap for r, ap in zip(anc_regs, anc_out_regs)}
            )
        if len(res_regs) > 0:
            soqs_out["resource"] = bb.add(
                res_part.adjoint(), **{r.name: ap for r, ap in zip(res_regs, res_out_regs)}
            )
        return soqs_out

    def __str__(self) -> str:
        return f"B[{'⊗'.join(str(u)[2:-1] for u in self.block_encodings)}]"


@bloq_example
def _tensor_product_block_encoding() -> TensorProduct:
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    tensor_product_block_encoding = TensorProduct((Unitary(TGate()), Unitary(Hadamard())))
    return tensor_product_block_encoding


@bloq_example
def _tensor_product_block_encoding_properties() -> TensorProduct:
    from attrs import evolve

    from qualtran.bloqs.basic_gates import CNOT, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    u1 = evolve(Unitary(TGate()), alpha=0.5, ancilla_bitsize=2, resource_bitsize=1, epsilon=0.01)
    u2 = evolve(Unitary(CNOT()), alpha=0.5, ancilla_bitsize=1, resource_bitsize=1, epsilon=0.1)
    tensor_product_block_encoding_properties = TensorProduct((u1, u2))
    return tensor_product_block_encoding_properties


@bloq_example
def _tensor_product_block_encoding_symb() -> TensorProduct:
    import sympy

    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    alpha1 = sympy.Symbol('alpha1')
    a1 = sympy.Symbol('a1')
    eps1 = sympy.Symbol('eps1')
    alpha2 = sympy.Symbol('alpha2')
    a2 = sympy.Symbol('a2')
    eps2 = sympy.Symbol('eps2')
    tensor_product_block_encoding_symb = TensorProduct(
        (
            Unitary(TGate(), alpha=alpha1, ancilla_bitsize=a1, epsilon=eps1),
            Unitary(Hadamard(), alpha=alpha2, ancilla_bitsize=a2, epsilon=eps2),
        )
    )
    return tensor_product_block_encoding_symb


_TENSOR_PRODUCT_DOC = BloqDocSpec(
    bloq_cls=TensorProduct,
    examples=[
        _tensor_product_block_encoding,
        _tensor_product_block_encoding_properties,
        _tensor_product_block_encoding_symb,
    ],
)
