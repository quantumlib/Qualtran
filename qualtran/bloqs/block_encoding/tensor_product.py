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
from typing import Dict, Set, Tuple, TYPE_CHECKING

from attrs import evolve, field, frozen, validators

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.bookkeeping import Partition
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
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
        U: A sequence of block encodings.

    Registers:
        system: The system register.
        ancilla: The ancilla register.
        resource: The resource register.

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    block_encodings: Tuple[BlockEncoding, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),
            resource=QAny(self.resource_bitsize),
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return ssum(u.system_bitsize for u in self.block_encodings)

    def pretty_name(self) -> str:
        return f"B[{'⊗'.join(u.pretty_name()[2:-1] for u in self.block_encodings)}]"

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
    def target_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("system"),)

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("resource"),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),)

    @property
    def signal_state(self) -> PrepareOracle:
        """This method will be implemented in the future after PrepareOracle is updated for the BlockEncoding interface."""
        raise NotImplementedError

    def build_call_graph(self, ssa: SympySymbolAllocator) -> Set[BloqCountT]:
        return {(bloq, 1) for bloq in self.block_encodings}

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT, resource: SoquetT
    ) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        if TYPE_CHECKING:
            assert isinstance(self.system_bitsize, int)
            assert isinstance(self.ancilla_bitsize, int)
            assert isinstance(self.resource_bitsize, int)

        sys_regs, anc_regs, res_regs = zip(
            *(
                [evolve(r, name=f"{r.name}{i}_") for r in u.signature]
                for i, u in enumerate(self.block_encodings)
            )
        )
        sys_part = Partition(self.system_bitsize, regs=sys_regs)
        anc_part = Partition(self.ancilla_bitsize, regs=anc_regs)
        res_part = Partition(self.resource_bitsize, regs=res_regs)
        sys_out_regs = list(bb.add_t(sys_part, x=system))
        anc_out_regs = list(bb.add_t(anc_part, x=ancilla))
        res_out_regs = list(bb.add_t(res_part, x=resource))
        for i, u in enumerate(self.block_encodings):
            sys_out_regs[i], anc_out_regs[i], res_out_regs[i] = bb.add_t(
                u, system=sys_out_regs[i], ancilla=anc_out_regs[i], resource=res_out_regs[i]
            )
        system = bb.add(sys_part.adjoint(), **{r.name: sp for r, sp in zip(sys_regs, sys_out_regs)})
        ancilla = bb.add(
            anc_part.adjoint(), **{r.name: ap for r, ap in zip(anc_regs, anc_out_regs)}
        )
        resource = bb.add(
            res_part.adjoint(), **{r.name: ap for r, ap in zip(res_regs, res_out_regs)}
        )
        return {"system": system, "ancilla": ancilla, "resource": resource}


@bloq_example
def _tensor_product_block_encoding() -> TensorProduct:
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    tensor_product_block_encoding = TensorProduct((Unitary(TGate()), Unitary(Hadamard())))
    return tensor_product_block_encoding


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
    import_line="from qualtran.bloqs.block_encoding import TensorProduct",
    examples=[_tensor_product_block_encoding, _tensor_product_block_encoding_symb],
)
