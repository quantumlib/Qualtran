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

from functools import cached_property, reduce
from typing import Dict, Sequence, Tuple

from attrs import evolve, field, frozen

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QDType,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.bookkeeping import Partition
from qualtran.symbolics import SymbolicFloat


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

    U: Sequence[BlockEncoding] = field(converter=lambda x: x if isinstance(x, tuple) else tuple(x))

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=self.dtype, ancilla=QAny(self.num_ancillas), resource=QAny(self.num_resource)
        )

    @cached_property
    def dtype(self) -> QDType:
        return QAny(bitsize=sum(u.dtype.num_qubits for u in self.U))

    def pretty_name(self) -> str:
        return f"B[{'⊗'.join(u.pretty_name()[2:-1] for u in self.U)}]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return reduce(lambda a, b: a * b.alpha, self.U, 1.0)

    @cached_property
    def num_ancillas(self) -> int:
        return sum(u.num_ancillas for u in self.U)

    @cached_property
    def num_resource(self) -> int:
        return sum(u.num_resource for u in self.U)

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return sum(u.alpha * u.epsilon for u in self.U)

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
        raise NotImplementedError

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT, resource: SoquetT
    ) -> Dict[str, SoquetT]:
        transpose = lambda x: zip(*x)
        sys_regs, anc_regs, res_regs = transpose(
            [evolve(r, name=f"{r.name}{i}") for r in u.signature.lefts()]
            for i, u in enumerate(self.U)
        )
        sys_part = Partition(self.dtype.num_qubits, regs=sys_regs)
        anc_part = Partition(self.num_ancillas, regs=anc_regs)
        res_part = Partition(self.num_resource, regs=res_regs)
        sys_out_regs = list(bb.add_t(sys_part, x=system))
        anc_out_regs = list(bb.add_t(anc_part, x=ancilla))
        res_out_regs = list(bb.add_t(res_part, x=resource))
        for i, u in enumerate(self.U):
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
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    tensor_product_block_encoding = TensorProduct(
        [Unitary(TGate(), dtype=QBit()), Unitary(Hadamard(), dtype=QBit())]
    )
    return tensor_product_block_encoding


_TENSOR_PRODUCT_DOC = BloqDocSpec(
    bloq_cls=TensorProduct,
    import_line="from qualtran.bloqs.block_encoding import TensorProduct",
    examples=[_tensor_product_block_encoding],
)
