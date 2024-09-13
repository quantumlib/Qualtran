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
from typing import cast, Dict, List, Sequence, Tuple, Union

from attrs import field, frozen, validators
from numpy.typing import NDArray
from typing_extensions import Self

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates.x_basis import XGate
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import HasLength, is_symbolic, prod, smax, ssum, SymbolicFloat, SymbolicInt
from qualtran.symbolics.math_funcs import is_zero


@frozen
class Product(BlockEncoding):
    r"""Product of a sequence of block encodings.

    Builds the block encoding $B[U_1 * U_2 * \cdots * U_n]$ given block encodings
    $B[U_1], \ldots, B[U_n]$.

    When each $B[U_i]$ is a $(\alpha_i, a_i, \epsilon_i)$-block encoding of $U_i$, we have that
    $B[U_1 * \cdots * U_n]$ is a block encoding of $U_1 * \cdots * U_n$ with normalization
    constant $\prod_i \alpha_i$, ancilla bitsize $n - 1 + \max_i a_i$, and precision
    $\sum_i \alpha_i \epsilon_i$.

    Following Fig. 2 in Dalzell et al. (2023), Ch. 10.2, the product is encoded by concatenating
    each constituent block encoding, using a shared ancilla register and a set of flag qubits to
    verify that the ancilla is left as zero after each use:
    ```
           ┌────────┐
      |0> ─┤        ├─     |0> ───────────X──────X────
           │        │                     │
           │ U_(AB) │  =        ┌─────┐   │   ┌─────┐
      |0> ─┤        ├─     |0> ─┤     ├──(0)──┤     ├─
           │        │           │ U_B │       │ U_A │
    |Psi> ─┤        ├─   |Psi> ─┤     ├───────┤     ├─
           └────────┘           └─────┘       └─────┘
    ```

    Args:
        block_encodings: A sequence of block encodings.

    Registers:
        system: The system register.
        ancilla: The ancilla register (present only if bitsize > 0).
        resource: The resource register (present only if bitsize > 0).

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](
        https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    block_encodings: Tuple[BlockEncoding, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )

    def __attrs_post_init__(self):
        if not all(u.system_bitsize == self.system_bitsize for u in self.block_encodings):
            raise ValueError("All block encodings must have the same system size.")
        if not all(
            isinstance(u.signal_state.prepare, PrepareIdentity) for u in self.block_encodings
        ):
            raise ValueError("Cannot take product of block encodings with non-zero signal state.")

    @classmethod
    def of(cls, *block_encodings: BlockEncoding) -> Self:
        """Construct a `Product` from block encodings."""
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
        return self.block_encodings[0].system_bitsize

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return prod(u.alpha for u in self.block_encodings)

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return smax(u.ancilla_bitsize for u in self.block_encodings) + len(self.block_encodings) - 1

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return smax(u.resource_bitsize for u in self.block_encodings)

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return ssum(u.alpha * u.epsilon for u in self.block_encodings)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @property
    def anc_part(self) -> Partition:
        n = len(self.block_encodings)
        anc_regs = []
        if n - 1 > 0:
            anc_regs.append(Register("flag_bits", dtype=QBit(), shape=(n - 1,)))
        anc_bits = self.ancilla_bitsize - (n - 1)
        if not is_zero(anc_bits):
            anc_regs.append(Register("ancilla", dtype=QAny(anc_bits)))
        return Partition(cast(int, self.ancilla_bitsize), tuple(anc_regs))

    @property
    def constituents(self) -> Sequence[Bloq]:
        n = len(self.block_encodings)
        anc_bits = self.ancilla_bitsize - (n - 1)
        ret = []
        for u in reversed(self.block_encodings):
            partition: List[Tuple[Register, List[Union[str, Unused]]]] = [
                (Register("system", dtype=QAny(u.system_bitsize)), ["system"])
            ]
            if is_symbolic(u.ancilla_bitsize) or u.ancilla_bitsize > 0:
                regs: List[Union[str, Unused]] = ["ancilla"]
                if (
                    is_symbolic(anc_bits)
                    or is_symbolic(u.ancilla_bitsize)
                    or anc_bits > u.ancilla_bitsize
                ):
                    regs.append(Unused(anc_bits - u.ancilla_bitsize))
                partition.append((Register("ancilla", dtype=QAny(anc_bits)), regs))
            if not is_zero(u.resource_bitsize):
                regs = ["resource"]
                if is_symbolic(self.resource_bitsize) or self.resource_bitsize > u.resource_bitsize:
                    regs.append(Unused(self.resource_bitsize - u.resource_bitsize))
                partition.append((Register("resource", dtype=QAny(u.resource_bitsize)), regs))
            ret.append(AutoPartition(u, partition, left_only=False))
        return ret

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        counts = Counter[Bloq]()
        for bloq in self.constituents:
            counts[bloq] += 1
        n = len(self.block_encodings)
        for i, u in enumerate(reversed(self.block_encodings)):
            if not is_zero(u.ancilla_bitsize) and n - 1 > 0 and i != n - 1:
                counts[MultiControlX(HasLength(u.ancilla_bitsize))] += 1
                counts[XGate()] += 1
        return counts

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        n = len(self.block_encodings)

        if self.ancilla_bitsize > 0:
            # partition ancilla into flag and inner ancilla
            anc_bits = self.ancilla_bitsize - (n - 1)
            anc_part_soqs = bb.add_d(self.anc_part, x=soqs.pop("ancilla"))
            if n - 1 > 0:
                flag_bits_soq = cast(NDArray, anc_part_soqs.pop("flag_bits"))
            if anc_bits > 0:
                anc_soq = anc_part_soqs.pop("ancilla")
        if self.resource_bitsize > 0:
            # Currently, we assume that block encodings restore their resource register to zero.
            # If so, the resource register can be reused among all the consistuents.
            # See https://github.com/quantumlib/Qualtran/issues/1138
            # which tracks necessary changes if this assumption becomes false.
            res_soq = soqs.pop("resource")

        # connect constituent bloqs
        for i, u in enumerate(reversed(self.block_encodings)):
            assert not is_symbolic(u.ancilla_bitsize)
            assert not is_symbolic(u.resource_bitsize)
            u_soqs = {"system": system}
            if u.ancilla_bitsize > 0:
                u_soqs["ancilla"] = anc_soq
            if u.resource_bitsize > 0:
                u_soqs["resource"] = res_soq
            u_out_soqs = bb.add_d(self.constituents[i], **u_soqs)
            system = u_out_soqs.pop("system")
            if u.ancilla_bitsize > 0:
                anc_soq = u_out_soqs.pop("ancilla")
            if u.resource_bitsize > 0:
                res_soq = u_out_soqs.pop("resource")

            # set corresponding flag if ancillas are all zero
            if u.ancilla_bitsize > 0 and n - 1 > 0 and i != n - 1:
                controls = bb.split(cast(Soquet, anc_soq))
                controls[: u.ancilla_bitsize], flag_bits_soq[i] = bb.add_t(
                    MultiControlX(tuple([0] * u.ancilla_bitsize)),
                    controls=controls[: u.ancilla_bitsize],
                    target=flag_bits_soq[i],
                )
                flag_bits_soq[i] = bb.add(XGate(), q=flag_bits_soq[i])
                anc_soq = bb.join(controls)

        out = {"system": system}
        if self.resource_bitsize > 0:
            out["resource"] = res_soq
        if self.ancilla_bitsize > 0:
            anc_soqs: Dict[str, SoquetT] = dict()
            if n - 1 > 0:
                anc_soqs["flag_bits"] = flag_bits_soq
            if anc_bits > 0:
                anc_soqs["ancilla"] = anc_soq
            out["ancilla"] = cast(Soquet, bb.add(self.anc_part.adjoint(), **anc_soqs))
        return out

    def __str__(self) -> str:
        return f"B[{'*'.join(str(u)[2:-1] for u in self.block_encodings)}]"


@bloq_example(generalizer=ignore_split_join)
def _product_block_encoding() -> Product:
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    product_block_encoding = Product((Unitary(TGate()), Unitary(Hadamard())))
    return product_block_encoding


@bloq_example
def _product_block_encoding_properties() -> Product:
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    u1 = Unitary(TGate(), alpha=0.5, ancilla_bitsize=2, resource_bitsize=1, epsilon=0.01)
    u2 = Unitary(Hadamard(), alpha=0.5, ancilla_bitsize=1, resource_bitsize=1, epsilon=0.1)
    product_block_encoding_properties = Product((u1, u2))
    return product_block_encoding_properties


@bloq_example
def _product_block_encoding_symb() -> Product:
    import sympy

    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    alpha1 = sympy.Symbol('alpha1')
    a1 = sympy.Symbol('a1')
    eps1 = sympy.Symbol('eps1')
    alpha2 = sympy.Symbol('alpha2')
    a2 = sympy.Symbol('a2')
    eps2 = sympy.Symbol('eps2')
    product_block_encoding_symb = Product(
        (
            Unitary(TGate(), alpha=alpha1, ancilla_bitsize=a1, epsilon=eps1),
            Unitary(Hadamard(), alpha=alpha2, ancilla_bitsize=a2, epsilon=eps2),
        )
    )
    return product_block_encoding_symb


_PRODUCT_DOC = BloqDocSpec(
    bloq_cls=Product,
    examples=[
        _product_block_encoding,
        _product_block_encoding_properties,
        _product_block_encoding_symb,
    ],
)
