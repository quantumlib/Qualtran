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
from typing import cast, Dict, Tuple

import cirq
from attrs import evolve, field, frozen, validators
from numpy.typing import NDArray

from qualtran import (
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
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.symbolics import is_symbolic, prod, smax, ssum, SymbolicFloat, SymbolicInt


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

    def pretty_name(self) -> str:
        return f"B[{'*'.join(u.pretty_name()[2:-1] for u in self.block_encodings)}]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return prod(u.alpha for u in self.block_encodings)

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return smax(u.ancilla_bitsize for u in self.block_encodings) + len(self.block_encodings) - 1

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
        # This method will be implemented in the future after PrepareOracle
        # is updated for the BlockEncoding interface.
        # Github issue: https://github.com/quantumlib/Qualtran/issues/1104
        raise NotImplementedError

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        if (
            is_symbolic(self.system_bitsize)
            or is_symbolic(self.ancilla_bitsize)
            or is_symbolic(self.resource_bitsize)
        ):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        assert (
            isinstance(self.system_bitsize, int)
            and isinstance(self.ancilla_bitsize, int)
            and isinstance(self.resource_bitsize, int)
        )

        n = len(self.block_encodings)
        res_bits_used = 0
        for i, u in enumerate(reversed(self.block_encodings)):
            u_soqs = {"system": system}

            # split ancilla register if necessary
            if self.ancilla_bitsize > 0:
                anc_bits = cast(int, u.ancilla_bitsize)
                flag_bits = Register("flag_bits", dtype=QBit(), shape=(n - 1,))  # type: ignore
                anc_used = Register("anc_used", dtype=QAny(anc_bits))
                anc_unused_bits = self.ancilla_bitsize - (n - 1) - anc_bits
                anc_unused = Register("anc_unused", dtype=QAny(anc_unused_bits))
                anc_regs = [flag_bits]
                if anc_bits > 0:
                    anc_regs.append(anc_used)
                if anc_unused_bits > 0:
                    anc_regs.append(anc_unused)
                anc_part = Partition(self.ancilla_bitsize, tuple(anc_regs))
                anc_part_soqs = bb.add_d(anc_part, x=soqs["ancilla"])
                if anc_bits > 0:
                    u_soqs["ancilla"] = anc_part_soqs["anc_used"]

            # split resource register if necessary
            res_bits = cast(int, u.resource_bitsize)
            if res_bits > 0:
                res_before = Register("res_before", dtype=QAny(res_bits_used))
                res = Register("res", dtype=QAny(res_bits))
                res_bits_left = self.resource_bitsize - res_bits_used - res_bits
                res_after = Register("res_after", dtype=QAny(res_bits_left))
                res_regs = []
                if res_bits_used > 0:
                    res_regs.append(res_before)
                res_regs.append(res)
                res_bits_used += res_bits
                if res_bits_left > 0:
                    res_regs.append(res_after)
                res_part = Partition(self.resource_bitsize, tuple(res_regs))
                res_part_soqs = bb.add_d(res_part, x=soqs["resource"])
                u_soqs["resource"] = res_part_soqs["res"]

            # connect the constituent bloq
            u_out_soqs = bb.add_d(u, **u_soqs)
            system = u_out_soqs["system"]

            # un-partition the resource register
            if res_bits > 0:
                res_part_soqs["res"] = u_out_soqs["resource"]
                soqs["resource"] = cast(
                    Soquet, bb.add(evolve(res_part, partition=False), **res_part_soqs)
                )

            # un-partition the ancilla register
            if self.ancilla_bitsize > 0:
                flag_bits_soq = cast(NDArray, anc_part_soqs["flag_bits"])
                if anc_bits > 0:
                    anc_used_soq = cast(Soquet, u_out_soqs["ancilla"])
                    if i == n - 1:
                        anc_part_soqs["anc_used"] = anc_used_soq
                    else:
                        # set corresponding flag if ancillas are all zero
                        ctrl, flag_bits_soq[i] = bb.add_t(
                            MultiControlPauli(tuple([0] * anc_bits), cirq.X),
                            controls=bb.split(anc_used_soq),
                            target=flag_bits_soq[i],
                        )
                        flag_bits_soq[i] = bb.add(XGate(), q=flag_bits_soq[i])
                        anc_part_soqs["anc_used"] = bb.join(cast(NDArray, ctrl))
                anc_part_soqs["flag_bits"] = flag_bits_soq
                soqs["ancilla"] = cast(
                    Soquet, bb.add(evolve(anc_part, partition=False), **anc_part_soqs)
                )

        out = {"system": system}
        if self.ancilla_bitsize > 0:
            out["ancilla"] = soqs["ancilla"]
        if self.resource_bitsize > 0:
            out["resource"] = soqs["resource"]
        return out


@bloq_example
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
    import_line="from qualtran.bloqs.block_encoding import Product",
    examples=[
        _product_block_encoding,
        _product_block_encoding_properties,
        _product_block_encoding_symb,
    ],
)
