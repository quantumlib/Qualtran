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
from itertools import chain
from typing import Dict, Sequence, Tuple, Union

from attrs import evolve, field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.symbolics.types import SymbolicInt


@frozen
class Unused:
    """Placeholder indicating that a portion of a register is unused in an AutoPartition."""

    bitsize: SymbolicInt


@frozen
class AutoPartition(Bloq):
    """Automatically adds and undoes `Partition` of registers to match the signature of a sub-bloq.

    This tool enables using a bloq in a context expecting an alternative signature that combines
    registers in the bloq's signature or operates over more registers than the bloq does.
    For example, it can adapt a bloq exposing multiple selection registers to a quantum interface
    that expects only one unified selection register.

    Wrapping in `AutoPartition` also hides splits and joins behind a level of decomposition, which
    can produce more helpful circuit diagrams compared to manually splitting and joining.

    Args:
        bloq: The sub-bloq to wrap. Its register names are used within the second items in each
            pair in the `partitions` argument below.
        partitions: A sequence of pairs specifying each register that is exposed in the external
            signature of the `AutoPartition` and its relationship to the registers of `bloq`. The
            first element of each pair is a `Register` exposed externally. The second is a list of
            register names of `bloq` that concatenate to form the externally exposed register.
            If `bloq` does not operate on some portion (of `n` bits) of the externally exposed
            register, the sentinel value `Unused(n)` can be used in place of a register name.
        left_only: If False, the output registers will also follow `partition`.
            Otherwise, the output registers will follow `bloq.signature.rights()`.
            This flag must be set to True if `bloq` does not have the same LEFT and RIGHT registers,
            as is required for the bloq to be fully wrapped on the left and right.

    Registers:
        [user_spec]: The output registers of the wrapped bloq.
    """

    bloq: Bloq
    partitions: Sequence[Tuple[Register, Sequence[Union[str, Unused]]]] = field(
        converter=lambda s: tuple((r, tuple(rs)) for r, rs in s)
    )
    left_only: bool = False

    def __attrs_post_init__(self):
        regs = {r.name for r in self.bloq.signature.lefts()}
        if len(regs) != len(self.bloq.signature):
            raise ValueError("Length of regs and signature do not match")
        if regs != {r for _, rs in self.partitions for r in rs if isinstance(r, str)}:
            raise ValueError("Bloq registers do not match given partitions")
        if self.left_only:
            for _, rs in self.partitions:
                for r in rs:
                    if isinstance(r, Unused):
                        raise ValueError("Cannot use left_only with unused registers")

    @cached_property
    def signature(self) -> Signature:
        if self.left_only:
            return Signature(
                chain(
                    (evolve(r, side=Side.LEFT) for r, _ in self.partitions),
                    (evolve(r, side=Side.RIGHT) for r in self.bloq.signature.rights()),
                )
            )
        else:
            return Signature(r for r, _ in self.partitions)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        parts: Dict[str, Partition] = dict()
        in_regs: Dict[str, SoquetT] = dict()
        unused_regs: Dict[str, SoquetT] = dict()
        for parti, (out_reg, bloq_regs) in enumerate(self.partitions):
            part = Partition(
                out_reg.bitsize,
                regs=tuple(
                    (
                        self.bloq.signature.get_left(r)
                        if isinstance(r, str)
                        else Register(f"_unused{parti}_{i}", QAny(r.bitsize))
                    )
                    for i, r in enumerate(bloq_regs)
                ),
            )
            parts[out_reg.name] = part
            part_regs = bb.add_d(part, x=soqs[out_reg.name])
            for k, v in part_regs.items():
                if k in self.bloq.signature._lefts:
                    in_regs[k] = v
                else:
                    unused_regs[k] = v

        bloq_out_regs = bb.add_d(self.bloq, **in_regs)
        if self.left_only:
            return bloq_out_regs

        out_regs = {}
        for soq_name in soqs.keys():
            out_regs[soq_name] = bb.add(
                parts[soq_name].adjoint(),
                **{
                    reg.name: (bloq_out_regs | unused_regs)[reg.name]
                    for reg in parts[soq_name].signature.rights()
                },
            )
        return out_regs

    def __str__(self) -> str:
        return str(self.bloq)


@bloq_example
def _auto_partition() -> AutoPartition:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Swap

    bloq = Controlled(Swap(1), CtrlSpec())
    auto_partition = AutoPartition(
        bloq, [(Register('x', QAny(2)), ['ctrl', 'x']), (Register('y', QAny(1)), ['y'])]
    )
    return auto_partition


@bloq_example
def _auto_partition_unused() -> AutoPartition:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Swap
    from qualtran.bloqs.bookkeeping.auto_partition import Unused

    bloq = Controlled(Swap(1), CtrlSpec())
    auto_partition_unused = AutoPartition(
        bloq,
        [
            (Register('x', QAny(3)), ['ctrl', 'x', Unused(1)]),
            (Register('y', QAny(1)), ['y']),
            (Register('z', QAny(2)), [Unused(2)]),
        ],
    )
    return auto_partition_unused


_AUTO_PARTITION_DOC = BloqDocSpec(
    bloq_cls=AutoPartition, examples=[_auto_partition, _auto_partition_unused]
)
