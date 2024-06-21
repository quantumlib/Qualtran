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
from typing import Dict, Sequence, Tuple

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


@frozen
class AutoPartition(Bloq):
    """Wrap a bloq with `Partition` to fit an alternative set of input and output registers
       such that splits / joins can be avoided in diagrams.

    Args:
        bloq: The bloq to wrap.
        partitions: A sequence of pairs specifying each register that the wrapped bloq should accept
            and the names of registers from `bloq.signature.lefts()` that concatenate to form it.
        partition_output: If True, the output registers will also follow `partition`.
            Otherwise, the output registers will follow `bloq.signature.rights()`.

    Registers:
        [user_spec]: The output registers of the wrapped bloq.
    """

    bloq: Bloq
    partitions: Sequence[Tuple[Register, Sequence[str]]] = field(
        converter=lambda s: tuple((r, tuple(rs)) for r, rs in s)
    )
    partition_output: bool = True

    def __attrs_post_init__(self):
        regs = {r.name for r in self.bloq.signature.lefts()}
        assert len(regs) == len(self.bloq.signature)
        assert regs == {r for _, rs in self.partitions for r in rs}

    @cached_property
    def signature(self) -> Signature:
        if self.partition_output:
            return Signature(r for r, _ in self.partitions)
        else:
            return Signature(
                chain(
                    (evolve(r, side=Side.LEFT) for r, _ in self.partitions),
                    (evolve(r, side=Side.RIGHT) for r in self.bloq.signature.rights()),
                )
            )

    def pretty_name(self) -> str:
        return self.bloq.pretty_name()

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        parts: Dict[str, Partition] = dict()
        in_regs: Dict[str, SoquetT] = dict()
        for out_reg, bloq_regs in self.partitions:
            part = Partition(
                out_reg.bitsize, regs=tuple(self.bloq.signature.get_left(r) for r in bloq_regs)
            )
            parts[out_reg.name] = part
            in_regs |= bb.add_d(part, x=soqs[out_reg.name])
        bloq_out_regs = bb.add_d(self.bloq, **in_regs)
        if not self.partition_output:
            return bloq_out_regs

        out_regs = {}
        for soq_name in soqs.keys():
            out_regs[soq_name] = bb.add(
                parts[soq_name].adjoint(),
                **{reg.name: bloq_out_regs[reg.name] for reg in parts[soq_name].signature.rights()},
            )
        return out_regs


@bloq_example
def _auto_partition() -> AutoPartition:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Swap

    bloq = Controlled(Swap(1), CtrlSpec())
    auto_partition = AutoPartition(
        bloq, [(Register('x', QAny(2)), ['ctrl', 'x']), (Register('y', QAny(1)), ['y'])]
    )
    return auto_partition


_AUTO_PARTITION_DOC = BloqDocSpec(
    bloq_cls=AutoPartition,
    import_line="from qualtran.bloqs.bookkeeping import AutoPartition",
    examples=[_auto_partition],
)
