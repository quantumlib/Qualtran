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
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from attrs import evolve, field, frozen, validators
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QAny,
    QDType,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.symbolics import is_symbolic, ssum, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Partition(_BookkeepingBloq):
    """Partition a generic index into multiple registers.

    Args:
        n: The total bitsize of the un-partitioned register
        regs: Registers to partition into. The `side` attribute is ignored.
        partition: `False` means un-partition instead.

    Registers:
        x: the un-partitioned register. LEFT by default.
        [user spec]: The registers provided by the `regs` argument. RIGHT by default.
    """

    n: SymbolicInt
    regs: Tuple[Register, ...] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )
    partition: bool = True

    def __attrs_post_init__(self):
        if self.n != ssum(r.total_bits() for r in self.regs):
            raise ValueError("Total bitsize not equal to sum of registers to partition into")
        if len(set(r.name for r in self.regs)) != len(self.regs):
            raise ValueError("Duplicate register names")

    @cached_property
    def lumped_dtype(self) -> QDType:
        return QAny(bitsize=self.n)

    @cached_property
    def signature(self) -> 'Signature':
        lumped = Side.LEFT if self.partition else Side.RIGHT
        partitioned = Side.RIGHT if self.partition else Side.LEFT

        return Signature(
            [Register('x', self.lumped_dtype, side=lumped)]
            + [evolve(reg, side=partitioned) for reg in self.regs]
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self):
        return evolve(self, partition=not self.partition)

    def as_cirq_op(self, qubit_manager, **cirq_quregs) -> Tuple[None, Dict[str, 'CirqQuregT']]:
        if self.partition:
            outregs = {}
            start = 0
            for reg in self.regs:
                shape = reg.shape + (reg.bitsize,)
                size = int(np.prod(shape))
                outregs[reg.name] = np.array(cirq_quregs['x'][start : start + size]).reshape(shape)
                start += size
            return None, outregs
        else:
            return None, {'x': np.concatenate([v.ravel() for _, v in cirq_quregs.items()])}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        if is_symbolic(self.n):
            raise DecomposeTypeError(f"cannot compute tensors for symbolic {self}")

        grouped = incoming['x'] if self.partition else outgoing['x']
        partitioned = outgoing if self.partition else incoming

        partitioned_inds = []
        for reg in self.regs:
            part_ind = partitioned[reg.name]
            for idx in reg.all_idxs():
                for j in range(reg.bitsize):
                    if isinstance(part_ind, np.ndarray):
                        partitioned_inds.append((part_ind[idx], j))
                    else:
                        partitioned_inds.append((part_ind, j))

        return [
            qtn.Tensor(data=np.eye(2), inds=[partitioned_inds[j], (grouped, j)], tags=[str(self)])
            for j in range(self.n)
        ]

    def _classical_partition(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        out_vals = {}
        xbits = self.lumped_dtype.to_bits(x)
        start = 0
        for reg in self.regs:
            size = int(np.prod(reg.shape + (reg.bitsize,)))
            bits_reg = xbits[start : start + size]
            if reg.shape == ():
                out_vals[reg.name] = reg.dtype.from_bits(bits_reg)
            else:
                out_vals[reg.name] = reg.dtype.from_bits_array(
                    np.asarray(bits_reg).reshape(reg.shape + (reg.bitsize,))
                )
            start += size
        return out_vals

    def _classical_unpartition_to_bits(self, **vals: 'ClassicalValT') -> NDArray[np.uint8]:
        out_vals: list[NDArray[np.uint8]] = []
        for reg in self.regs:
            reg_val = np.asarray(vals[reg.name])
            bitstrings = reg.dtype.to_bits_array(reg_val.ravel())
            out_vals.append(bitstrings.ravel())
        return np.concatenate(out_vals)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if self.partition:
            return self._classical_partition(vals['x'])
        else:
            big_int_bits = self._classical_unpartition_to_bits(**vals)
            big_int = self.lumped_dtype.from_bits(big_int_bits.tolist())
            return {'x': big_int}

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.shape:
            text = f'[{",".join(str(i) for i in idx)}]'
            return directional_text_box(text, side=reg.side)
        return directional_text_box(' ', side=reg.side)


@bloq_example
def _partition() -> Partition:
    regs = (Register('xx', QAny(2), shape=(2, 3)), Register('yy', QAny(37)))
    bitsize = sum(reg.total_bits() for reg in regs)
    partition = Partition(n=bitsize, regs=regs)
    return partition


_PARTITION_DOC = BloqDocSpec(bloq_cls=Partition, examples=[_partition], call_graph_example=None)
