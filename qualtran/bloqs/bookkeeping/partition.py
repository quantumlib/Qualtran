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
from typing import Any, Dict, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    DecomposeTypeError,
    QAny,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.simulation.classical_sim import bits_to_ints, ints_to_bits

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

    n: int
    regs: Tuple[Register, ...]
    partition: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        lumped = Side.LEFT if self.partition else Side.RIGHT
        partitioned = Side.RIGHT if self.partition else Side.LEFT

        return Signature(
            [Register('x', QAny(bitsize=self.n), side=lumped)]
            + [attrs.evolve(reg, side=partitioned) for reg in self.regs]
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self):
        return attrs.evolve(self, partition=not self.partition)

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

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        unitary_shape = []
        soquets = []
        _incoming = incoming if self.partition else outgoing
        _outgoing = outgoing if self.partition else incoming
        for reg in self.regs:
            for i in range(int(np.prod(reg.shape))):
                unitary_shape.append(2**reg.bitsize)
                outgoing_reg = _outgoing[reg.name]
                if isinstance(outgoing_reg, np.ndarray):
                    soquets.append(outgoing_reg.ravel()[i])
                else:
                    soquets.append(outgoing_reg)

        tn.add(
            qtn.Tensor(
                data=np.eye(2**self.n, 2**self.n).reshape(
                    tuple(unitary_shape) + (2**self.n,)
                ),
                inds=soquets + [_incoming['x']],
                tags=['Partition', tag],
            )
        )

    def _classical_partition(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        out_vals = {}
        xbits = ints_to_bits(x, self.n)[0]
        start = 0
        for reg in self.regs:
            size = int(np.prod(reg.shape + (reg.bitsize,)))
            bits_reg = xbits[start : start + size]
            if reg.shape == ():
                out_vals[reg.name] = bits_to_ints(bits_reg)[0]
            else:
                ints_reg = bits_to_ints(
                    [
                        bits_reg[i * reg.bitsize : (i + 1) * reg.bitsize]
                        for i in range(np.prod(reg.shape))
                    ]
                )
                out_vals[reg.name] = np.array(ints_reg).reshape(reg.shape)
            start += size
        return out_vals

    def _classical_unpartition(self, **vals: 'ClassicalValT'):
        out_vals = []
        for reg in self.regs:
            reg_val = vals[reg.name]
            if isinstance(reg_val, np.ndarray):
                out_vals.append(ints_to_bits(reg_val.ravel(), reg.bitsize).ravel())
            else:
                out_vals.append(ints_to_bits(reg_val, reg.bitsize)[0])
        big_int = np.concatenate(out_vals)
        return {'x': bits_to_ints(big_int)[0]}

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if self.partition:
            return self._classical_partition(vals['x'])
        else:
            return self._classical_unpartition(**vals)

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
