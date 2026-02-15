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
import abc
import warnings
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

import numpy as np
import sympy
from attrs import evolve, field, frozen, validators
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QAny,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.symbolics import is_symbolic, ssum, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from pennylane.operation import Operation
    from pennylane.wires import Wires

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


class LegacyPartitionWarning(DeprecationWarning):
    """Warnings for legacy Partition usage, when declaring only n."""


def _constrain_qany_reg(reg: Register):
    """Changes the dtype of a register to note break legacy code

    This function should be bound to dissapear
    """
    if isinstance(reg.dtype, QAny):
        warnings.warn(
            f"Doing classical casting with QAny ({reg=}) is ambiguous, transforming it as QUInt for legacy purposes",
            category=LegacyPartitionWarning,
        )
        return evolve(reg, dtype=QUInt(reg.dtype.bitsize))
    return reg


def _regs_to_tuple(x):
    if x is None:
        return None
    return x if isinstance(x, tuple) else tuple(x)


def _not_none(_inst, attr, value):
    if value is None:
        raise ValueError(f"{attr.name} cannot be None")


class _PartitionBase(_BookkeepingBloq, metaclass=abc.ABCMeta):
    """Generalized paritioning functionality."""

    @property
    @abc.abstractmethod
    def n(self) -> Optional[SymbolicInt]: ...

    @property
    @abc.abstractmethod
    def lumped_dtype(self) -> QDType: ...

    @property
    @abc.abstractmethod
    def _regs(self) -> Sequence[Register]:
        """A way of splitting up `QAny(n)` into registers"""

    @property
    @abc.abstractmethod
    def partition(self) -> bool:
        """Whether this takes one register into seveal smaller registers (or vice-versa)"""

    def _validate(self):
        if self.n != ssum(r.total_bits() for r in self._regs):
            raise ValueError("Total bitsize not equal to sum of registers to partition into")
        if len(set(r.name for r in self._regs)) != len(self._regs):
            raise ValueError("Duplicate register names")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def as_cirq_op(self, qubit_manager, **cirq_quregs) -> Tuple[None, Dict[str, 'CirqQuregT']]:
        self._validate()
        if self.partition:
            outregs = {}
            start = 0
            for reg in self._regs:
                shape = reg.shape + (reg.bitsize,)
                size = int(np.prod(shape))
                outregs[reg.name] = np.array(cirq_quregs['x'][start : start + size]).reshape(shape)
                start += size
            return None, outregs
        else:
            return None, {'x': np.concatenate([v.ravel() for _, v in cirq_quregs.items()])}

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        return None

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        if self.n is None:
            raise DecomposeTypeError(f"cannot compute tensors with unknown n for {self}")
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"cannot compute tensors for symbolic {self}")

        grouped = incoming['x'] if self.partition else outgoing['x']
        partitioned = outgoing if self.partition else incoming

        partitioned_inds = []
        for reg in self._regs:
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
        for reg in self._regs:
            reg = _constrain_qany_reg(reg)
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
        for reg in self._regs:
            reg = _constrain_qany_reg(reg)
            reg_val = np.asanyarray(vals[reg.name])
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

        label = reg.name
        if len(idx) > 0:
            label = f'{label}[{", ".join(str(i) for i in idx)}]'
        return directional_text_box(label, side=reg.side)


@frozen
class Partition(_PartitionBase):
    """Partition a generic index into multiple registers.

    Args:
        n: The total bit-size of the un-partitioned register. Required if `dtype_in` is None.
            Deprecated. Kept for backward compatibility. Use `dtype_in` instead whenever possible.
        regs: Registers to partition into. The `side` attribute is ignored.
        dtype_in: Type of the un-partitioned register. Required if `n` is None. If None,
            the type is inferred as `QUInt(n)`.
        partition: `False` means un-partition instead.

    Registers:
        x: the un-partitioned register. LEFT by default.
        [user spec]: The registers provided by the `regs` argument. RIGHT by default.
    """

    n: Optional[SymbolicInt] = field(default=None)
    regs: Optional[Tuple[Register, ...]] = field(
        converter=_regs_to_tuple, validator=(_not_none, validators.min_len(1)), default=None
    )
    dtype_in: Optional[QDType] = field(default=None)
    partition: bool = field(default=True)

    def __attrs_post_init__(self):
        if self.n is None and self.dtype_in is None:
            raise ValueError(f"Provide exactly n or dtype_in {self.n=}, {self.dtype_in=}")
        elif self.n is not None and self.dtype_in is None:
            warnings.warn(
                "Partition: By not setting dtype_in you could encounter errors when running "
                "assert_consistent_classical_action",
                category=LegacyPartitionWarning,
            )
        elif self.n is None and self.dtype_in is not None:
            object.__setattr__(self, "n", self.dtype_in.num_qubits)
        elif self.n is not None and self.dtype_in is not None:
            if self.n != self.dtype_in.num_qubits:
                raise ValueError(
                    f"{self.dtype_in=} should have size {self.n=}, currently {self.dtype_in.num_qubits=}"
                )
            warnings.warn(
                "Specifying both n and dtype_in is redundant", category=UserWarning, stacklevel=1
            )

        self._validate()

    @property
    def lumped_dtype(self) -> QDType:
        return QUInt(bitsize=cast(SymbolicInt, self.n)) if self.dtype_in is None else self.dtype_in

    @property
    def _regs(self) -> Sequence[Register]:
        return cast(Tuple[Register, ...], self.regs)

    @cached_property
    def signature(self) -> 'Signature':
        lumped = Side.LEFT if self.partition else Side.RIGHT
        partitioned = Side.RIGHT if self.partition else Side.LEFT

        return Signature(
            [Register('x', self.lumped_dtype, side=lumped)]
            + [evolve(reg, side=partitioned) for reg in self._regs]
        )

    def adjoint(self):
        return evolve(self, n=None, dtype_in=self.lumped_dtype, partition=not self.partition)


@frozen
class Split2(_PartitionBase):
    """Split one register into two registers.

    Contrast this with `Split`, which splits one register into `n` 1-bit wires. See also
    `Partition` which allows arbitrary partitioning schemes.

    Args:
        n1: The size of the y1 output register. n1 + n2 must add up to the size of the
            input register.
        n2: The size of the y2 output register. n1 + n2 must add up to the size of the
            input register.

    Registers:
        x [LEFT]: The input register of size n = n1 + n2 and type QAny.
        y1 [RIGHT]: The first output register of size n1 and type QAny.
        y2 [RIGHT]: The second output register of size n2 and type QAny.
    """

    n1: SymbolicInt
    n2: SymbolicInt

    @property
    def n(self) -> SymbolicInt:
        return self.n1 + self.n2

    @property
    def lumped_dtype(self) -> QDType:
        return QUInt(bitsize=self.n)

    @property
    def partition(self) -> bool:
        return True

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.lumped_dtype, side=Side.LEFT),
                Register('y1', QAny(self.n1), side=Side.RIGHT),
                Register('y2', QAny(self.n2), side=Side.RIGHT),
            ]
        )

    @property
    def _regs(self) -> Sequence[Register]:
        return (Register('y1', QAny(self.n1)), Register('y2', QAny(self.n2)))

    def adjoint(self) -> 'Bloq':
        return Join2(self.n1, self.n2)

    def __str__(self):
        return f'Split2({self.n1}, {self.n2})'


@bloq_example
def _split2() -> Split2:
    n1, n2 = sympy.symbols('n1 n2')
    split2 = Split2(n1, n2)
    return split2


_SPLIT2_DOC = BloqDocSpec(bloq_cls=Split2, examples=[_split2], call_graph_example=None)


@frozen
class Join2(_PartitionBase):
    """Join two registers into one register.

    Contrast this with `Join`, which joins `n` 1-bit registers into one. See also
    `Partition` which allows arbitrary partitioning schemes.

    Args:
        n1: The size of the y1 input register. n1 + n2 must add up to the size of the
            output register.
        n2: The size of the y2 input register. n1 + n2 must add up to the size of the
            output register.

    Registers:
        x [RIGHT]: The output register of size n = n1 + n2 and type QAny.
        y1 [LEFT]: The first input register of size n1 and type QAny.
        y2 [LEFT]: The second input register of size n2 and type QAny.
    """

    n1: SymbolicInt
    n2: SymbolicInt

    @property
    def n(self) -> SymbolicInt:
        return self.n1 + self.n2

    @property
    def lumped_dtype(self) -> QDType:
        return QUInt(bitsize=self.n)

    @property
    def partition(self) -> bool:
        return False

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.lumped_dtype, side=Side.RIGHT),
                Register('y1', QAny(self.n1), side=Side.LEFT),
                Register('y2', QAny(self.n2), side=Side.LEFT),
            ]
        )

    @property
    def _regs(self) -> Sequence[Register]:
        return (Register('y1', QAny(self.n1)), Register('y2', QAny(self.n2)))

    def adjoint(self) -> 'Bloq':
        return Split2(self.n1, self.n2)

    def __str__(self):
        return f'Join2({self.n1}, {self.n2})'


@bloq_example
def _join2() -> Join2:
    n1, n2 = sympy.symbols('n1 n2')
    join2 = Join2(n1, n2)
    return join2


_JOIN2_DOC = BloqDocSpec(bloq_cls=Join2, examples=[_join2], call_graph_example=None)


@bloq_example
def _partition() -> Partition:
    regs = (Register('xx', QAny(2), shape=(2, 3)), Register('yy', QAny(37)))
    bitsize = sum(reg.total_bits() for reg in regs)
    partition = Partition(n=bitsize, regs=regs)
    return partition


_PARTITION_DOC = BloqDocSpec(bloq_cls=Partition, examples=[_partition], call_graph_example=None)
