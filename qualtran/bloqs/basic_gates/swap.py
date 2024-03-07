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
from typing import Any, Dict, Sequence, Set, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import quimb.tensor as qtn
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    GateWithRegisters,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join

from .t_gate import TGate

if TYPE_CHECKING:
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _swap_matrix() -> NDArray[np.complex128]:
    x = np.eye(2**2, dtype=np.complex128).reshape((2,) * 2 * 2)
    return x.transpose([0, 3, 1, 2])


def _controlled_swap_matrix():
    x = np.eye(2**3, dtype=np.complex128).reshape((2,) * 3 * 2)
    x[1, :, :, 1, ::] = _swap_matrix()
    return x


@frozen
class TwoBitSwap(Bloq):
    """Swap two bits.

    Registers:
        x: the first bit
        y: the second bit
    """

    def short_name(self) -> str:
        return 'swap'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(x=1, y=1)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', x: 'CirqQuregT', y: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        (x,) = x
        (y,) = y
        return cirq.SWAP.on(x, y), {'x': [x], 'y': [y]}

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        matrix = _swap_matrix()
        out_inds = [outgoing['x'], outgoing['y']]
        in_inds = [incoming['x'], incoming['y']]
        tn.add(qtn.Tensor(data=matrix, inds=out_inds + in_inds, tags=[self.short_name(), tag]))

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'x': y, 'y': x}


@frozen
class TwoBitCSwap(Bloq):
    """Swap two bits controlled on a control bit.

    This is sometimes known as the [Fredkin Gate](https://en.wikipedia.org/wiki/Fredkin_gate).

    Registers:
        ctrl: the control bit
        x: the first bit
        y: the second bit
    """

    def short_name(self) -> str:
        return 'swap'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=1, y=1)

    def as_cirq_op(
        self,
        qubit_manager: 'cirq.QubitManager',
        ctrl: 'CirqQuregT',
        x: 'CirqQuregT',
        y: 'CirqQuregT',
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        (ctrl,) = ctrl
        (x,) = x
        (y,) = y
        return cirq.CSWAP.on(ctrl, x, y), {'ctrl': [ctrl], 'x': [x], 'y': [y]}

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        matrix = _controlled_swap_matrix()
        out_inds = [outgoing['ctrl'], outgoing['x'], outgoing['y']]
        in_inds = [incoming['ctrl'], incoming['x'], incoming['y']]
        tn.add(qtn.Tensor(data=matrix, inds=out_inds + in_inds, tags=[self.short_name(), tag]))

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for TwoBitCSwap classical simulation.")

    def _t_complexity_(self) -> 'TComplexity':
        """The t complexity.

        References:
            [An algorithm for the T-count](https://arxiv.org/abs/1308.4134). Gosset et. al. 2013.
            Figure 5.2.
        """
        # https://arxiv.org/abs/1308.4134
        return TComplexity(t=7, clifford=10)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TGate(), 7), (ArbitraryClifford(n=3), 10)}

    def adjoint(self) -> 'Bloq':
        return self


@frozen
class Swap(Bloq):
    """Swap two registers

    Args:
        bitsize: The bitsize of each of the two registers being swapped.

    Registers:
        x: the first register
        y: the second register
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize, y=self.bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise DecomposeTypeError("`bitsize` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.bitsize):
            xs[i], ys[i] = bb.add(TwoBitSwap(), x=xs[i], y=ys[i])

        return {'x': bb.join(xs), 'y': bb.join(ys)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TwoBitSwap(), self.bitsize)}

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'x': y, 'y': x}

    def short_name(self) -> str:
        return 'swap'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('×(x)')
        elif soq.reg.name == 'y':
            return TextBox('×(y)')
        raise ValueError(f"Bad register name {soq.reg.name}")

    def adjoint(self) -> 'Bloq':
        return self


@bloq_example(generalizer=ignore_split_join)
def _swap_small() -> Swap:
    swap_small = Swap(bitsize=4)
    return swap_small


@frozen
class CSwap(GateWithRegisters):
    """Swap two registers controlled on a control bit.

    Implements a multi-target controlled swap unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$.

    This decomposes into a qubitwise SWAP on the two target registers, and takes $14n$ T-gates.

    Args:
        bitsize: The bitsize of each of the two registers being swapped.

    Registers:
        ctrl: the control bit
        x: the first register
        y: the second register
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise DecomposeTypeError("`bitsize` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.bitsize):
            ctrl, xs[i], ys[i] = bb.add(TwoBitCSwap(), ctrl=ctrl, x=xs[i], y=ys[i])

        return {'ctrl': ctrl, 'x': bb.join(xs), 'y': bb.join(ys)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TwoBitCSwap(), self.bitsize)}

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for CSwap classical simulation.")

    def short_name(self) -> str:
        return r'x↔y'

    @classmethod
    def make_on(
        cls, **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]]  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce bitsize attributes."""
        return cls(bitsize=len(quregs['x'])).on_registers(**quregs)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@",) + ("swap_x",) * self.bitsize + ("swap_y",) * self.bitsize
            )
        return cirq.CircuitDiagramInfo(("@",) + ("×(x)",) * self.bitsize + ("×(y)",) * self.bitsize)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('×(x)')
        elif soq.reg.name == 'y':
            return TextBox('×(y)')
        else:
            return Circle(filled=True)

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)

    def adjoint(self) -> 'Bloq':
        return self


@bloq_example
def _cswap_symb() -> CSwap:
    # A symbolic version. The bitsize is the symbol 'n'.
    from sympy import sympify

    cswap_symb = CSwap(bitsize=sympify('n'))
    return cswap_symb


@bloq_example(generalizer=ignore_split_join)
def _cswap_small() -> CSwap:
    # A small version on four bits.
    cswap_small = CSwap(bitsize=4)
    return cswap_small


@bloq_example(generalizer=ignore_split_join)
def _cswap_large() -> CSwap:
    # A large version that swaps 64-bit registers.
    cswap_large = CSwap(bitsize=64)
    return cswap_large


_CSWAP_DOC = BloqDocSpec(
    bloq_cls=CSwap,
    import_line='from qualtran.bloqs.basic_gates import CSwap',
    examples=(_cswap_symb, _cswap_small, _cswap_large),
)
