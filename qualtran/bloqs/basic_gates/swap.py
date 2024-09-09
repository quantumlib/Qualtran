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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.cirq_interop import CirqQuregT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import AddControlledT, CompositeBloq
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
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

    This is a Clifford operation.

    Registers:
        x: the first bit
        y: the second bit
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(x=1, y=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic.")

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', x: 'CirqQuregT', y: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        (x,) = x
        (y,) = y
        return cirq.SWAP.on(x, y), {'x': np.asarray([x]), 'y': np.asarray([y])}

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        matrix = _swap_matrix()
        out_inds = [(outgoing['x'], 0), (outgoing['y'], 0)]
        in_inds = [(incoming['x'], 0), (incoming['y'], 0)]
        return [qtn.Tensor(data=matrix, inds=out_inds + in_inds, tags=[str(self)])]

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'x': y, 'y': x}

    def adjoint(self) -> 'Bloq':
        return self

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        cswap = TwoBitCSwap()

        def adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl, x, y = bb.add(cswap, ctrl=ctrl, x=in_soqs['x'], y=in_soqs['y'])
            return [ctrl], [x, y]

        return cswap, adder


@bloq_example
def _swap_bit() -> TwoBitSwap:
    swap_bit = TwoBitSwap()
    return swap_bit


_TWO_BIT_SWAP_DOC = BloqDocSpec(bloq_cls=TwoBitSwap, examples=[_swap_bit], call_graph_example=None)


@frozen
class TwoBitCSwap(Bloq):
    """Swap two bits controlled on a control bit.

    This is sometimes known as the [Fredkin Gate](https://en.wikipedia.org/wiki/Fredkin_gate).

    Registers:
        ctrl: the control bit
        x: the first bit
        y: the second bit

    References:
        [An algorithm for the T-count](https://arxiv.org/abs/1308.4134).
        Gosset et. al. 2013. Figure 5.2.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=1, y=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic.")

    def to_clifford_t_circuit(self) -> 'cirq.FrozenCircuit':
        ctrl = cirq.NamedQubit('ctrl')
        x = cirq.NamedQubit('x')
        y = cirq.NamedQubit('y')
        circuit = cirq.Circuit()
        circuit += [cirq.CNOT(y, x)]
        circuit += [cirq.CNOT(ctrl, x), cirq.H(y)]
        circuit += [cirq.T(ctrl), cirq.T(x) ** -1, cirq.T(y)]
        circuit += [cirq.CNOT(y, x)]
        circuit += [cirq.CNOT(ctrl, y), cirq.T(x)]
        circuit += [cirq.CNOT(ctrl, x), cirq.T(y) ** -1]
        circuit += [cirq.T(x) ** -1, cirq.CNOT(ctrl, y)]
        circuit += [cirq.CNOT(y, x)]
        circuit += [cirq.T(x), cirq.H(y)]
        circuit += [cirq.CNOT(y, x)]
        return circuit.freeze()

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        matrix = _controlled_swap_matrix()
        out_inds = [(outgoing['ctrl'], 0), (outgoing['x'], 0), (outgoing['y'], 0)]
        in_inds = [(incoming['ctrl'], 0), (incoming['x'], 0), (incoming['y'], 0)]
        return [qtn.Tensor(data=matrix, inds=out_inds + in_inds, tags=[str(self)])]

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for TwoBitCSwap classical simulation.")

    def adjoint(self) -> 'Bloq':
        return self

    def wire_symbol(self, reg: Optional['Register'], idx: Tuple[int, ...] = ()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle(filled=True)
        else:
            return TextBox('×')


@bloq_example
def _cswap_bit() -> TwoBitCSwap:
    cswap_bit = TwoBitCSwap()
    return cswap_bit


_TWO_BIT_CSWAP_DOC = BloqDocSpec(
    bloq_cls=TwoBitCSwap, examples=[_cswap_bit], call_graph_example=None
)


@frozen
class Swap(Bloq):
    """Swap two registers

    This corresponds to a qubitwise `TwoBitSwap` on the two registers.

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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {TwoBitSwap(): self.bitsize}

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'x': y, 'y': x}

    def wire_symbol(self, reg: Optional['Register'], idx: Tuple[int, ...] = ()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'x':
            return TextBox('×(x)')
        elif reg.name == 'y':
            return TextBox('×(y)')
        raise ValueError(f"Bad register name {reg.name}")

    def adjoint(self) -> 'Bloq':
        return self

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        cswap = CSwap(self.bitsize)

        def adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl, x, y = bb.add(cswap, ctrl=ctrl, x=in_soqs['x'], y=in_soqs['y'])
            return [ctrl], [x, y]

        return cswap, adder


@bloq_example
def _swap() -> Swap:
    n = sympy.Symbol('n', positive=True, integer=True)
    swap = Swap(bitsize=n)
    return swap


@bloq_example(generalizer=ignore_split_join)
def _swap_small() -> Swap:
    swap_small = Swap(bitsize=4)
    return swap_small


@bloq_example
def _swap_large() -> Swap:
    swap_large = Swap(bitsize=64)
    return swap_large


_SWAP_DOC = BloqDocSpec(bloq_cls=Swap, examples=[_swap, _swap_small, _swap_large])


@frozen
class CSwap(GateWithRegisters):
    """Swap two registers controlled on a control bit.

    This decomposes into a qubitwise `TwoBitCSwap` on the two target registers,
    and takes $n$ TwoBitCSwap gates.

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
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise DecomposeTypeError("`bitsize` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.bitsize):
            ctrl, xs[i], ys[i] = bb.add(TwoBitCSwap(), ctrl=ctrl, x=xs[i], y=ys[i])

        return {'ctrl': ctrl, 'x': bb.join(xs), 'y': bb.join(ys)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {TwoBitCSwap(): self.bitsize}

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for CSwap classical simulation.")

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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'x':
            return TextBox('×(x)')
        elif reg.name == 'y':
            return TextBox('×(y)')
        else:
            return Circle(filled=True)

    def adjoint(self) -> 'Bloq':
        return self


@bloq_example
def _cswap() -> CSwap:
    n = sympy.Symbol('n', positive=True, integer=True)
    cswap = CSwap(bitsize=n)
    return cswap


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


_CSWAP_DOC = BloqDocSpec(bloq_cls=CSwap, examples=(_cswap, _cswap_small, _cswap_large))
