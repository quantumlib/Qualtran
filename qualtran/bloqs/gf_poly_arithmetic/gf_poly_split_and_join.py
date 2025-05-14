#  Copyright 2025 Google LLC
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
from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING

import galois
import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QGFPoly,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from pennylane.operation import Operation
    from pennylane.wires import Wires

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class GFPolySplit(_BookkeepingBloq):
    r"""Split a register representing coefficients of a polynomial into an array of `QGF` types.

    A register of type `QGFPoly` represents a univariate polynomial $f(x)$ with coefficients in a
    galois field GF($p^m$). Given an input quantum register representing a degree $n$ polynomial
    $f(x)$, this bloq splits it into $n + 1$ registers of type $QGF(p, m)$.

    Given a polynomial
    $$
        f(x) = \sum_{i = 0}^{n} a_{i} x^{i} \\ \forall a_{i} \in GF(p^m)
    $$

    the bloq splits it into a big-endian representation such that
    $$
        \ket{f(x)} \xrightarrow{\text{split}} \ket{a_{n}}\ket{a_{n - 1}} \cdots \ket{a_0}
    $$

    See `GFPolyJoin` for the inverse operation.

    Args:
        dtype: An instance of `QGFPoly` type that represents a degree $n$ polynomial defined
            over a galois field GF($p^m$).

    Registers:
        reg: The register to be split. On its left, it is of the type `qgf_poly`. On the right,
            it is an array of `QGF`s of shape `(qgf_poly.degree + 1,)`.
    """

    dtype: QGFPoly = field()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', self.dtype, shape=tuple(), side=Side.LEFT),
                Register('reg', self.dtype.qgf, shape=(self.dtype.degree + 1,), side=Side.RIGHT),
            ]
        )

    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if is_symbolic(value.degree):
            raise ValueError(f"{self} cannot have a symbolic data type.")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self) -> 'Bloq':
        return GFPolyJoin(dtype=self.dtype)

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {
            'reg': reg.reshape((int(self.dtype.degree) + 1, int(self.dtype.qgf.num_qubits)))
        }

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        return None

    def on_classical_vals(self, reg: galois.Poly) -> Dict[str, 'ClassicalValT']:
        return {'reg': self.dtype.to_gf_coefficients(reg)}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        incoming = incoming['reg']
        outgoing = cast(NDArray, outgoing['reg'])
        inp_inds = [(incoming, i) for i in range(int(self.dtype.num_qubits))]
        out_inds = [
            (outgoing[i], j)
            for i in range(int(self.dtype.degree) + 1)
            for j in range(int(self.dtype.qgf.num_qubits))
        ]
        assert len(inp_inds) == len(out_inds)

        return [
            qtn.Tensor(data=np.eye(2), inds=[out_inds[i], inp_inds[i]], tags=[str(self)])
            for i in range(int(self.dtype.num_qubits))
        ]

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.shape:
            text = f'[{", ".join(str(i) for i in idx)}]'
            return directional_text_box(text, side=reg.side)
        return directional_text_box(' ', side=reg.side)


@bloq_example
def _gf_poly_split() -> GFPolySplit:
    from qualtran import QGF, QGFPoly

    gf_poly_split = GFPolySplit(QGFPoly(4, QGF(2, 3)))
    return gf_poly_split


_GF_POLY_SPLIT_DOC = BloqDocSpec(
    bloq_cls=GFPolySplit, examples=[_gf_poly_split], call_graph_example=None
)


@frozen
class GFPolyJoin(_BookkeepingBloq):
    r"""Join $n+1$ registers representing coefficients of a polynomial into a `QGFPoly` type.

    A register of type `QGFPoly` represents a univariate polynomial $f(x)$ with coefficients in a
    galois field GF($p^m$). Given an input quantum register of shape (n + 1,) and type `QGF`
    representing coefficients of a degree $n$ polynomial $f(x)$, this bloq joins it into
    a register of type `QGFPoly`.

    Given a polynomial
    $$
        f(x) = \sum_{i = 0}^{n} a_{i} x^{i} \\ \forall a_{i} \in GF(p^m)
    $$

    the bloq joins registers representing coefficients of the polynomial in big-endian representation
    such that
    $$
        \ket{a_{n}}\ket{a_{n - 1}} \cdots \ket{a_0} \xrightarrow{\text{join}} \ket{f(x)}
    $$

    See `GFPolySplit` for the inverse operation.

    Args:
        dtype: An instance of `QGFPoly` type that represents a degree $n$ polynomial defined
            over a galois field GF($p^m$).

    Registers:
        reg: The register to be joined. On its left, it is an array of `QGF`s of shape
            `(qgf_poly.degree + 1,)`. On the right, it is of the type `qgf_poly`.

    """

    dtype: QGFPoly = field()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', self.dtype.qgf, shape=(self.dtype.degree + 1,), side=Side.LEFT),
                Register('reg', self.dtype, shape=tuple(), side=Side.RIGHT),
            ]
        )

    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if is_symbolic(value.degree):
            raise ValueError(f"{self} cannot have a symbolic data type.")

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    def adjoint(self) -> 'Bloq':
        return GFPolySplit(dtype=self.dtype)

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'reg': reg.reshape(int(self.dtype.num_qubits))}

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        return None

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        incoming = cast(NDArray, incoming['reg'])
        outgoing = outgoing['reg']

        inp_inds = [
            (incoming[i], j)
            for i in range(int(self.dtype.degree) + 1)
            for j in range(int(self.dtype.qgf.num_qubits))
        ]
        out_inds = [(outgoing, i) for i in range(int(self.dtype.num_qubits))]
        assert len(inp_inds) == len(out_inds)

        return [
            qtn.Tensor(data=np.eye(2), inds=[out_inds[i], inp_inds[i]], tags=[str(self)])
            for i in range(int(self.dtype.num_qubits))
        ]

    def on_classical_vals(self, reg: 'galois.Array') -> Dict[str, galois.Poly]:
        return {'reg': self.dtype.from_gf_coefficients(reg)}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.shape:
            text = f'[{", ".join(str(i) for i in idx)}]'
            return directional_text_box(text, side=reg.side)
        return directional_text_box(' ', side=reg.side)


@bloq_example
def _gf_poly_join() -> GFPolyJoin:
    from qualtran import QGF, QGFPoly

    gf_poly_join = GFPolyJoin(QGFPoly(4, QGF(2, 3)))
    return gf_poly_join


_GF_POLY_JOIN_DOC = BloqDocSpec(
    bloq_cls=GFPolyJoin, examples=[_gf_poly_join], call_graph_example=None
)
