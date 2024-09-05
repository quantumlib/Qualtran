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

from collections import Counter
from functools import cached_property
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cirq
from attrs import frozen

from .composite_bloq import _binst_to_cxns, _cxns_to_soq_dict, _map_soqs, _reg_to_soq, BloqBuilder
from .gate_with_registers import GateWithRegisters
from .quantum_graph import LeftDangle, RightDangle
from .registers import Signature

if TYPE_CHECKING:
    from qualtran import Bloq, CompositeBloq, Register, Signature, SoquetT
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _adjoint_final_soqs(cbloq: 'CompositeBloq', new_signature: Signature) -> Dict[str, 'SoquetT']:
    """`CompositeBloq.final_soqs()` but backwards."""
    if LeftDangle not in cbloq._binst_graph:
        return {}
    _, init_succs = _binst_to_cxns(LeftDangle, binst_graph=cbloq._binst_graph)
    return _cxns_to_soq_dict(
        new_signature.rights(), init_succs, get_me=lambda x: x.left, get_assign=lambda x: x.right
    )


def _adjoint_cbloq(cbloq: 'CompositeBloq') -> 'CompositeBloq':
    """Automatically derive the adjoint of `cbloq`.

    The adjoint of a composite bloq is another composite bloq where the order of
    operations is reversed and each subbloq is replaced with its adjoint.

    This is the implementation of `CompositeBloq.adjoint()`.

    Note that this function implementation is analogous to `CompositeBloq.copy()` but with the
    order of most things reversed. First, we reverse the registers to initialize the BloqBuilder.
    Then we reverse the order of subbloqs. And when we add the subbloq back in, we call
    `.adjoint()` on it.
    """
    # First, we reverse the registers to initialize the BloqBuilder.
    old_signature = cbloq.signature
    new_signature = cbloq.signature.adjoint()
    old_i_soqs = [_reg_to_soq(RightDangle, reg) for reg in old_signature.rights()]
    new_i_soqs = [_reg_to_soq(LeftDangle, reg) for reg in new_signature.lefts()]
    soq_map: List[Tuple[SoquetT, SoquetT]] = list(zip(old_i_soqs, new_i_soqs))

    # Then we reverse the order of subbloqs
    bloqnections = reversed(list(cbloq.iter_bloqnections()))

    # And add subbloq.adjoint() back in for each subbloq.
    bb, _ = BloqBuilder.from_signature(new_signature)
    for binst, preds, succs in bloqnections:
        # Instead of get_me returning the right element of a predecessor connection,
        # it's the left element of a successor connection.
        soqs = _cxns_to_soq_dict(
            binst.bloq.signature.rights(),
            succs,
            get_me=lambda x: x.left,
            get_assign=lambda x: x.right,
        )
        soqs = _map_soqs(soqs, soq_map)

        old_o_soqs = tuple(_reg_to_soq(binst, reg) for reg in binst.bloq.signature.lefts())
        new_o_soqs = bb.add_t(binst.bloq.adjoint(), **soqs)
        soq_map.extend(zip(old_o_soqs, new_o_soqs))

    # Instead of finalizing with RightDangle predecessors, we use LeftDangle successors
    fsoqs = _map_soqs(_adjoint_final_soqs(cbloq, new_signature), soq_map)
    return bb.finalize(**fsoqs)


@frozen
class Adjoint(GateWithRegisters):
    """The standard adjoint of `subbloq`.

    This metabloq generally delegates all of its protocols (with modifications, read on) to
    `subbloq`. This class is used in the default implementation of the adjoint protocol, i.e.,
    in the default implementation of `Bloq.adjoint()`.

    This metabloq is appropriate in most cases since there rarely a specialized
    (one level) decomposition for a bloq's adjoint. Exceptions can be found for decomposing
    some low-level primitives, for example `And`. Even if you use bloqs with specialized
    adjoints in your decomposition (i.e. you use `And`), you can still rely on this standard
    behavior.

    This bloq is defined entirely in terms of how it delegates its protocols. The following
    protocols delegate to `subbloq` (with appropriate modification):

     - **Signature**: The signature is the adjoint of `subbloqs`'s signature. Namely, LEFT
       and RIGHT registers are swapped.
     - **Decomposition**: The decomposition is the adjoint of `subbloq`'s decomposition. Namely,
       the order of operations in the resultant `CompositeBloq` is reversed and each bloq is
       replaced with its adjoint.
     - **Adjoint**: The adjoint of an `Adjoint` bloq is the subbloq itself.
     - **Call graph**: The call graph is the subbloq's call graph, but each bloq is replaced
       with its adjoint.
     - **Cirq Interop**: The default `Bloq` implementation is used, which goes via `BloqAsCirqGate`
       as usual.
     - **Wire Symbol**: The wire symbols are the adjoint of `subbloq`'s wire symbols. Namely,
       left- and right-oriented symbols are flipped.
     - **Names**: The string names / labels are that of the `subbloq` with a dagger symbol appended.

    Some protocols are impossible to delegate specialized implementations. The `Adjoint` bloq
    supports the following protocols with "decompose-only" implementations. This means we always
    go via the bloq's decomposition instead of preferring specialized implementations provided by
    the bloq author. If a specialized implementation of these protocols are required or you
    are trying to represent an adjoint bloq without a decomposition and need to support these
    protocols, use a specialized adjoint bloq or attribute instead of this class.

     - Classical simulation is "decompose-only". It is impossible to invert a generic python
       function.
     - Tensor simulation is "decompose-only" due to technical details around the Quimb interop.

    Args:
        subbloq: The bloq to wrap.
    """

    subbloq: 'Bloq'

    @cached_property
    def signature(self) -> 'Signature':
        """The signature is the adjoint of `subbloq`'s signature."""
        return self.subbloq.signature.adjoint()

    def decompose_bloq(self) -> 'CompositeBloq':
        """The decomposition is the adjoint of `subbloq`'s decomposition."""
        return self.subbloq.decompose_bloq().adjoint()

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> cirq.CircuitDiagramInfo:
        sub_info = cirq.circuit_diagram_info(self.subbloq, args, default=NotImplemented)
        if sub_info is NotImplemented:
            return NotImplemented
        sub_info.exponent *= -1
        return sub_info

    def adjoint(self) -> 'Bloq':
        """The 'double adjoint' brings you back to the original bloq."""
        return self.subbloq

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """The call graph takes the adjoint of each of the bloqs in `subbloq`'s call graph."""
        sub_cg = self.subbloq.build_call_graph(ssa=ssa)
        counts = Counter['Bloq']()
        if isinstance(sub_cg, set):
            for bloq, n in sub_cg:
                counts[bloq.adjoint()] += n
        else:
            for bloq, n in sub_cg.items():
                counts[bloq.adjoint()] += n
        return counts

    def pretty_name(self) -> str:
        """The subbloq's pretty_name with a dagger."""
        return self.subbloq.pretty_name() + '†'

    def __str__(self) -> str:
        """Delegate to subbloq's `__str__` method."""
        return f'Adjoint(subbloq={str(self.subbloq)})'

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        # Note: since we pass are passed a soquet which has the 'new' side, we flip it before
        # delegating and then flip back. Subbloqs only have to answer this protocol
        # if the provided soquet is facing the correct direction.
        if reg is None:
            return self.subbloq.wire_symbol(reg=None).adjoint()

        return self.subbloq.wire_symbol(reg=reg.adjoint(), idx=idx).adjoint()

    def _t_complexity_(self):
        """The cirq-style `_t_complexity_` delegates to the subbloq's method with a special shim.

        The cirq-style t complexity protocol does not leverage the heirarchical decomposition
        of high-level bloqs, so we need to shim in an extra `adjoint` boolean flag.
        """
        # TODO: https://github.com/quantumlib/Qualtran/issues/735
        if not hasattr(self.subbloq, '_t_complexity_'):
            return NotImplemented

        try:
            return self.subbloq._t_complexity_(adjoint=True)  # type: ignore[call-arg]
        except TypeError as e:
            if 'adjoint' in str(e):
                return self.subbloq._t_complexity_()
            else:
                raise e
