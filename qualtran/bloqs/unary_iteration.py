from functools import cached_property
from typing import *  # TODO
from typing import Any, Dict, Tuple, TYPE_CHECKING

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import cirq_infra
from cirq_qubitization.bloq_algos.and_bloq import And
from cirq_qubitization.bloq_algos.basic_gates import CNOT, XGate
from cirq_qubitization.bloq_algos.basic_gates.swap import CSwap
from cirq_qubitization.bloq_algos.set_constant import SetConstant
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.t_complexity_protocol import t_complexity, TComplexity

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT


def _unary_iteration_segtree(
    bb: CompositeBloqBuilder,
    add_selected: Callable[[CompositeBloqBuilder, int, Soquet], Soquet],
    ctrl: Soquet,
    selection: NDArray[Soquet],
    l: int,
    r: int,
    l_iter: int,
    r_iter: int,
) -> Tuple[Soquet, NDArray[Soquet]]:
    if l >= r_iter or l_iter >= r:
        # Range corresponding to this node is completely outside of iteration range.
        raise NotImplementedError()
        return ctrl, selection
    if l == (r - 1):
        assert len(selection) == 0, selection
        ctrl = add_selected(bb, l, ctrl)
        return ctrl, selection
    m = (l + r) >> 1
    if r_iter <= m:
        # Yield only left sub-tree.
        sq = selection[0]
        subsel = selection[1:]
        ctrl, subsel = _unary_iteration_segtree(
            bb, add_selected, ctrl, subsel, l, m, l_iter, r_iter
        )
        return ctrl, np.concatenate(([sq], subsel))
    if l_iter >= m:
        sq = selection[0]
        subsel = selection[1:]
        ctrl, subsel = _unary_iteration_segtree(
            bb, add_selected, ctrl, subsel, m, r, l_iter, r_iter
        )
        return ctrl, np.concatenate(([sq], subsel))

    sq = selection[0]
    subsel = selection[1:]
    (ctrl, sq), anc = bb.add(And(1, 0), ctrl=np.array([ctrl, sq]))
    anc, subsel = _unary_iteration_segtree(bb, add_selected, anc, subsel, l, m, l_iter, r_iter)
    ctrl, anc = bb.add(CNOT(), ctrl=ctrl, target=anc)
    anc, subsel = _unary_iteration_segtree(bb, add_selected, anc, subsel, m, r, l_iter, r_iter)
    ((ctrl, sq),) = bb.add(And(adjoint=True), ctrl=np.array([ctrl, sq]), target=anc)
    selection = np.concatenate(([sq], subsel))
    return ctrl, selection


@frozen
class IndexedBloq(Bloq):
    selection_bitsize: int
    target_bitsize: int
    bloqs: Tuple[Bloq, ...]
    target_reg_name: str

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(
            ctrl=1, selection=self.selection_bitsize, target=self.target_bitsize
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: Soquet, selection: Soquet, target: Soquet
    ) -> Dict[str, 'SoquetT']:
        def add_indexed_op(bb, i, ctrl):
            nonlocal target
            bloq = self.bloqs[i]
            ctrl, target = bb.add(bloq.controlled(), ctrl=ctrl, **{self.target_reg_name: target})
            return ctrl

        selection = bb.split(selection)
        ctrl, selection = _unary_iteration_segtree(
            bb=bb,
            add_selected=add_indexed_op,
            ctrl=ctrl,
            selection=selection,
            l=0,
            r=2**self.selection_bitsize,
            l_iter=0,
            r_iter=len(self.bloqs),
        )
        selection = bb.join(selection)

        return {'ctrl': ctrl, 'selection': selection, 'target': target}

    def short_name(self) -> str:
        return 'bloqs[i]'
