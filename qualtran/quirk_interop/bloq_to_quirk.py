import subprocess
from typing import Optional

from qualtran import Bloq, DecomposeTypeError, CompositeBloq
from qualtran.bloqs.bookkeeping import Join, Split
from qualtran.drawing import (
    ModPlus,
    Circle,
    LarrowTextBox,
    RarrowTextBox,
    LineManager,
    get_musical_score_data,
)
from qualtran.drawing.musical_score import _cbloq_musical_score


class SparseLineManager(LineManager):
    """
    LineManager which keeps partitioned line slots reserved for them until they need it again

    This implementation only supports partitions between an n-bit data type
    such as QAny(n) or QUInt(n) and an equivalent length-n QBit register.
    """

    def __init__(self, cbloq: CompositeBloq, max_n_lines: int = 100):
        super().__init__(max_n_lines)
        # Pre-layout pass with a plain LineManager, used only to infer Join/Split pairing.
        _, self.soq_assign, _ = _cbloq_musical_score(
            cbloq.signature, binst_graph=cbloq._binst_graph, manager=LineManager()
        )
        self._join_to_split_id = self._build_join_to_split_map()
        self._split_to_join_id = self._build_split_to_join_map()

    def _find_dual_on_line(self, line: int, start: int, dual_cls: Bloq):
        dual_candidates = [
            (rpos.seq_x, soq.binst.i)  # type: ignore[union-attr]
            for soq, rpos in self.soq_assign.items()
            if rpos.y == line and rpos.seq_x > start and soq.binst.bloq_is(dual_cls)
        ]
        if not dual_candidates:
            return None
        dual_candidates.sort(key=lambda x: x[0])
        return dual_candidates[0][1]

    def _build_join_to_split_map(self):
        join_to_split = {}
        for soq, rpos in self.soq_assign.items():
            if soq.binst.bloq_is(Join) and soq.idx == ():
                dual_id = self._find_dual_on_line(rpos.y, rpos.seq_x, Split)
                if dual_id is not None:
                    join_to_split[soq.binst.i] = dual_id  # type: ignore[union-attr]
        return join_to_split

    def _build_split_to_join_map(self):
        split_to_join = {}
        for soq, rpos in self.soq_assign.items():
            if soq.binst.bloq_is(Split) and soq.idx != ():
                dual_id = self._find_dual_on_line(rpos.y, rpos.seq_x, Join)
                if dual_id is not None:
                    split_to_join[soq.binst.i] = dual_id  # type: ignore[union-attr]
        return split_to_join

    def maybe_reserve(self, binst, reg, idx):
        # Reserve one slot so a partitioned wire can reclaim the same vertical region
        # at its dual Join/Split.
        if binst.bloq_is(Join) and reg.shape:
            dual_id = self._join_to_split_id.get(binst.i)
            self.reserve_n(1, lambda binst_to_check, reg_to_check: binst_to_check.i == dual_id)

        if binst.bloq_is(Split) and not reg.shape:
            dual_id = self._split_to_join_id.get(binst.i)
            self.reserve_n(1, lambda binst_to_check, reg_to_check: binst_to_check.i == dual_id)


handled_operations = {
    ModPlus(): '"X"',
    Circle(filled=True): '"•"',
    Circle(filled=False): '"◦"',
    LarrowTextBox(text='∧'): '"X"',
    RarrowTextBox(text='∧'): '"X"',
}


def composite_bloq_to_quirk(
    cbloq: CompositeBloq, line_manager: Optional[LineManager] = None, open_quirk: bool = False
) -> str:
    """Convert a CompositeBloq into a Quirk circuit URL."""
    if line_manager is None:
        line_manager = SparseLineManager(cbloq)

    msd = get_musical_score_data(cbloq, manager=line_manager)

    sparse_circuit = [(['1'] * (msd.max_y + 1)).copy() for _ in range(msd.max_x)]
    for soq in msd.soqs:
        try:
            gate = handled_operations[soq.symb]
            sparse_circuit[soq.rpos.seq_x][soq.rpos.y] = gate
        except KeyError:
            pass

    empty_col = ['1'] * (msd.max_y + 1)
    circuit = [col for col in sparse_circuit if col != empty_col]
    if circuit == []:
        raise ValueError(f"{cbloq} is an empty circuit")
    # deleting lines of the circuit which are not used (happens with partition)
    if circuit:
        num_lines = len(circuit[0])
        lines_to_keep = [i for i in range(num_lines) if any(col[i] != '1' for col in circuit)]
        circuit = [[col[i] for i in lines_to_keep] for col in circuit]

    quirk_url = "https://algassert.com/quirk"
    start = '#circuit={"cols":['
    end = ']}'
    url = quirk_url + start + ','.join('[' + ','.join(col) + ']' for col in circuit) + end

    if open_quirk:
        subprocess.run(["firefox", url], check=False)

    return url


def bloq_to_quirk(
    bloq: Bloq, line_manager: Optional[LineManager] = None, open_quirk: bool = False
) -> str:
    """Convert a Bloq into a Quirk circuit URL.

    The input bloq is decomposed and flattened before conversion. Only a limited set
    of operations is currently supported: control, anti-control, and NOT.

    Args:
        bloq: The bloq to export to Quirk.
        line_manager: Line manager used to assign and order circuit lines.
        open_quirk: If True, opens the generated URL in Firefox.

    Returns:
        A URL encoding the corresponding Quirk circuit.
    """
    try:
        cbloq = bloq.decompose_bloq().flatten()
    except DecomposeTypeError:  # no need to flatten the bloq if it is atomic
        cbloq = bloq.as_composite_bloq()

    return composite_bloq_to_quirk(cbloq, line_manager=line_manager, open_quirk=open_quirk)
