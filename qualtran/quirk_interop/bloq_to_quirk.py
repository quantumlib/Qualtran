import subprocess

from qualtran import Bloq, DecomposeTypeError
from qualtran.bloqs.bookkeeping import Join, Split
from qualtran.drawing import (
    ModPlus,
    Circle,
    LarrowTextBox,
    RarrowTextBox,
    LineManager,
    get_musical_score_data,
)


class SparseLineManager(LineManager):
    """
    LineManager which keeps partitioned line slots reserved for them until they need it again
    """

    # DIDN'TDO: only handles partition patterns of the type (QAny(n)/QUInt(n)/... -> QBit((n,)) or QBit((n,)) -> QAny(n))

    def maybe_reserve(self, binst, reg, idx):
        if binst.bloq_is(Join) and reg.shape:

            def _keep_split_lines(binst_to_check, reg_to_check):
                binst_cond = binst_to_check.bloq == binst.bloq.adjoint()
                reg_cond = reg_to_check == reg.adjoint()
                return binst_cond and reg_cond

            self.reserve_n(1, _keep_split_lines)

        if binst.bloq_is(Split) and not reg.shape:

            def _keep_joined_line(binst_to_check, reg_to_check):
                binst_cond = binst_to_check.bloq == binst.bloq.adjoint()
                reg_cond = reg_to_check == reg.adjoint()
                return binst_cond and reg_cond

            self.reserve_n(1, _keep_joined_line)


handled_operations = {
    ModPlus(): '"X"',
    Circle(filled=True): '"•"',
    Circle(filled=False): '"◦"',
    LarrowTextBox(text='∧'): '"X"',
    RarrowTextBox(text='∧'): '"X"',
}


def bloq_to_quirk(
    bloq: Bloq,
    line_manager: LineManager = SparseLineManager().__init__(),  # type: ignore[misc]
    open_quirk=False,
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
        flat_bloq = bloq.decompose_bloq().flatten()
    except DecomposeTypeError:  # no need to flatten the bloq if it is atomic
        flat_bloq = bloq.as_composite_bloq()
    msd = get_musical_score_data(flat_bloq, manager=line_manager)

    sparse_circuit = [(['1'] * (msd.max_y + 1)).copy() for _ in range(msd.max_x)]
    for soq in msd.json_dict()['soqs']:
        try:
            gate = handled_operations[soq.symb]
            sparse_circuit[soq.rpos.seq_x][soq.rpos.y] = gate
        except KeyError:
            None

    circuit = list(filter((['1'] * (msd.max_y + 1)).__ne__, sparse_circuit))
    nb_deleted_lines = 0
    for i in range(
        msd.max_y + 1
    ):  # deleting lines of the circuit which are not used (happens with partition)
        ind = i - nb_deleted_lines
        for col in circuit:
            line_is_useless = col[ind] == '1'
            if not line_is_useless:
                break
        if line_is_useless:
            for col in circuit:
                col.pop(ind)
            nb_deleted_lines += 1

    quirk_url = "https://algassert.com/quirk"
    start = '#circuit={"cols":['
    end = ']}'
    url = quirk_url + start + ','.join('[' + ','.join(col) + ']' for col in circuit) + end

    if open_quirk:
        subprocess.run(["firefox", url], check=False)

    return url
