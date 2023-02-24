import random
from functools import cached_property
from typing import Callable, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple, Union

import attrs
import cirq
import numpy as np
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _binst_to_cxns,
    BloqBuilderError,
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.meta_bloq import TestSerialBloq
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)
from cirq_qubitization.quantum_graph.util_bloqs import Join, Partition, Split, Unpartition


def _cxn_to_soq_dict(
    regs: Iterable[FancyRegister],
    cxns: Iterable[Connection],
    get_me: Callable[[Connection], Soquet],
    get_assign: Callable[[Connection], Soquet],
) -> Dict[str, SoquetT]:
    """Helper function to get a dictionary of incoming or outgoing soquets.

    This is used in `cbloq_to_quimb`.

    Args:
        regs: Left or right registers (used as a reference to initialize multidimensional
            registers correctly).
        cxns: Predecessor or successor connections from which we get the soquets of interest.
        get_me: A function that says which soquet is used to derive keys for the returned
            dictionary. Generally: if `cxns` is predecessor connections, this will return the
            `right` element of the connection and opposite of successor connections.
        get_assign: A function that says which soquet is used to dervice the values for the
            returned dictionary. Generally, this is the opposite side vs. `get_me`, but we
            do something fancier in `cbloq_to_quimb`.
    """
    soqdict: Dict[str, SoquetT] = {}

    # Initialize multi-dimensional dictionary values.
    for reg in regs:
        if reg.wireshape:
            soqdict[reg.name] = np.empty(reg.wireshape, dtype=object)

    # In the abstract: set `soqdict[me] = assign`. Specifically: use the register name as
    # keys and handle multi-dimensional registers.
    for cxn in cxns:
        me = get_me(cxn)
        assign = get_assign(cxn)

        if me.reg.wireshape:
            soqdict[me.reg.name][me.idx] = assign
        else:
            soqdict[me.reg.name] = assign

    return soqdict


def map_soq(soq: Soquet, mapping) -> Soquet:
    if isinstance(soq.binst, DanglingT):
        return soq
    return attrs.evolve(soq, binst=mapping[soq.binst])


def re_add():
    bloq = TestSerialBloq().decompose_bloq()

    regs = bloq.registers
    # .. modify regs ..

    bb = CompositeBloqBuilder(regs)
    soqs: Dict[str, SoquetT] = bb.initial_soquets()

    for binst, pred, succ in bloq.iter_bloqnections():
        bloq = binst.bloq

        in_soqs = _cxn_to_soq_dict(
            bloq.registers.lefts(),
            pred,
            get_me=lambda cxn: cxn.right,
            get_assign=lambda cxn: map_soq(cxn.left),
        )
        out_d = _cxn_to_soq_dict(
            bloq.registers.rights(),
            succ,
            get_me=lambda cxn: cxn.left,
            get_assign=lambda cxn: cxn.right,
        )

        ret_soqs = bb.add(binst.bloq, **in_soqs)


def add(
    registers: FancyRegisters,
    in_soqs: Dict[str, SoquetT],
    debug_name: str,  # for error messages
    _available: Set,  # for bookkeeping
) -> Tuple[SoquetT, ...]:
    def get_my_in_soq(reg, li):
        return Soquet(binst, reg, idx=li)

    def get_my_in_soq(reg, li):
        # Find soquet with register and index in first_cxns
        for xx in first_cxns:
            if xx.left.reg == reg and xx.left.idx == li:
                return xx.right
        raise ValueError()

    def get_my_out_soq(reg, ri):
        return Soquet(binst, reg, idx=ri)

    def get_my_out_soq(reg, ri):
        # Find soquet with register and index in first_cxns
        for xx in final_cxns:
            if xx.right.reg == reg and xx.right.idx == ri:
                return xx.left
        raise ValueError()

    _cxns = []

    for reg in registers.lefts():
        try:
            # if we want fancy indexing (which we do), we need numpy
            # this also supports length-zero indexing natively, which is good too.
            in_soq = np.asarray(in_soqs[reg.name])
        except KeyError:
            raise BloqBuilderError(
                f"{debug_name} requires an input Soquet named `{reg.name}`."
            ) from None

        del in_soqs[reg.name]  # so we can check for surplus arguments.

        for li in reg.wire_idxs():
            idxed_soq = in_soq[li]
            assert isinstance(idxed_soq, Soquet), idxed_soq
            try:
                _available.remove(idxed_soq)
            except KeyError:
                raise BloqBuilderError(
                    f"{idxed_soq} is not an available input Soquet for {reg}."
                ) from None

            my_in_soq = get_my_in_soq(reg, li)
            cxn = Connection(idxed_soq, my_in_soq)
            _cxns.append(cxn)

    if in_soqs:
        raise BloqBuilderError(
            f"{debug_name} does not accept input Soquets: {in_soqs.keys()}."
        ) from None

    do_nonesense()

    out_soqs: List[SoquetT] = []
    out: SoquetT
    for reg in registers.rights():
        if reg.wireshape:
            out = np.empty(reg.wireshape, dtype=object)
            for ri in reg.wire_idxs():
                out_soq = get_my_out_soq(reg, ri)
                out[ri] = out_soq
                _available.add(out_soq)
        else:
            # Annoyingly, the 0-dim case must be handled seprately.
            # Otherwise, x[i] = thing will nest *array* objects.
            out = get_my_out_soq(reg, tuple())
            _available.add(out)

        out_soqs.append(out)

    return tuple(out_soqs)
