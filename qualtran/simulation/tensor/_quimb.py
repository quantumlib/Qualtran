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
import logging
from typing import cast, Dict, Iterable

import numpy as np
import quimb.tensor as qtn

from qualtran import (
    Bloq,
    CompositeBloq,
    Connection,
    LeftDangle,
    QBit,
    Register,
    RightDangle,
    Soquet,
    SoquetT,
)
from qualtran._infra.composite_bloq import _cxns_to_cxn_dict, BloqBuilder

logger = logging.getLogger(__name__)


def cbloq_to_quimb(cbloq: CompositeBloq) -> qtn.TensorNetwork:
    """Convert a composite bloq into a tensor network.

    This function will call `Bloq.my_tensors` on each subbloq in the composite bloq to add
    tensors to a quimb tensor network. This method has no default fallback, so you likely want to
    call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
    smallest form first. The small bloqs that result from a flattening 1) likely already have
    their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
    path.
    """
    tn = qtn.TensorNetwork([])

    logging.info(
        "Constructing a tensor network for composite bloq of size %d", len(cbloq.bloq_instances)
    )

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxns_to_cxn_dict(bloq.signature.lefts(), pred_cxns, get_me=lambda cxn: cxn.right)
        out_d = _cxns_to_cxn_dict(bloq.signature.rights(), succ_cxns, get_me=lambda cxn: cxn.left)

        for tensor in bloq.my_tensors(inc_d, out_d):
            tn.add(tensor)

    # Special case: Add variables corresponding to all registers that don't connect to any Bloq.
    # This is needed because `CompositeBloq.iter_bloqnections` ignores `LeftDangle/RightDangle`
    # bloqs, and therefore we never see connections that exist only b/w LeftDangle and
    # RightDangle bloqs.
    for cxn in cbloq.connections:
        if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:
            # This register has no Bloq acting on it, and thus it would not have a variable in
            # the tensor network. Add an identity tensor acting on this register to make sure the
            # tensor network has variables corresponding to all input / output registers.

            n = cxn.left.reg.bitsize
            for j in range(cxn.left.reg.bitsize):

                placeholder = Soquet(None, Register('simulation_placeholder', QBit()))  # type: ignore
                Connection(cxn.left, placeholder)
                tn.add(
                    qtn.Tensor(
                        data=np.eye(2),
                        inds=[
                            (Connection(cxn.left, placeholder), j),
                            (Connection(placeholder, cxn.right), j),
                        ],
                    )
                )

    return tn


def _add_classical_kets(bb: BloqBuilder, registers: Iterable[Register]) -> Dict[str, 'SoquetT']:
    """Use `bb` to add `IntState(0)` for all the `vals`."""

    from qualtran.bloqs.basic_gates import IntState

    soqs: Dict[str, 'SoquetT'] = {}
    for reg in registers:
        if reg.shape:
            reg_vals = np.zeros(reg.shape, dtype=int)
            soq = np.empty(reg.shape, dtype=object)
            for idx in reg.all_idxs():
                soq[idx] = bb.add(IntState(val=cast(int, reg_vals[idx]), bitsize=reg.bitsize))
        else:
            soq = bb.add(IntState(val=0, bitsize=reg.bitsize))

        soqs[reg.name] = soq
    return soqs


def initialize_from_zero(bloq: Bloq):
    """Take `bloq` and compose it with initial zero states for each left register.

    This can be contracted to a state vector for a given unitary.
    """
    bb = BloqBuilder()

    # Add the initial 'kets' according to the provided values.
    in_soqs = _add_classical_kets(bb, bloq.signature.lefts())

    # Add the bloq itself
    out_soqs = bb.add_d(bloq, **in_soqs)
    return bb.finalize(**out_soqs)
