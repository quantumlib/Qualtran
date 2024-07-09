#  Copyright 2024 Google LLC
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

from typing import cast

import numpy as np
import pytest

from qualtran import (
    Bloq,
    BloqBuilder,
    BoundedQUInt,
    Controlled,
    CtrlSpec,
    QBit,
    Register,
    Signature,
    Soquet,
)
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    Identity,
    IntEffect,
    IntState,
    OneEffect,
    OneState,
    TGate,
    XGate,
    ZeroEffect,
    ZeroState,
    ZGate,
)
from qualtran.bloqs.bookkeeping.arbitrary_clifford import ArbitraryClifford
from qualtran.bloqs.multiplexers.apply_lth_bloq import _apply_lth_bloq, ApplyLthBloq
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import assert_valid_bloq_decomposition


def test_apply_lth_bloq(bloq_autotester):
    bloq_autotester(_apply_lth_bloq)


def test_signature():
    assert _apply_lth_bloq().signature == Signature(
        [
            Register("control", QBit()),
            Register("selection", BoundedQUInt(2, 4)),
            Register("q", QBit()),
        ]
    )

    with pytest.raises(ValueError):

        def f(_):
            assert False

        _ = ApplyLthBloq(f, (0,))


def test_bloq_has_consistent_decomposition():
    assert_valid_bloq_decomposition(_apply_lth_bloq())


def test_call_graph():
    _, sigma = _apply_lth_bloq().call_graph(generalizer=ignore_split_join)
    assert sigma == {
        Controlled(Hadamard(), CtrlSpec()): 1,
        Controlled(TGate(), CtrlSpec()): 1,
        Controlled(ZGate(), CtrlSpec()): 1,
        CNOT(): 4,
        TGate(): 12,
        ArbitraryClifford(2): 45,
    }


@pytest.mark.parametrize("i", range(4))
@pytest.mark.parametrize("ctrl", [True, False])
def test_tensors(i, ctrl):
    bb = BloqBuilder()
    control = cast(Soquet, bb.add(OneState() if ctrl else ZeroState()))
    selection = cast(Soquet, bb.add(IntState(i, 2)))
    q = bb.add_register("q", 1)
    control, selection, q = bb.add_t(_apply_lth_bloq(), control=control, selection=selection, q=q)
    bb.add(OneEffect() if ctrl else ZeroEffect(), q=control)
    bb.add(IntEffect(i, 2), val=selection)
    bloq = bb.finalize(q=q)

    from_gate = ((TGate, Hadamard, ZGate, XGate)[i] if ctrl else Identity)().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


@pytest.mark.parametrize("i", range(2))
@pytest.mark.parametrize("j", range(2))
@pytest.mark.parametrize("ctrl", [True, False])
def test_ndim(i, j, ctrl):
    bb = BloqBuilder()
    control = bb.add(OneState() if ctrl else ZeroState())
    selection1 = bb.add(IntState(i, 1))
    selection2 = bb.add(IntState(j, 1))
    q = bb.add_register("q", 1)

    ops = np.array([[TGate(), Hadamard()], [ZGate(), XGate()]])

    def f(x: int, y: int) -> Bloq:
        return ops[x, y]

    bloq = ApplyLthBloq(f, (2, 2), control_val=1)

    assert bloq.signature == Signature(
        [
            Register("control", QBit()),
            Register("selection1", BoundedQUInt(1, 2)),
            Register("selection2", BoundedQUInt(1, 2)),
            Register("q", QBit()),
        ]
    )

    control, selection1, selection2, q = bb.add_t(
        bloq,
        control=cast(Soquet, control),
        selection1=cast(Soquet, selection1),
        selection2=cast(Soquet, selection2),
        q=q,
    )
    bb.add(OneEffect() if ctrl else ZeroEffect(), q=control)
    bb.add(IntEffect(i, 1), val=selection1)
    bb.add(IntEffect(j, 1), val=selection2)
    bloq = bb.finalize(q=q)

    from_gate = (
        (TGate, Hadamard, ZGate, XGate)[i * 2 + j] if ctrl else Identity
    )().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)
