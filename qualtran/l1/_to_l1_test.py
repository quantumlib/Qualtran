#  Copyright 2026 Google LLC
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

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.drawing import Circle
from qualtran.l1 import L1ASTPrinter, L1ModuleBuilder


class MyBloq(qlt.Bloq):
    @property
    def signature(self) -> qlt.Signature:
        return qlt.Signature(
            [qlt.Register('ctrl', qdt.QBit()), qlt.Register('neg_ctrl', qdt.QBit())]
        )

    def wire_symbol(self, reg: qlt.Register):
        if reg.name == 'ctrl':
            return Circle(filled=False)
        elif reg.name == 'neg_ctrl':
            return Circle(filled=True)
        return super().wire_symbol(reg)


def test_wire_symbol_annotations():
    bloq = MyBloq()
    l1_mb = L1ModuleBuilder()
    l1_mb.add_bloqs(bloq, force_extern_pred=lambda b: True)
    l1_mod = l1_mb.finalize()
    l1_txt = L1ASTPrinter().visit(l1_mod)

    assert "ctrl: QBit @ circle" in l1_txt
    assert "neg_ctrl: QBit @ dot" in l1_txt


def test_alloc_free_are_extern_not_qcast():
    """Allocate and Free are _BookkeepingBloqs but should be extern qdef, not qcast."""
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split
    from qualtran.l1._to_l1 import bloq_to_ast
    from qualtran.l1.nodes import QCastNode, QDefExternNode

    # Allocate and Free should produce QDefExternNode
    for bloq in [
        Allocate(qdt.QBit()),
        Free(qdt.QBit()),
        Allocate(qdt.QUInt(4)),
        Free(qdt.QUInt(4)),
    ]:
        qdef_ctx, _ = bloq_to_ast(bloq, {}, extern_only_from=False)
        assert isinstance(
            qdef_ctx.qdef, QDefExternNode
        ), f'{bloq} should be QDefExternNode, got {type(qdef_ctx.qdef).__name__}'

    # True casting bloqs should still produce QCastNode
    for bloq in [Cast(qdt.QUInt(4), qdt.QUInt(4)), Split(qdt.QUInt(4)), Join(qdt.QUInt(4))]:
        qdef_ctx, _ = bloq_to_ast(bloq, {}, extern_only_from=False)
        assert isinstance(
            qdef_ctx.qdef, QCastNode
        ), f'{bloq} should be QCastNode, got {type(qdef_ctx.qdef).__name__}'
