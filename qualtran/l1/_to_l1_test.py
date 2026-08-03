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

import io

import pytest

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran import BloqBuilder, Register, Side
from qualtran.bloqs.basic_gates import CSwap, TGate
from qualtran.drawing import Circle
from qualtran.l1 import L1ASTPrinter, L1ModuleBuilder
from qualtran.l1._to_l1 import (
    bloq_to_ast,
    dump_l1,
    dump_root_l1,
    regs_to_sig_entry,
    signature_to_l1_entries,
)
from qualtran.l1.nodes import QDefExternNode, QDefImplNode, QSignatureEntry


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoDecompBloq(qlt.Bloq):
    """A leaf bloq that raises `DecomposeNotImplementedError`."""

    @property
    def signature(self) -> qlt.Signature:
        return qlt.Signature([qlt.Register('q', qdt.QBit())])


class _TypeErrorBloq(qlt.Bloq):
    """A bloq whose decomposition raises `DecomposeTypeError`."""

    @property
    def signature(self) -> qlt.Signature:
        return qlt.Signature([qlt.Register('q', qdt.QBit())])

    def build_composite_bloq(self, bb, **soqs):
        raise qlt.DecomposeTypeError("no decomposition for _TypeErrorBloq")


def _two_tgate_cbloq() -> qlt.CompositeBloq:
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(TGate(), q=q)
    q = bb.add(TGate(), q=q)
    return bb.finalize(q=q)


# ---------------------------------------------------------------------------
# regs_to_sig_entry / signature_to_l1_entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_regs', [0, 3])
def test_regs_to_sig_entry_bad_count(n_regs):
    regs = [Register(f'r{i}', qdt.QBit()) for i in range(n_regs)]
    with pytest.raises(ValueError, match='Bad regs'):
        regs_to_sig_entry('grp', regs)


def test_regs_to_sig_entry_bad_sides():
    # Two registers of the same non-THRU side is not a valid LEFT/RIGHT pair.
    regs = [
        Register('grp', qdt.QBit(), side=Side.RIGHT),
        Register('grp', qdt.QBit(), side=Side.RIGHT),
    ]
    with pytest.raises(ValueError, match='Bad register sides'):
        regs_to_sig_entry('grp', regs)


def test_regs_to_sig_entry_left_right_pair_orders_left_first():
    left = Register('grp', qdt.QBit(), side=Side.LEFT)
    right = Register('grp', qdt.QAny(2), side=Side.RIGHT)
    # Regardless of input order, the LEFT dtype is emitted first.
    for regs in ([left, right], [right, left]):
        entry = regs_to_sig_entry('grp', regs)
        assert isinstance(entry, QSignatureEntry)
        assert isinstance(entry.dtype, tuple)
        left_node, right_node = entry.dtype
        assert left_node is not None and right_node is not None
        assert left_node.dtype.name == 'QBit'
        assert right_node.dtype.name == 'QAny'


@pytest.mark.parametrize('side', [Side.THRU, Side.LEFT, Side.RIGHT])
def test_regs_to_sig_entry_single(side):
    entry = regs_to_sig_entry('grp', [Register('grp', qdt.QBit(), side=side)])
    assert isinstance(entry, QSignatureEntry)
    if side is Side.THRU:
        assert not isinstance(entry.dtype, tuple)
    else:
        assert isinstance(entry.dtype, tuple)


def test_signature_to_l1_entries():
    sig = qlt.Signature([Register('a', qdt.QBit()), Register('b', qdt.QAny(2))])
    entries = signature_to_l1_entries(sig)
    assert [e.name for e in entries] == ['a', 'b']


# ---------------------------------------------------------------------------
# bloq_to_ast fallbacks
# ---------------------------------------------------------------------------


def test_bloq_to_ast_decompose_not_implemented_becomes_extern():
    qdef_ctx, subbloqs = bloq_to_ast(_NoDecompBloq(), {}, extern_only_from=False)
    assert isinstance(qdef_ctx.qdef, QDefExternNode)
    assert subbloqs == []


def test_bloq_to_ast_decompose_type_error_becomes_extern():
    qdef_ctx, subbloqs = bloq_to_ast(_TypeErrorBloq(), {}, extern_only_from=False)
    assert isinstance(qdef_ctx.qdef, QDefExternNode)
    assert subbloqs == []


def test_bloq_to_ast_composite_bloq_is_implemented():
    cbloq = _two_tgate_cbloq()
    qdef_ctx, _ = bloq_to_ast(cbloq, {}, extern_only_from=False)
    # A CompositeBloq is emitted as an implemented qdef with no `from` object.
    assert isinstance(qdef_ctx.qdef, QDefImplNode)
    assert qdef_ctx.qdef.cobject_from is None


def test_bloq_to_ast_extern_only_from_has_no_from():
    qdef_ctx, _ = bloq_to_ast(CSwap(bitsize=2), {}, extern_only_from=True)
    assert isinstance(qdef_ctx.qdef, QDefImplNode)
    assert qdef_ctx.qdef.cobject_from is None


def test_force_extern_composite_bloq_warns():
    cbloq = _two_tgate_cbloq()
    with pytest.warns(UserWarning, match='Tried to `extern` a CompositeBloq'):
        qdef_ctx, _ = bloq_to_ast(cbloq, {}, extern_only_from=False, force_extern=True)
    assert isinstance(qdef_ctx.qdef, QDefExternNode)


# ---------------------------------------------------------------------------
# Public dump API
# ---------------------------------------------------------------------------


def test_dump_l1_returns_string():
    txt = dump_l1(CSwap(bitsize=2))
    assert isinstance(txt, str)
    assert txt.startswith('# Qualtran-L1')
    assert 'qdef CSwap' in txt


def test_dump_l1_writes_to_file_and_returns_root_key():
    buf = io.StringIO()
    root_key = dump_l1(CSwap(bitsize=2), f=buf)
    assert root_key == 'CSwap'
    assert 'qdef CSwap' in buf.getvalue()


def test_dump_l1_annotate_costs():
    txt = dump_l1(CSwap(bitsize=2), annotate_costs=True)
    assert isinstance(txt, str)
    assert 'qdef CSwap' in txt


def test_dump_root_l1_externs_everything_but_root():
    txt = dump_root_l1(CSwap(bitsize=2))
    # The root is implemented...
    assert 'qdef CSwap' in txt
    # ...and its subbloqs are externed.
    assert 'extern qdef' in txt


# ---------------------------------------------------------------------------
# L1ModuleBuilder.pretty_print_qdef / __str__
# ---------------------------------------------------------------------------


def test_pretty_print_qdef_to_file_and_stdout(capsys):
    l1_mb = L1ModuleBuilder()
    root_key = l1_mb.add_bloqs(CSwap(bitsize=2))

    # By bloq object, to a file.
    buf = io.StringIO()
    l1_mb.pretty_print_qdef(CSwap(bitsize=2), f=buf)
    assert 'qdef CSwap' in buf.getvalue()

    # By bloq key, to stdout.
    l1_mb.pretty_print_qdef(root_key)
    assert 'qdef CSwap' in capsys.readouterr().out


def test_pretty_print_qdef_unknown_bloq_raises():
    l1_mb = L1ModuleBuilder()
    l1_mb.add_bloqs(CSwap(bitsize=2))
    with pytest.raises(KeyError, match='Unknown bloq key'):
        l1_mb.pretty_print_qdef(TGate())


def test_module_builder_str():
    l1_mb = L1ModuleBuilder()
    l1_mb.add_bloqs(CSwap(bitsize=2))
    s = str(l1_mb)
    assert s.startswith('L1ModuleBuilder(')
    assert 'CSwap' in s
