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
from qualtran.qlt_ir import QltASTPrinter, QltModuleBuilder
from qualtran.qlt_ir._to_qlt_ir import (
    bloq_to_ast,
    dump_qlt_ir,
    dump_root_qlt_ir,
    Locals,
    QGlobals,
    regs_to_sig_entry,
    signature_to_qlt_ir_entries,
)
from qualtran.qlt_ir.nodes import QDefExternNode, QDefImplNode, QSignatureEntry


class MyBloq(qlt.Bloq):
    @property
    def signature(self) -> qlt.Signature:
        return qlt.Signature(
            [qlt.Register('ctrl', qdt.QBit()), qlt.Register('neg_ctrl', qdt.QBit())]
        )

    def wire_symbol(self, reg: qlt.Register | None, idx: tuple[int, ...] = ()):
        if reg is None:
            return super().wire_symbol(reg, idx)
        if reg.name == 'ctrl':
            return Circle(filled=False)
        elif reg.name == 'neg_ctrl':
            return Circle(filled=True)
        return super().wire_symbol(reg, idx)


def test_wire_symbol_annotations():
    bloq = MyBloq()

    # Default (include_annotations=False) omits annotations
    qlt_mb_default = QltModuleBuilder()
    qlt_mb_default.add_bloqs(bloq, force_extern_pred=lambda b: True)
    qlt_txt_default = QltASTPrinter().visit(qlt_mb_default.finalize())
    assert "@" not in qlt_txt_default

    # With include_annotations=True, wire symbol annotations are emitted
    qlt_mb = QltModuleBuilder()
    qlt_mb.add_bloqs(bloq, force_extern_pred=lambda b: True, include_annotations=True)
    qlt_mod = qlt_mb.finalize()
    qlt_txt = QltASTPrinter().visit(qlt_mod)

    assert "ctrl: QBit @ circle" in qlt_txt
    assert "neg_ctrl: QBit @ dot" in qlt_txt


class _RichWSBloq(qlt.Bloq):
    """Test bloq exercising cross, oplus, custom box labels, and shaped registers."""

    @property
    def signature(self) -> qlt.Signature:
        return qlt.Signature(
            [
                qlt.Register('default_reg', qdt.QBit()),
                qlt.Register('swap_reg', qdt.QBit()),
                qlt.Register('target_reg', qdt.QBit()),
                qlt.Register('labeled_reg', qdt.QBit()),
                qlt.Register('homog_ctrl', qdt.QBit(), shape=(2,)),
                qlt.Register('hetero_ctrl', qdt.QBit(), shape=(2,)),
                qlt.Register('mixed_arr', qdt.QBit(), shape=(2,)),
            ]
        )

    def wire_symbol(self, reg: qlt.Register | None, idx: tuple[int, ...] = ()):
        from qualtran.drawing import ModPlus, Text, TextBox

        if reg is None:
            return Text('')
        if reg.name == 'swap_reg':
            return TextBox('×')
        if reg.name == 'target_reg':
            return ModPlus()
        if reg.name == 'labeled_reg':
            return TextBox('H')
        if reg.name == 'homog_ctrl':
            return Circle(filled=True)
        if reg.name == 'hetero_ctrl':
            return Circle(filled=(idx == (0,)))
        if reg.name == 'mixed_arr':
            if idx == (0,):
                return TextBox('gamma')
            return super().wire_symbol(reg, idx)
        return super().wire_symbol(reg, idx)


def test_rich_wire_symbol_and_stave_call_annotations():
    from qualtran.qlt_ir._parse import parse_module

    bb = BloqBuilder()
    q0 = bb.add_register('q0', 1)
    q1 = bb.add_register('q1', 1)
    q2 = bb.add_register('q2', 1)
    q3 = bb.add_register('q3', 1)
    hc = bb.add_register(Register('hc', qdt.QBit(), shape=(2,)))
    htc = bb.add_register(Register('htc', qdt.QBit(), shape=(2,)))
    ma = bb.add_register(Register('ma', qdt.QBit(), shape=(2,)))
    assert hc is not None and htc is not None and ma is not None

    # _RichWSBloq returns Text('') for reg=None -> stave mode (@ (False,))
    q0, q1, q2, q3, hc, htc, ma = bb.add(
        _RichWSBloq(),
        default_reg=q0,
        swap_reg=q1,
        target_reg=q2,
        labeled_reg=q3,
        homog_ctrl=hc,
        hetero_ctrl=htc,
        mixed_arr=ma,
    )
    # MyBloq returns non-empty Text for reg=None -> schematic mode (no @ (False,))
    q0, q1 = bb.add(MyBloq(), ctrl=q0, neg_ctrl=q1)
    cbloq = bb.finalize(q0=q0, q1=q1, q2=q2, q3=q3, hc=hc, htc=htc, ma=ma)

    # Default dump_qlt_ir omits annotations
    qlt_txt_default = dump_qlt_ir(cbloq)
    assert qlt_txt_default is not None
    assert "@" not in qlt_txt_default

    qlt_txt = dump_qlt_ir(cbloq, include_annotations=True)
    assert qlt_txt is not None
    assert "default_reg: QBit," in qlt_txt or "default_reg: QBit]" in qlt_txt
    assert "swap_reg: QBit @ cross" in qlt_txt
    assert "target_reg: QBit @ oplus" in qlt_txt
    assert "labeled_reg: QBit @ box('H')" in qlt_txt
    assert "homog_ctrl: QBit[2] @ dot" in qlt_txt
    assert "hetero_ctrl: QBit[2] @ (dot, circle)" in qlt_txt
    assert "mixed_arr: QBit[2] @ (box('gamma'), box)" in qlt_txt
    assert "= _RichWSBloq @ (False,)" in qlt_txt
    assert "= MyBloq           [" in qlt_txt
    assert "= MyBloq @" not in qlt_txt

    # Annotated QLT IR should parse back cleanly into a valid QltModule AST.
    parsed = parse_module(qlt_txt)
    assert len(parsed.qdefs) == 3


def test_alloc_free_are_extern_not_qcast():
    """Allocate and Free are _BookkeepingBloqs but should be extern qdef, not qcast."""
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split
    from qualtran.qlt_ir._to_qlt_ir import bloq_to_ast
    from qualtran.qlt_ir.nodes import QCastNode, QDefExternNode

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
# regs_to_sig_entry / signature_to_qlt_ir_entries
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


def test_signature_to_qlt_ir_entries():
    sig = qlt.Signature([Register('a', qdt.QBit()), Register('b', qdt.QAny(2))])
    entries = signature_to_qlt_ir_entries(sig)
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


def test_dump_qlt_ir_returns_string():
    txt = dump_qlt_ir(CSwap(bitsize=2))
    assert isinstance(txt, str)
    assert txt.startswith('# QLT IR')
    assert 'qdef CSwap' in txt


def test_dump_qlt_ir_writes_to_file_and_returns_root_key():
    buf = io.StringIO()
    root_key = dump_qlt_ir(CSwap(bitsize=2), f=buf)
    assert root_key == 'CSwap'
    assert 'qdef CSwap' in buf.getvalue()


def test_dump_qlt_ir_annotate_costs():
    txt = dump_qlt_ir(CSwap(bitsize=2), annotate_costs=True)
    assert isinstance(txt, str)
    assert 'qdef CSwap' in txt


def test_dump_root_qlt_ir_externs_everything_but_root():
    txt = dump_root_qlt_ir(CSwap(bitsize=2))
    # The root is implemented...
    assert 'qdef CSwap' in txt
    # ...and its subbloqs are externed.
    assert 'extern qdef' in txt


# ---------------------------------------------------------------------------
# QltModuleBuilder.pretty_print_qdef / __str__
# ---------------------------------------------------------------------------


def test_pretty_print_qdef_to_file_and_stdout(capsys):
    qlt_mb = QltModuleBuilder()
    root_key = qlt_mb.add_bloqs(CSwap(bitsize=2))

    # By bloq object, to a file.
    buf = io.StringIO()
    qlt_mb.pretty_print_qdef(CSwap(bitsize=2), f=buf)
    assert 'qdef CSwap' in buf.getvalue()

    # By bloq key, to stdout.
    qlt_mb.pretty_print_qdef(root_key)
    assert 'qdef CSwap' in capsys.readouterr().out


def test_pretty_print_qdef_unknown_bloq_raises():
    qlt_mb = QltModuleBuilder()
    qlt_mb.add_bloqs(CSwap(bitsize=2))
    with pytest.raises(KeyError, match='Unknown bloq key'):
        qlt_mb.pretty_print_qdef(TGate())


def test_module_builder_str():
    qlt_mb = QltModuleBuilder()
    qlt_mb.add_bloqs(CSwap(bitsize=2))
    s = str(qlt_mb)
    assert s.startswith('QltModuleBuilder(')
    assert 'CSwap' in s


# ---------------------------------------------------------------------------
# Locals and QGlobals unique key generation tests
# ---------------------------------------------------------------------------


def test_locals_get_unique_name():
    locals_mgr = Locals()
    assert locals_mgr.get_unique_name('reg') == 'reg'
    assert locals_mgr.get_unique_name('reg') == 'reg2'
    assert locals_mgr.get_unique_name('reg') == 'reg3'

    # Independent prefix
    assert locals_mgr.get_unique_name('other') == 'other'
    assert locals_mgr.get_unique_name('other') == 'other2'

    # Pre-registered name handling
    locals_mgr.register_name('reg4')
    assert locals_mgr.get_unique_name('reg') == 'reg5'
    assert locals_mgr.get_unique_name('reg') == 'reg6'


def test_qglobals_unique_bloq_keys():
    import attrs

    @attrs.frozen
    class DummyBloq(qlt.Bloq):
        val: int

        @property
        def signature(self) -> qlt.Signature:
            return qlt.Signature([qlt.Register('q', qdt.QBit())])

        def __str__(self) -> str:
            return "DummyBloq()"

    qglobals = QGlobals()
    b1 = DummyBloq(val=1)
    b2 = DummyBloq(val=2)
    b3 = DummyBloq(val=3)

    k1 = qglobals.get_unique_bloq_key(b1)
    k2 = qglobals.get_unique_bloq_key(b2)
    k3 = qglobals.get_unique_bloq_key(b3)

    assert k1 == 'DummyBloq'
    assert k2 == 'DummyBloq(variant=1)'
    assert k3 == 'DummyBloq(variant=2)'

    # Re-querying existing bloq returns cached key
    assert qglobals.get_unique_bloq_key(b1) == 'DummyBloq'
    assert qglobals[b2] == 'DummyBloq(variant=1)'
    assert len(qglobals) == 3
    assert b3 in qglobals


def test_dump_qlt_ir_skip_aliases():
    # Long bloq name that would normally be aliased
    bloq = CSwap(bitsize=2)
    txt_with_aliases = dump_qlt_ir(bloq, skip_aliases=False)
    txt_without_aliases = dump_qlt_ir(bloq, skip_aliases=True)

    assert isinstance(txt_with_aliases, str)
    assert isinstance(txt_without_aliases, str)
    assert "alias" not in txt_without_aliases


def test_bloq_with_no_output_registers_emits_return():
    from qualtran.bloqs.basic_gates.qconst import QIntEffect
    from qualtran.qlt_ir import load_module

    bloq = QIntEffect(-5, bitsize=8)
    qlt_txt = dump_root_qlt_ir(bloq)
    assert "return" in qlt_txt
    loaded = load_module(qlt_txt)
    assert "QIntEffect(-5)" in loaded
