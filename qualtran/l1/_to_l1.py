#  Copyright 2025 Google LLC
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
from __future__ import annotations

import io
import itertools
import logging
import uuid
import warnings
from typing import (
    Callable,
    cast,
    Container,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeAlias,
    Union,
)

import attrs
import numpy as np

import qualtran as qlt
from qualtran._infra.binst_graph_iterators import greedy_topological_sort
from qualtran._infra.composite_bloq import _binst_to_cxns, _cxns_to_soq_dict
from qualtran._infra.quantum_graph import _Soquet

from . import nodes as qualtran_l1_nodes
from ._dtypes import reg_to_qdtype_node
from ._to_cobject_node import to_cobject_node, unserializable_object
from .nodes import L1Nodes

if TYPE_CHECKING:
    from .nodes import (
        CObjectNode,
        L1Module,
        QArgNode,
        QArgValueNode,
        QDefNode,
        QSignatureEntry,
        StatementNode,
    )

BloqKey: TypeAlias = str

log = logging.getLogger(__name__)


@attrs.frozen
class QDefWithContext:
    qdef: QDefNode
    bloq: qlt.Bloq
    implemented: bool
    extern_reason: Optional[str] = None

    @property
    def bloq_key(self) -> BloqKey:
        return self.qdef.bloq_key


@attrs.mutable
class Locals:
    """Handles assigning variable names for our variable *objects*, i.e. soquets and bloqs."""

    soqvars: Dict[_Soquet, QArgValueNode] = attrs.field(factory=dict)
    bloqvars: Dict[qlt.Bloq, BloqKey] = attrs.field(factory=dict)
    varnames: Set[str] = attrs.field(factory=set)
    nodes: L1Nodes = attrs.field(default=qualtran_l1_nodes)

    def get_unique_name(self, prefix: str) -> str:
        """Get and register a unique name."""
        candidate = f'{prefix}'
        i = 1
        while True:
            if candidate not in self.varnames:
                self.varnames.add(candidate)
                return candidate
            i += 1
            candidate = f'{prefix}{i}'

    def register_name(self, name: str):
        """Register a specific name, so we don't accidentally re-use it."""
        if name in self.varnames:
            raise ValueError(f"Cannot register {name}, it already exists!")
        self.varnames.add(name)

    def assign_soq(self, soq: _Soquet, prefix: str) -> str:
        name = self.get_unique_name(prefix)
        self.soqvars[soq] = self.nodes.QArgValueNode(name, ())
        return name

    def assign_bloq(self, bloq: qlt.Bloq, prefix: str) -> str:
        name = self.get_unique_name(prefix=prefix)
        self.bloqvars[bloq] = name
        return name


def regs_to_sig_entry(
    reg_name: str, regs: Sequence['qlt.Register'], *, nodes: L1Nodes = qualtran_l1_nodes
) -> QSignatureEntry:
    """Convert a register group (from `Signature.groups()`) to a `QSignatureEntry`.

    This is the pure conversion logic without any side effects such as
    annotation or local-variable registration.

    Args:
        reg_name: The register group name.
        regs: The 1- or 2-element sequence of registers for this group.
        nodes: The module providing the AST node constructors.
    """
    if len(regs) not in [1, 2]:
        raise ValueError(f'Bad regs for {reg_name}')

    if len(regs) == 2:
        r1, r2 = regs
        if r1.side is qlt.Side.LEFT and r2.side is qlt.Side.RIGHT:
            return nodes.QSignatureEntry(
                reg_name, (reg_to_qdtype_node(r1, nodes=nodes), reg_to_qdtype_node(r2, nodes=nodes))
            )
        elif r2.side is qlt.Side.LEFT and r1.side is qlt.Side.RIGHT:
            return nodes.QSignatureEntry(
                reg_name, (reg_to_qdtype_node(r2, nodes=nodes), reg_to_qdtype_node(r1, nodes=nodes))
            )
        raise ValueError(f'Bad register sides for {reg_name}: {regs}')

    (r,) = regs
    if r.side is qlt.Side.THRU:
        return nodes.QSignatureEntry(reg_name, reg_to_qdtype_node(r, nodes=nodes))
    elif r.side is qlt.Side.RIGHT:
        return nodes.QSignatureEntry(reg_name, (None, reg_to_qdtype_node(r, nodes=nodes)))
    elif r.side is qlt.Side.LEFT:
        return nodes.QSignatureEntry(reg_name, (reg_to_qdtype_node(r, nodes=nodes), None))
    else:
        raise ValueError(f'Bad side for {r}')


def signature_to_l1_entries(
    signature: 'qlt.Signature', *, nodes: L1Nodes = qualtran_l1_nodes
) -> List[QSignatureEntry]:
    """Convert a `qualtran.Signature` to a list of `QSignatureEntry` AST nodes.

    This is a convenience wrapper around `regs_to_sig_entry` for all register
    groups in the signature.
    """
    return [regs_to_sig_entry(reg_name, regs, nodes=nodes) for reg_name, regs in signature.groups()]


@attrs.mutable
class QDefBuilder:
    bloq: qlt.Bloq
    bloq_key: str
    qglobals: Dict[qlt.Bloq, str]
    nodes: L1Nodes = attrs.field(default=qualtran_l1_nodes)
    qlocals: Locals = attrs.field(factory=Locals)

    _sig_entries: List[QSignatureEntry] = attrs.field(factory=list)
    _stmnts: List[StatementNode] = attrs.field(factory=list)

    def __attrs_post_init__(self):
        if self.qlocals.nodes is qualtran_l1_nodes and self.nodes is not qualtran_l1_nodes:
            self.qlocals.nodes = self.nodes

    def _get_sig_entry_annotation(self, reg: 'qlt.Register') -> Optional[CObjectNode]:
        """Determine annotation based on wire symbol."""
        from qualtran.drawing import Circle, ModPlus

        if reg.shape:
            # TODO: Support for shaped registers.
            return None

        annotation: Optional[CObjectNode] = None
        symbol = self.bloq.wire_symbol(reg)
        if isinstance(symbol, Circle):
            if symbol.filled:
                annotation = self.nodes.CObjectNode('dot', cargs=())
            else:
                annotation = self.nodes.CObjectNode('circle', cargs=())
        elif isinstance(symbol, ModPlus):
            annotation = self.nodes.CObjectNode('oplus', cargs=())
        return annotation

    def _get_sig_entry(self, reg_name: str, regs: Sequence['qlt.Register']) -> QSignatureEntry:
        entry = regs_to_sig_entry(reg_name, regs, nodes=self.nodes)

        # Layer on annotation from wire symbols
        # TODO: Handle the case for different r1 vs r2
        annotation = self._get_sig_entry_annotation(regs[0])
        if annotation is not None:
            entry = attrs.evolve(entry, annotation=annotation)

        # Register local variable names for LEFT/THRU registers
        for r in regs:
            if r.side in (qlt.Side.LEFT, qlt.Side.THRU):
                self.qlocals.register_name(r.name)

        return entry

    def add_signature(self, signature: 'qlt.Signature') -> None:
        for reg_name, regs in signature.groups():
            entry = self._get_sig_entry(reg_name, regs)
            self._sig_entries.append(entry)

    def finalize_extern(self, reason: str = ''):
        cobject_from: CObjectNode
        if isinstance(self.bloq, qlt.CompositeBloq):
            warnings.warn("Tried to `extern` a CompositeBloq. You will not be able to load this.")
            cobject_from = unserializable_object('CompositeBloq', nodes=self.nodes)
        else:
            cobject_from = to_cobject_node(self.bloq, nodes=self.nodes)  # type: ignore[assignment]
        return QDefWithContext(
            qdef=self.nodes.QDefExternNode(
                bloq_key=self.bloq_key, qsignature=self._sig_entries, cobject_from=cobject_from
            ),
            bloq=self.bloq,
            implemented=False,
            extern_reason=reason,
        )

    def finalize_qcast(self) -> QDefWithContext:
        """Finalize as a qcast (casting operation) definition."""
        return QDefWithContext(
            qdef=self.nodes.QCastNode(bloq_key=self.bloq_key, qsignature=self._sig_entries),
            bloq=self.bloq,
            implemented=False,
            extern_reason='qcast',
        )

    def add_alias_assignment_heuristically(self, bloq: qlt.Bloq) -> None:

        if bloq not in self.qglobals:
            bloq_key = _get_unique_bloq_key(bloq, self.qglobals.values(), nodes=self.nodes)
            self.qglobals[bloq] = bloq_key
        else:
            bloq_key = self.qglobals[bloq]

        if bloq in self.qlocals.bloqvars:
            return

        # Should we make an 'alias' or just use the original name?
        # If it's longer than 20 characters, make an alias.
        should_alias = len(bloq_key) > 20
        if should_alias:
            prefix = _guess_good_alias_prefix(bloq, nodes=self.nodes)
            alias_str = self.qlocals.assign_bloq(bloq, prefix=prefix)
            self._stmnts.append(self.nodes.AliasAssignmentNode(alias=alias_str, bloq_key=bloq_key))
        else:
            self.qlocals.bloqvars[bloq] = bloq_key

    def add_bloqnection(
        self,
        binst: qlt.BloqInstance,
        preds: Sequence[qlt.Connection],
        succs: Sequence[qlt.Connection],
    ) -> None:

        # Case: This bloqnection represents an input to self.bloq from the outside world.
        if binst is qlt.LeftDangle:
            assert len(preds) == 0
            for suc in succs:
                reg = suc.left.reg
                if reg.shape:
                    v = self.nodes.QArgValueNode(reg.name, suc.left.idx)
                    self.qlocals.soqvars[suc.left] = v
                else:
                    v = self.nodes.QArgValueNode(reg.name, ())
                    self.qlocals.soqvars[suc.left] = v
            return

        kwargs: List[QArgNode] = []

        # Case: this bloqnection represents an output from self.bloq to the outside world.
        if binst is qlt.RightDangle:
            assert len(succs) == 0
            finsoqs = _cxns_to_soq_dict(
                self.bloq.signature.rights(),
                preds,
                get_me=lambda cxn: cxn.right,
                get_assign=lambda cxn: cxn.left,
            )
            for reg in self.bloq.signature.rights():
                regname = reg.name
                soqs = finsoqs[regname]
                if qlt.BloqBuilder.is_ndarray(soqs):
                    arr = np.empty(soqs.shape, dtype=object)
                    for idx in itertools.product(*[range(sh) for sh in soqs.shape]):
                        arr[idx] = self.qlocals.soqvars[soqs[idx]]
                    kwargs.append(self.nodes.QArgNode(regname, arr.tolist()))
                else:
                    kwargs.append(
                        self.nodes.QArgNode(regname, self.qlocals.soqvars[cast(_Soquet, soqs)])
                    )

            self._stmnts.append(self.nodes.QReturnNode(ret_mapping=kwargs))
            return

        # Otherwise, this is a qcall to a sub-bloq.
        # A. Handle input variables to the subbloq
        inpsoqs = _cxns_to_soq_dict(
            binst.bloq.signature.lefts(),
            preds,
            get_me=lambda cxn: cxn.right,
            get_assign=lambda cxn: cxn.left,
        )
        for reg in binst.bloq.signature.lefts():
            regname = reg.name
            soqs = inpsoqs[regname]
            if qlt.BloqBuilder.is_ndarray(soqs):
                arr = np.empty(soqs.shape, dtype=object)
                for idx in itertools.product(*[range(sh) for sh in soqs.shape]):
                    arr[idx] = self.qlocals.soqvars[soqs[idx]]
                kwargs.append(self.nodes.QArgNode(regname, arr.tolist()))
            else:
                kwargs.append(
                    self.nodes.QArgNode(regname, self.qlocals.soqvars[cast(_Soquet, soqs)])
                )

        # B: Handle output variables from the subbloq
        retsoqs = _cxns_to_soq_dict(
            binst.bloq.signature.rights(),
            succs,
            get_me=lambda cxn: cxn.left,
            get_assign=lambda cxn: cxn.left,
        )
        rets: List[str] = []
        for reg in binst.bloq.signature.rights():
            regname = reg.name
            soqs = retsoqs[regname]
            basename = self.qlocals.get_unique_name(regname)

            if qlt.BloqBuilder.is_ndarray(soqs):
                for idx in itertools.product(*[range(sh) for sh in soqs.shape]):
                    # Track indexed soquets as independent local variables
                    self.qlocals.soqvars[soqs[idx]] = self.nodes.QArgValueNode(basename, idx)
                # But syntactically, we assign to an indexless basename.
                rets.append(basename)
            else:
                self.qlocals.soqvars[cast(_Soquet, soqs)] = self.nodes.QArgValueNode(basename, ())
                rets.append(basename)

        # C. Record the call itself to `binst.bloq`.
        self._stmnts.append(
            self.nodes.QCallNode(
                bloq_key=self.qlocals.bloqvars[binst.bloq],
                lvalues=[self.nodes.LValueNode(r) for r in rets],
                qargs=kwargs,
            )
        )

    def finalize(self, extern_only_from: bool) -> QDefWithContext:
        if extern_only_from:
            cobject_from = None
        elif isinstance(self.bloq, qlt.CompositeBloq):
            cobject_from = None
        else:
            cobject_from = to_cobject_node(self.bloq, nodes=self.nodes)  # type: ignore[assignment]

        return QDefWithContext(
            qdef=self.nodes.QDefImplNode(
                bloq_key=self.bloq_key,
                qsignature=self._sig_entries,
                body=self._stmnts,
                cobject_from=cobject_from,
            ),
            bloq=self.bloq,
            implemented=True,
        )


def bloq_to_ast(
    bloq: qlt.Bloq,
    qglobals: Dict[qlt.Bloq, BloqKey],
    *,
    extern_only_from: bool,
    force_extern: bool = False,
    level: int = 0,
    nodes: L1Nodes = qualtran_l1_nodes,
) -> Tuple[QDefWithContext, List['qlt.Bloq']]:
    """Turn a bloq into Qualtran-L1 code.

    Generally just calls the right methods on `SubroutineFormatter`.

    Returns:
        fmt: A filled-in subroutine formatter for this bloq.
        subbloqs: The subbloqs found in the traversal.
    """

    if bloq in qglobals:
        bloq_key = qglobals[bloq]
    else:
        bloq_key = _get_unique_bloq_key(bloq, qglobals.values(), nodes=nodes)
        qglobals[bloq] = bloq_key

    qdb = QDefBuilder(
        bloq=bloq, bloq_key=bloq_key, qglobals=qglobals, nodes=nodes, qlocals=Locals(nodes=nodes)
    )
    indent = ' ' * level
    log.info("%sCompiling %s -> %s", indent, repr(bloq), bloq_key)

    # Signature
    qdb.add_signature(bloq.signature)

    if force_extern:
        return qdb.finalize_extern(reason='force_extern'), []

    # Detect bookkeeping / casting bloqs and emit as qcast.
    # Alloc and Free are _BookkeepingBloqs but are *not* casts.
    from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.bloqs.bookkeeping.qcast import QCast

    if isinstance(bloq, (Allocate, Free)):
        return qdb.finalize_extern(reason='alloc/free'), []
    if isinstance(bloq, (_BookkeepingBloq, QCast)):
        return qdb.finalize_qcast(), []
    if isinstance(bloq, qlt.CompositeBloq):
        cbloq = bloq
    else:
        try:
            cbloq = bloq.decompose_bloq()
        except qlt.DecomposeTypeError:
            return qdb.finalize_extern(reason='DecomposeTypeError'), []
        except qlt.DecomposeNotImplementedError:
            logging.warning("Missing decomposition for %s", bloq_key)
            return qdb.finalize_extern(reason='DecomposeNotImplementedError'), []

    g = cbloq._binst_graph

    # Make aliases for all the bloqs we find
    for binst in greedy_topological_sort(g):
        if isinstance(binst, qlt.DanglingT):
            continue
        qdb.add_alias_assignment_heuristically(binst.bloq)

    # Add calls and returns
    for binst in greedy_topological_sort(g):
        preds, succs = _binst_to_cxns(binst, binst_graph=g)
        qdb.add_bloqnection(binst, preds, succs)

    return qdb.finalize(extern_only_from=extern_only_from), list(qdb.qlocals.bloqvars.keys())


@attrs.mutable(kw_only=True)
class L1ModuleBuilder:
    """Format and export a Qualtran-L1 'module': a collection of 'qdef' bloq definitions."""

    qglobals: Dict[qlt.Bloq, BloqKey] = attrs.field(factory=dict)
    done: Set[qlt.Bloq] = attrs.field(factory=set)
    qdefs: List[QDefWithContext] = attrs.field(factory=list)
    extern_qdefs: List[QDefWithContext] = attrs.field(factory=list)
    nodes: L1Nodes = attrs.field(default=qualtran_l1_nodes)
    # all_costs: Dict['CostKey', Dict] = attrs.field(factory=dict)

    def add_bloqs(
        self,
        root: qlt.Bloq,
        *,
        annotate_costs: bool = False,
        extern_only_from: bool = True,
        force_extern_pred: Callable[['qlt.Bloq'], bool] = lambda b: False,
    ) -> BloqKey:

        # Stack of (Bloq, level) indices where `level` is the level of recursion
        subbloqs: List[Tuple[qlt.Bloq, int]] = [(root, 0)]

        # cks = [QECGatesCost(), QubitCount()]
        # if not self.all_costs:
        #     self.all_costs = {ck: {} for ck in cks}

        while subbloqs:
            bloq, level = subbloqs.pop(0)
            if bloq in self.done:
                continue

            force_extern = force_extern_pred(bloq)

            qdef, new_subbloqs = bloq_to_ast(
                bloq=bloq,
                qglobals=self.qglobals,
                extern_only_from=extern_only_from,
                force_extern=force_extern,
                level=level,
                nodes=self.nodes,
            )
            subbloqs += [(nsb, level + 1) for nsb in new_subbloqs]

            # if annotate_costs:
            #     lines = get_cost_lines(bloq, cks, self.all_costs)
            #     print(bloq)
            #     print('\n'.join(lines))
            #     print()

            self.qdefs.append(qdef)
            if not qdef.implemented:
                self.extern_qdefs.append(qdef)
            self.done.add(bloq)

        return self.qglobals[root]

    def finalize(self) -> L1Module:
        return self.nodes.L1Module(qdefs=tuple(qdef_with_ctx.qdef for qdef_with_ctx in self.qdefs))

    def pretty_print_qdef(self, bloq_or_bloq_key: Union[qlt.Bloq, BloqKey], f=None) -> None:
        from qualtran.l1 import L1ASTPrinter

        bloq_key: BloqKey
        if isinstance(bloq_or_bloq_key, qlt.Bloq):
            if bloq_or_bloq_key in self.qglobals:
                bloq_key = self.qglobals[bloq_or_bloq_key]
            else:
                raise KeyError(f"Unknown bloq key {bloq_or_bloq_key!r}")
        else:
            bloq_key = bloq_or_bloq_key

        qdefs = {qdef.bloq_key: qdef for qdef in self.qdefs}
        qdef = qdefs[bloq_key]
        qdef = qdef.qdef  # remove context
        txt = L1ASTPrinter().visit(qdef)
        if f is not None:
            f.write(txt)
        else:
            print(txt)

    def __str__(self):
        bloq_keys = ', '.join(x.bloq_key for x in self.qdefs)
        return f'L1ModuleBuilder({bloq_keys})'


def dump_l1(
    bloq: qlt.Bloq,
    f: Optional[io.IOBase] = None,
    *,
    annotate_costs: bool = False,
    extern_only_from: bool = False,
    force_extern_pred: Callable[['qlt.Bloq'], bool] = lambda b: False,
    nodes: L1Nodes = qualtran_l1_nodes,
) -> Optional[str]:
    from qualtran.l1 import L1ASTPrinter

    l1_mb = L1ModuleBuilder(nodes=nodes)
    root_bloq_key = l1_mb.add_bloqs(
        root=bloq,
        annotate_costs=annotate_costs,
        extern_only_from=extern_only_from,
        force_extern_pred=force_extern_pred,
    )
    l1_mod = l1_mb.finalize()
    l1_txt = L1ASTPrinter().visit(l1_mod)

    if f is None:
        return l1_txt

    f.write(l1_txt)
    return root_bloq_key


def dump_root_l1(bloq: qlt.Bloq, *, nodes: L1Nodes = qualtran_l1_nodes) -> str:
    from qualtran.l1 import L1ASTPrinter

    def extern_all_but_root(b: qlt.Bloq) -> bool:
        if b == bloq:
            return False
        return True

    l1_mb = L1ModuleBuilder(nodes=nodes)
    _root_bloq_key = l1_mb.add_bloqs(
        root=bloq, extern_only_from=True, force_extern_pred=extern_all_but_root
    )
    l1_mod = l1_mb.finalize()
    l1_txt = L1ASTPrinter().visit(l1_mod)

    return l1_txt


def _get_unique_bloq_key(
    bloq: qlt.Bloq, bloq_keys: Container[str], *, nodes: L1Nodes = qualtran_l1_nodes
) -> str:
    """Heuristic for writing a human-readable bloq key"""
    from qualtran.l1 import parse_objectstring

    key = str(bloq).replace('†', '_dag')
    try:
        structured_key = parse_objectstring(key, nodes=nodes)
        key = structured_key.canonical_str()
        if key not in bloq_keys:
            log.debug("Using structured __str__ for %s", bloq)
            return key
        i = 1
        while True:
            new_cargs = tuple(structured_key.cargs) + (
                nodes.CArgNode(key='variant', value=nodes.LiteralNode(value=i)),
            )
            key = attrs.evolve(structured_key, cargs=new_cargs).canonical_str()
            if key not in bloq_keys:
                log.warning(
                    "Structured __str__ for %s found, but had to be disambiguated, i=%d", bloq, i
                )
                return key
            i += 1
    except ValueError as e:
        log.error("Invalid bloq __str__ %s for %s: %s", bloq, repr(bloq), e)
        return f'bloq{uuid.uuid4().hex}'


def _guess_good_alias_prefix(bloq: qlt.Bloq, *, nodes: L1Nodes = qualtran_l1_nodes):
    """Heuristic for coming up with a good bloq object alias name"""

    from qualtran.l1 import parse_objectstring

    key = str(bloq).replace('†', '_dag')
    try:
        structured_key = parse_objectstring(key, nodes=nodes)
        name = structured_key.name
    except ValueError:
        name = bloq.__class__.__name__

    capitals = [char for char in name if char.isupper()]
    if len(capitals) >= 3:
        prefix = ''.join(capitals).lower()
    else:
        prefix = name.lower()

    return prefix
