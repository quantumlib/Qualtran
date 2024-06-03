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

"""Classes for drawing bloqs with Graphviz."""
import html
import itertools
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import IPython.display
import pydot

from qualtran import (
    Bloq,
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    QBit,
    QDType,
    Register,
    RightDangle,
    Side,
    Signature,
    Soquet,
)


def _assign_ids_to_bloqs_and_soqs(
    bloq_instances: Iterable[BloqInstance], all_soquets: Iterable[Soquet]
) -> Dict[Any, str]:
    """Assign unique identifiers to bloq instances, soquets, and register groups.

    Graphviz is very forgiving in its input format. If you accidentally introduce a new id (e.g.
    when defining an edge) that doesn't correspond to an existing node it will silently accept
    this and draw a wonky graph. This function forces the declaration of all
    possible graphviz objects ahead of time to remove this class of errors.

    Returns:
        A dictionary mapping objects to string identifiers. The objects are as follows:
        1) Each BloqInstance in `bloq_instances`. 2) For each bloq instance, a collection of
        (bloq_instance, group_name) tuples for each register group name. Registers with
        shared names (but differing `side` attributes) are implicitly grouped. 3) Each
        Soquet in `all_soquets`.
    """
    to_id: Dict[Any, str] = {}
    ids: Set[str] = set()
    disambiguator = 0

    def add(item: Any, desired_id: str):
        nonlocal disambiguator
        if item in to_id:
            raise ValueError(f"Item {item} was already added to the ID mapping.")

        if desired_id not in ids:
            unique_id = desired_id
        else:
            unique_id = f'{desired_id}_G{disambiguator}'
            disambiguator += 1

        ids.add(unique_id)
        to_id[item] = unique_id

    for binst in bloq_instances:
        add(binst, f'{binst.bloq.__class__.__name__}')

        for groupname, _ in binst.bloq.signature.groups():
            add((binst, groupname), groupname)

    for soq in all_soquets:
        add(soq, f'{soq.reg.name}')

    return to_id


def _parition_registers_in_a_group(
    regs: Iterable[Register], binst: BloqInstance
) -> Tuple[List[Soquet], List[Soquet], List[Soquet]]:
    """Construct and sort the expected Soquets for a given register group.

    Since we expect the input registers to be in a group, we assert that
    if they are THRU register there are not LEFT and RIGHT registers as well.

    This is a helper method used in `GraphDrawer.add_binst()`.
    """
    lefts = []
    rights = []
    thrus = []
    for reg in regs:
        for idx in reg.all_idxs():
            soq = Soquet(binst, reg, idx)
            if reg.side is Side.LEFT:
                lefts.append(soq)
            elif reg.side is Side.RIGHT:
                rights.append(soq)
            else:
                assert reg.side is Side.THRU
                thrus.append(soq)

    if len(thrus) > 0:
        if len(lefts) > 0 or len(rights) > 0:
            raise ValueError(
                "A register group containing THRU registers cannot "
                "also contain LEFT and RIGHT registers."
            )

    return lefts, rights, thrus


class GraphDrawer:
    """A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

    Graphviz has nodes, edges, and ports. Nodes are HTML tables representing bloq instances.
    Each cell in the table has a graphviz port and represents a soquet. Edges connect
    node:port tuples representing connections between soquets.

    Each node and port has a string identifier. We use the `_IDBuilder` helper class
    to assign unique, readable IDs to each object.

    Users should call `GraphDrawer.get_graph()` as the primary entry point. Other methods
    can be overridden to customize the look of the resulting graph.

    To display a graph in a jupyter notebook consider using the SVG utilities:

    >>> dr = GraphDrawer(cbloq)
    >>> dr.get_svg()

    Args:
        bloq: The bloq or composite bloq to draw.
    """

    def __init__(self, bloq: Bloq):
        cbloq = bloq.as_composite_bloq()
        self._cbloq = cbloq
        self._binsts = cbloq.bloq_instances
        self._soquets = cbloq.all_soquets

        self.ids = _assign_ids_to_bloqs_and_soqs(self._binsts, self._soquets)

    def get_dangle_node(self, soq: Soquet) -> pydot.Node:
        """Overridable method to create a Node representing dangling Soquets."""
        return pydot.Node(self.ids[soq], label=soq.pretty(), shape='plaintext')

    def add_dangles(
        self, graph: pydot.Graph, signature: Signature, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        if dangle is LeftDangle:
            regs = signature.lefts()
        elif dangle is RightDangle:
            regs = signature.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for reg in regs:
            for idx in reg.all_idxs():
                subg.add_node(self.get_dangle_node(Soquet(dangle, reg, idx=idx)))
        graph.add_subgraph(subg)
        return graph

    def soq_label(self, soq: Soquet) -> str:
        """Overridable method for getting label text for a Soquet."""
        return soq.pretty()

    def get_thru_register(self, thru: Soquet) -> str:
        """Overridable method for generating a <TR> representing a THRU soquet.

        This should have a `colspan="2"` to make sure there aren't separate left and right
        cells / soquets.
        """
        return (
            f'  <TR><TD colspan="2" port="{self.ids[thru]}">'
            f'{html.escape(self.soq_label(thru))}</TD></TR>\n'
        )

    def _register_td(self, soq: Optional[Soquet], *, with_empty_td: bool, rowspan: int = 1) -> str:
        """Return the html code for an individual <TD>.

        This includes some factored-out complexity which aims to correctly pad cells that
        are empty due to a differing number of left and right soquets.

        Args:
            soq: The optional soquet.
            with_empty_td: If `soq` is `None`, return an empty `<TD>` in its place if
                this is set to True. Otherwise, omit the empty TD and rely on the rowspan arguments.
            rowspan: If greater than `1`, include the `rowspan` html attribute on the TD to
                span multiple rows.
        """
        if soq is None:
            if with_empty_td:
                return '<TD></TD>'
            else:
                return ''

        if rowspan != 1:
            assert rowspan > 1
            rowspan_html = f'rowspan="{rowspan}"'
        else:
            rowspan_html = ''

        return f'<TD {rowspan_html} port="{self.ids[soq]}">{html.escape(self.soq_label(soq))}</TD>'

    def _get_register_tr(
        self,
        left: Optional[Soquet],
        right: Optional[Soquet],
        *,
        with_empty_td: bool = True,
        left_rowspan: int = 1,
        right_rowspan: int = 1,
    ) -> str:
        """Return the html code for a <TR> where `left` and `right` may be `None`.

        Args:
            left: The optional left soquet.
            right: the optional right soquet.
            with_empty_td: If `left` or `right` is `None`, put an empty `<TD>` in its place if
                this is set to True. Otherwise, omit the empty TD and rely on the rowspan arguments.
            left_rowspan: If greater than `1`, include the `rowspan` html attribute on left TDs to
                span multiple rows.
            right_rowspan: If greater than `1`, include the `rowspan` html attribute on right TDs to
                span multiple rows.
        """
        tr_code = '  <TR>'
        tr_code += self._register_td(left, rowspan=left_rowspan, with_empty_td=with_empty_td)
        tr_code += self._register_td(right, rowspan=right_rowspan, with_empty_td=with_empty_td)
        tr_code += '</TR>\n'
        return tr_code

    def get_binst_table_attributes(self) -> str:
        """Overridable method to configure the desired table attributes for the bloq."""
        return ''

    def get_binst_header_text(self, binst: BloqInstance) -> str:
        """Overridable method returning the text used for the header cell of a bloq."""
        return f'{html.escape(str(binst.bloq))}'

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        """Process and add a bloq instance to the Graph."""

        label = '<'  # graphviz: start an HTML section
        label += f'<TABLE {self.get_binst_table_attributes()}>\n'

        label += f'  <TR><TD colspan="2">{self.get_binst_header_text(binst)}</TD></TR>\n'

        for groupname, groupregs in binst.bloq.signature.groups():
            lefts, rights, thrus = _parition_registers_in_a_group(groupregs, binst)

            # Special case: all registers are THRU and we don't need different left and right
            # columns.
            if len(thrus) > 0:
                for t in thrus:
                    label += self.get_thru_register(t)
                continue

            # To do nice drawing into an html-style table, we need to manually bookkeep
            # which columns have extra, empty rows.
            n_surplus_lefts = max(0, len(lefts) - len(rights))
            n_surplus_rights = max(0, len(rights) - len(lefts))
            n_common = min(len(lefts), len(rights))

            if n_common >= 1:
                # We have some rows ("common" rows) that have both left and right soquets.
                # This branch will correctly deal with rowspan arguments for one column having
                # greater or fewer items.

                # add all but the last common rows. Both lefts[i] and rights[i] are non-None.
                for i in range(n_common - 1):
                    label += self._get_register_tr(lefts[i], rights[i])

                # For the last common row, we need to include an increased rowspan for
                # to pad the less-full column.
                label += self._get_register_tr(
                    lefts[n_common - 1],
                    rights[n_common - 1],
                    left_rowspan=n_surplus_rights + 1,
                    right_rowspan=n_surplus_lefts + 1,
                )

                # For the rest of the registers, we don't include empty TDs
                # because we used the rowspan argument above.
                with_empty_td = False
            else:
                # No common rows; no place to include rowspan arguments so we pad with empty TDs
                with_empty_td = True

            # Add the rest of the registers.
            for l, r in itertools.zip_longest(lefts[n_common:], rights[n_common:], fillvalue=None):
                label += self._get_register_tr(l, r, with_empty_td=with_empty_td)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        graph.add_node(pydot.Node(self.ids[binst], label=label, shape='plain'))
        return graph

    def cxn_label(self, cxn: Connection) -> str:
        """Overridable method to return labels for connections."""
        return str(cxn.shape)

    def cxn_edge(self, left_id: str, right_id: str, cxn: Connection) -> pydot.Edge:
        """Overridable method to style a pydot.Edge for connecionts."""
        return pydot.Edge(left_id, right_id, label=self.cxn_label(cxn))

    def add_cxn(self, graph: pydot.Graph, cxn: Connection) -> pydot.Graph:
        """Process and add a connection to the Graph.

        Connections are specified using a `:` delimited set of ids. The first element
        is the node (bloq instance). For most bloq instances, the second element is
        the port (soquet). The final element is the compass direction of where exactly
        the connecting line should be anchored.

        For DangleT nodes, there aren't any Soquets so the second element is omitted.
        """

        if cxn.left.binst is LeftDangle:
            left = f'{self.ids[cxn.left]}:e'
        else:
            left = f'{self.ids[cxn.left.binst]}:{self.ids[cxn.left]}:e'

        if cxn.right.binst is RightDangle:
            right = f'{self.ids[cxn.right]}:w'
        else:
            right = f'{self.ids[cxn.right.binst]}:{self.ids[cxn.right]}:w'

        graph.add_edge(self.cxn_edge(left, right, cxn))
        return graph

    def get_graph(self) -> pydot.Dot:
        """Get the graphviz graph representing the Bloq.

        This is the main entry-point to this class.
        """
        graph = pydot.Dot('my_graph', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, self._cbloq.signature, LeftDangle)

        for binst in self._binsts:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, self._cbloq.signature, RightDangle)

        for cxn in self._cbloq.connections:
            graph = self.add_cxn(graph, cxn)

        return graph

    def get_svg_bytes(self) -> bytes:
        """Get the SVG code (as bytes) for drawing the graph."""
        return self.get_graph().create(prog='dot', format='svg', encoding='utf-8')

    def get_svg(self) -> IPython.display.SVG:
        """Get an IPython SVG object displaying the graph."""
        return IPython.display.SVG(self.get_svg_bytes())


class PrettyGraphDrawer(GraphDrawer):
    def get_binst_table_attributes(self) -> str:
        return 'BORDER="0" CELLBORDER="1" CELLSPACING="0"'

    def get_binst_header_text(self, binst: BloqInstance):
        from qualtran.bloqs.bookkeeping import Join, Split

        if isinstance(binst.bloq, (Split, Join)):
            return ''
        return f'<font point-size="10">{html.escape(str(binst.bloq))}</font>'

    def soq_label(self, soq: Soquet):
        from qualtran.bloqs.bookkeeping import Join, Split

        if isinstance(soq.binst, BloqInstance) and isinstance(soq.binst.bloq, (Split, Join)):
            return ''
        return soq.pretty()

    def get_default_text(self, reg: Register) -> str:
        if reg.name == 'control':
            return '\u2b24'

        return reg.name

    def cxn_edge(self, left_id: str, right_id: str, cxn: Connection) -> pydot.Edge:
        return pydot.Edge(
            left_id,
            right_id,
            label=self.cxn_label(cxn),
            labelfloat=True,
            fontsize=10,
            arrowhead='dot',
            arrowsize=0.25,
        )


class TypedGraphDrawer(PrettyGraphDrawer):
    @staticmethod
    def _fmt_dtype(dtype: QDType):
        return str(dtype)

    def cxn_label(self, cxn: Connection) -> str:
        """Overridable method to return labels for connections."""

        l, r = cxn.left.reg.dtype, cxn.right.reg.dtype
        if l == r:
            return self._fmt_dtype(l)
        elif l.num_qubits == 1:
            return self._fmt_dtype(l if isinstance(l, QBit) else r)
        else:
            return f'{self._fmt_dtype(l)}-{self._fmt_dtype(r)}'

    def cxn_edge(self, left_id: str, right_id: str, cxn: Connection) -> pydot.Edge:
        return pydot.Edge(
            left_id,
            right_id,
            label=self.cxn_label(cxn),
            labelfloat=False,
            fontcolor='red' if '-' in self.cxn_label(cxn) else 'black',
            fontsize=10,
            arrowhead='dot',
            arrowsize=0.25,
        )
