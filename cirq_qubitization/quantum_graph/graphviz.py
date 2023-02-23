import itertools
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pydot

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)
from cirq_qubitization.quantum_graph.util_bloqs import Join, Partition, Split, Unpartition


class _IDBuilder:
    """A helper builder class for assigning unique, readable string identifiers to objects.

    Any hashable Python object can be added to the ID mapping. Each addition should provide
    a string `desired_id` to which we will add a disambiguating integer if required.
    """

    def __init__(self):
        self._to_id: Dict[Any, str] = {}
        self._ids: Set[str] = set()
        self._disambiguator = 0
        self._valid = True

    def add(self, item: Any, desired_id: str):
        if item in self._to_id:
            raise ValueError(f"Item {item} was already added to the ID mapping.")

        if not self._valid:
            raise ValueError("This builder is no longer valid.")

        if desired_id not in self._ids:
            unique_id = desired_id
        else:
            unique_id = f'{desired_id}_G{self._disambiguator}'
            self._disambiguator += 1

        self._ids.add(unique_id)
        self._to_id[item] = unique_id

    def build(self) -> Dict[Any, str]:
        self._valid = False
        return self._to_id

    @classmethod
    def build_bloqs_and_soqs(
        cls, bloq_instances: Set[BloqInstance], all_soquets: Set[Soquet]
    ) -> Dict[Any, str]:
        """Assign unique identifiers to bloq instances, soquets, and register groups."""
        ibuilder = cls()
        for binst in bloq_instances:
            ibuilder.add(binst, f'{binst.bloq.__class__.__name__}')

            for groupname, groupregs in binst.bloq.registers.groups():
                ibuilder.add((binst, groupname), groupname)

        for soq in all_soquets:
            ibuilder.add(soq, f'{soq.reg.name}')

        return ibuilder.build()


def _parition_registers_in_a_group(
    regs: Iterable[FancyRegister], binst: BloqInstance
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
        for idx in reg.wire_idxs():
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

    >>> from IPython.display import SVG
    >>> dr = GraphDrawer(cbloq)
    >>> SVG(dr.get_graph().create_svg())

    Args:
        bloq: The bloq or composite bloq to draw.
    """

    def __init__(self, bloq: Bloq):
        cbloq = bloq.as_composite_bloq()
        self._cbloq = cbloq
        self._binsts = cbloq.bloq_instances
        self._soquets = cbloq.all_soquets

        self.ids = _IDBuilder.build_bloqs_and_soqs(self._binsts, self._soquets)

    def get_dangle_node(self, soq: Soquet) -> pydot.Node:
        """Overridable method to create a Node representing dangling Soquets."""
        return pydot.Node(self.ids[soq], label=soq.pretty(), shape='plaintext')

    def add_dangles(
        self, graph: pydot.Graph, soquets: FancyRegisters, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        if dangle is LeftDangle:
            regs = soquets.lefts()
        elif dangle is RightDangle:
            regs = soquets.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for reg in regs:
            for idx in reg.wire_idxs():
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
        return f'<TR><TD colspan="2" port="{self.ids[thru]}">{self.soq_label(thru)}</TD></TR>'

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
            rowspan = f'rowspan="{rowspan}"'
        else:
            rowspan = ''

        return f'<TD {rowspan} port="{self.ids[soq]}">{self.soq_label(soq)}</TD>'

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
        tr_code = '<TR>'
        tr_code += self._register_td(left, rowspan=left_rowspan, with_empty_td=with_empty_td)
        tr_code += self._register_td(right, rowspan=right_rowspan, with_empty_td=with_empty_td)
        tr_code += '</TR>'
        return tr_code

    def get_binst_table_attributes(self) -> str:
        """Overridable method to configure the desired table attributes for the bloq."""
        return 'BORDER="1" CELLBORDER="1" CELLSPACING="3"'

    def get_binst_header_text(self, binst: BloqInstance) -> str:
        """Overridable method returning the text used for the header cell of a bloq."""
        return f'{binst.bloq.pretty_name()}'

    def add_binst(self, graph: pydot.Graph, binst: BloqInstance) -> pydot.Graph:
        """Process and add a bloq instance to the Graph."""

        label = '<'  # graphviz: start an HTML section
        label += f'<TABLE {self.get_binst_table_attributes()}>'

        label += f'<tr><td colspan="2">{self.get_binst_header_text(binst)}</td></tr>'

        for groupname, groupregs in binst.bloq.registers.groups():
            lefts, rights, thrus = _parition_registers_in_a_group(groupregs, binst)

            # Case 1: all registers are THRU and we don't need different left and right
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

            # Case 2: we have some rows ("common" rows) that have both left and right soquets.
            # This case will correctly deal with rowspan arguments for one column having
            # greater or fewer items.
            if n_common >= 1:
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

                # Add the rest of the registers. We don't include empty TDs
                # because we used the rowspan argument above.
                for l, r in itertools.zip_longest(
                    lefts[n_common:], rights[n_common:], fillvalue=None
                ):
                    label += self._get_register_tr(l, r, with_empty_td=False)
                continue

            # Case 3: Only one column has values. The other will have an empty TD.
            for l, r in itertools.zip_longest(lefts, rights, fillvalue=None):
                label += self._get_register_tr(l, r, with_empty_td=True)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section

        graph.add_node(pydot.Node(self.ids[binst], label=label, shape='plain'))
        return graph

    def cxn_label(self, cxn: Connection) -> str:
        """Overridable method to return labels for connections."""
        return str(cxn.shape)

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

        graph.add_edge(
            pydot.Edge(
                left,
                right,
                label=str(cxn.shape),
                labelfloat=True,
                fontsize=10,
                arrowhead='dot',
                arrowsize=0.25,
            )
        )

        return graph

    def get_graph(self) -> pydot.Graph:
        """Get the graphviz graph representing the Bloq.

        This is the main entry-point to this class.
        """
        graph = pydot.Dot('my_graph', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, self._cbloq.registers, LeftDangle)

        for binst in self._binsts:
            graph = self.add_binst(graph, binst)

        graph = self.add_dangles(graph, self._cbloq.registers, RightDangle)

        for cxn in self._cbloq.connections:
            graph = self.add_cxn(graph, cxn)

        return graph


class PrettyGraphDrawer(GraphDrawer):
    INFRA_BLOQ_TYPES = (Split, Join, Partition, Unpartition)

    def get_binst_table_attributes(self) -> str:
        return 'BORDER="0" CELLBORDER="1" CELLSPACING="0"'

    def get_binst_header_text(self, binst: BloqInstance):
        if isinstance(binst.bloq, self.INFRA_BLOQ_TYPES):
            return ''
        return f'<font point-size="10">{binst.bloq.short_name()}</font>'

    def soq_label(self, soq: Soquet):
        if isinstance(soq.binst, BloqInstance) and isinstance(
            soq.binst.bloq, self.INFRA_BLOQ_TYPES
        ):
            return ''
        return soq.pretty()

    def get_default_text(self, reg: FancyRegister) -> str:
        if reg.name == 'control':
            return '\u2b24'

        return reg.name
