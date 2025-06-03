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

"""Convenience functions for showing rich displays in Jupyter notebook."""

import os
import re
import sympy
from typing import Dict, Optional, Text, overload, Sequence, TYPE_CHECKING, Union, Tuple

import IPython.display
import ipywidgets

from qualtran import Bloq

from .bloq_counts_graph import format_counts_sigma, GraphvizCallGraph
from .flame_graph import get_flame_graph_svg_data
from .graphviz import PrettyGraphDrawer, TypedGraphDrawer
from .musical_score import MusicalScoreData, TextBox, draw_musical_score, get_musical_score_data
from .qpic_diagram import qpic_diagram_for_bloq

if TYPE_CHECKING:
    import networkx as nx
    

def show_bloq(bloq: 'Bloq', type: str = 'graph'):  # pylint: disable=redefined-builtin
    """Display a visual representation of the bloq in IPython.

    Args:
        bloq: The bloq to show
        type: Either 'graph', 'dtype', 'musical_score' or 'latex'. By default, display
            a directed acyclic graph of the bloq connectivity. If dtype then the
            connections are labelled with their dtypes rather than bitsizes. If 'latex',
            then latex diagrams are drawn using `qpic`, which should be installed already
            and is invoked via a subprocess.run() call. Otherwise, draw a musical score diagram.
    """
    if type.lower() == 'graph':
        IPython.display.display(PrettyGraphDrawer(bloq).get_svg())
    elif type.lower() == 'dtype':
        IPython.display.display(TypedGraphDrawer(bloq).get_svg())
    elif type.lower() == 'musical_score':
        msd = get_musical_score_data(bloq)
        pretty_msd = pretty_format_msd(msd)
        draw_musical_score(pretty_msd)
    elif type.lower() == 'latex':
        show_bloq_via_qpic(bloq)
    else:
        raise ValueError(
            f"Unknown `show_bloq` type: {type}."
            "Allowed types are [graph, dtype, musical_score, latex]"
        )


def show_bloqs(bloqs: Sequence['Bloq'], labels: Optional[Sequence[Optional[str]]] = None):
    """Display multiple bloqs side-by-side in IPython."""
    n = len(bloqs)
    if labels is not None:
        assert len(labels) == n, 'Must provide exactly as many labels as bloqs'
    else:
        labels = [None] * n

    outs = [ipywidgets.Output() for _ in range(n)]
    box = ipywidgets.HBox(outs)

    for i, (bloq, label) in enumerate(zip(bloqs, labels)):
        if label:
            outs[i].append_display_data(IPython.display.Markdown(label))
        outs[i].append_display_data(PrettyGraphDrawer(bloq).get_svg())

    IPython.display.display(box)


@overload
def show_call_graph(
    item: 'Bloq', /, *, max_depth: Optional[int] = None, agg_gate_counts: Optional[str] = None
) -> None: ...


@overload
def show_call_graph(
    item: 'nx.Graph', /, *, max_depth: Optional[int] = None, agg_gate_counts: Optional[str] = None
) -> None: ...


def show_call_graph(
    item: Union['Bloq', 'nx.Graph'],
    /,
    *,
    max_depth: Optional[int] = None,
    agg_gate_counts: Optional[str] = None,
) -> None:
    """Display a graph representation of the call graph.

    Args:
        item: Either a networkx graph or a bloq. If a networkx graph, it should be a "call graph"
            which is passed verbatim to the graph drawer and the additional arguments to this
            function are ignored. If it is a bloq, the factory
            method `GraphvizCallGraph.from_bloq()` is used to construct the call graph, compute
            relevant costs, and display the call graph annotated with the costs.
        max_depth: The maximum depth (from the root bloq) of the call graph to draw. Note
            that the cost computations will walk the whole call graph, but only the nodes
            within this depth will be drawn.
        agg_gate_counts: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
            (optionally) aggregate the gate counts. If not specified, the 'factored'
            approach is used where each type of gate is counted individually.

    """
    if isinstance(item, Bloq):
        IPython.display.display(
            GraphvizCallGraph.from_bloq(
                item, max_depth=max_depth, agg_gate_counts=agg_gate_counts
            ).get_svg()
        )
    else:
        IPython.display.display(GraphvizCallGraph(item).get_svg())


def show_counts_sigma(sigma: Dict['Bloq', Union[int, 'sympy.Expr']]):
    """Display nicely formatted bloq counts sums `sigma`."""
    IPython.display.display(IPython.display.Markdown(format_counts_sigma(sigma)))


def show_flame_graph(*bloqs: 'Bloq', **kwargs):
    """Display hiearchical decomposition and T-complexity costs as a Flame Graph."""
    svg_data = get_flame_graph_svg_data(*bloqs, **kwargs)
    IPython.display.display(IPython.display.SVG(svg_data))


def show_bloq_via_qpic(bloq: 'Bloq', width: int = 1000, height: int = 400):
    """Display latex diagram for bloq by invoking `qpic`. Assumes qpic is already installed."""
    output_file_path = qpic_diagram_for_bloq(bloq, output_type='png')

    from IPython.display import Image

    IPython.display.display(Image(output_file_path, width=width, height=height))
    os.remove(output_file_path)


def pretty_format_msd(msd: MusicalScoreData) -> MusicalScoreData:
    """
    Beautifies MSD to enable pretty diagrams

    Args:
        msd: A raw MSD

    Returns:
        new_msd: A pretty MSD

    """

    def symbols_to_identity(lbl: str) -> Tuple[str, str]:
        """
        Exchanges any symbols in the label for integer 1 or returns lbl if no symbols found.

        Args:
            lbl: The label to be processed.

        Returns:
            new_lbl: A label without symbols

        """
        
        pattern = r"Abs\((?P<symbol>[a-zA-Z])\)"
        match = re.search(pattern, lbl)
        if match:
            symbol = match.group("symbol")
            new_lbl = lbl.replace(symbol, "1")
            return new_lbl, symbol
        return lbl, ""

    simpify_locals = {
        "Min": sympy.Min,
        "ceiling": sympy.ceiling,
        "log2": lambda x: sympy.log(x, 2)
    }

    mult = 1
    pretty_soqs = []
    for soq_item in msd.soqs:
        if isinstance(soq_item.symb, (TextBox, Text)):
            try:
                lbl_raw = soq_item.symb.text
                lbl_no_symbols, symbol = symbols_to_identity(lbl_raw)
                gate, base, exponent = sum([p.split("**", 1) for p in lbl_no_symbols.split("^")], [])

                if len(base.split("*", 1)) > 1:
                    mult_str, base = base.rsplit("*", 1)
                    mult = sympy.sympify(mult_str, evaluate=True)

                exponent = sympy.sympify(exponent, locals=simpify_locals, evaluate=True)
                expression = str(base) + "**" + str(exponent)
                expression = sympy.sympify(expression, locals=simpify_locals, evaluate=True)
                new_lbl = str(gate) + "^" + symbol + "*" + str(expression * mult)
                
                new_soq = soq_item.__class__(
                    symb= TextBox(text=new_lbl) if isinstance(soq_item.symb, TextBox) else Text(text=new_lbl, fontsize=soq_item.symb.fontsize),
                    rpos= soq_item.rpos,
                    ident= soq_item.ident
                )
                pretty_soqs.append(new_soq)
            except (ValueError, TypeError, NameError, sympy.SympifyError):
                pretty_soqs.append(soq_item)
        else:
            pretty_soqs.append(soq_item)

    pretty_msd = MusicalScoreData(
        max_x=msd.max_x,
        max_y=msd.max_y,
        soqs=pretty_soqs,
        hlines=msd.hlines,
        vlines=msd.vlines
    )

    return pretty_msd
