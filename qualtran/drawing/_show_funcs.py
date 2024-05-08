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
from typing import Dict, Optional, Sequence, TYPE_CHECKING, Union

import IPython.display
import ipywidgets

from .bloq_counts_graph import format_counts_sigma, GraphvizCounts
from .flame_graph import get_flame_graph_svg_data
from .graphviz import PrettyGraphDrawer, TypedGraphDrawer
from .musical_score import draw_musical_score, get_musical_score_data
from .qpic_diagram import qpic_diagram_for_bloq

if TYPE_CHECKING:
    import networkx as nx
    import sympy

    from qualtran import Bloq


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
        draw_musical_score(get_musical_score_data(bloq))
    elif type.lower() == 'latex':
        show_bloq_via_qpic(bloq)
    else:
        raise ValueError(f"Unknown `show_bloq` type: {type}.")


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


def show_call_graph(g: 'nx.DiGraph') -> None:
    """Display a graph representation of the counts graph `g`."""
    IPython.display.display(GraphvizCounts(g).get_svg())


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
