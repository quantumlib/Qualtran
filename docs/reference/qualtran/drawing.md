# Module: drawing


Draw and visualize bloqs




isort:skip_file
## Modules

[`bloq_counts_graph`](../qualtran/drawing/bloq_counts_graph.md): Classes for drawing bloq counts graphs with Graphviz.

[`classical_sim_graph`](../qualtran/drawing/classical_sim_graph.md): Classes for drawing classical data flows with Graphviz.

[`graphviz`](../qualtran/drawing/graphviz.md): Classes for drawing bloqs with Graphviz.

[`musical_score`](../qualtran/drawing/musical_score.md): Tools for laying out composite bloq graphs onto a "musical score".

## Classes

[`class GraphDrawer`](../qualtran/drawing/GraphDrawer.md): A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

[`class PrettyGraphDrawer`](../qualtran/drawing/PrettyGraphDrawer.md): A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

[`class RegPosition`](../qualtran/drawing/RegPosition.md): Coordinates for a register when visualizing on a musical score.

[`class HLine`](../qualtran/drawing/HLine.md): Dataclass representing a horizontal line segment at a given vertical position `x`.

[`class VLine`](../qualtran/drawing/VLine.md): Data for drawing vertical lines.

[`class LineManager`](../qualtran/drawing/LineManager.md): Methods to manage allocation and de-allocation of lines representing a register of qubits.

[`class WireSymbol`](../qualtran/drawing/WireSymbol.md): Base class for a symbol.

[`class TextBox`](../qualtran/drawing/TextBox.md): Base class for a symbol.

[`class Text`](../qualtran/drawing/Text.md): Base class for a symbol.

[`class RarrowTextBox`](../qualtran/drawing/RarrowTextBox.md): Base class for a symbol.

[`class LarrowTextBox`](../qualtran/drawing/LarrowTextBox.md): Base class for a symbol.

[`class Circle`](../qualtran/drawing/Circle.md): Base class for a symbol.

[`class ModPlus`](../qualtran/drawing/ModPlus.md): Base class for a symbol.

[`class MusicalScoreData`](../qualtran/drawing/MusicalScoreData.md): All the data required to draw a musical score.

[`class ClassicalSimGraphDrawer`](../qualtran/drawing/ClassicalSimGraphDrawer.md): A graph drawer that labels each edge with a classical value.

[`class GraphvizCounts`](../qualtran/drawing/GraphvizCounts.md): This class turns a bloqs count graph into Graphviz objects and drawings.

## Functions

[`directional_text_box(...)`](../qualtran/drawing/directional_text_box.md)

[`draw_musical_score(...)`](../qualtran/drawing/draw_musical_score.md)

[`dump_musical_score(...)`](../qualtran/drawing/dump_musical_score.md)

[`format_counts_graph_markdown(...)`](../qualtran/drawing/format_counts_graph_markdown.md): Format a text version of `graph` as markdown.

[`format_counts_sigma(...)`](../qualtran/drawing/format_counts_sigma.md): Format `sigma` as markdown.

[`get_musical_score_data(...)`](../qualtran/drawing/get_musical_score_data.md): Get the musical score data for a (composite) bloq.

[`show_bloq(...)`](../qualtran/drawing/show_bloq.md): Display a visual representation of the bloq in IPython.

[`show_bloqs(...)`](../qualtran/drawing/show_bloqs.md): Display multiple bloqs side-by-side in IPython.

[`show_call_graph(...)`](../qualtran/drawing/show_call_graph.md): Display a graph representation of the counts graph `g`.

[`show_counts_sigma(...)`](../qualtran/drawing/show_counts_sigma.md): Display nicely formatted bloq counts sums `sigma`.

