# TypedGraphDrawer
`qualtran.drawing.graphviz.TypedGraphDrawer`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L405-L430">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

Inherits From: [`PrettyGraphDrawer`](../../../qualtran/drawing/PrettyGraphDrawer.md), [`GraphDrawer`](../../../qualtran/drawing/GraphDrawer.md)

<!-- Placeholder for "Used in" -->

Graphviz has nodes, edges, and ports. Nodes are HTML tables representing bloq instances.
Each cell in the table has a graphviz port and represents a soquet. Edges connect
node:port tuples representing connections between soquets.

Each node and port has a string identifier. We use the `_IDBuilder` helper class
to assign unique, readable IDs to each object.

Users should call <a href="../../../qualtran/drawing/GraphDrawer.html#get_graph"><code>GraphDrawer.get_graph()</code></a> as the primary entry point. Other methods
can be overridden to customize the look of the resulting graph.

To display a graph in a jupyter notebook consider using the SVG utilities:

```
>>> dr = GraphDrawer(cbloq)
>>> dr.get_svg()
```

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq or composite bloq to draw.




## Methods

<h3 id="cxn_label"><code>cxn_label</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L411-L418">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_label(
    cxn: <a href="../../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> str
</code></pre>

Overridable method to return labels for connections.


<h3 id="cxn_edge"><code>cxn_edge</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L420-L430">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_edge(
    left_id: str,
    right_id: str,
    cxn: <a href="../../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> pydot.Edge
</code></pre>

Overridable method to style a pydot.Edge for connecionts.




