# GraphDrawer
`qualtran.drawing.GraphDrawer`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L118-L365">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.graphviz.GraphDrawer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.GraphDrawer(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

Graphviz has nodes, edges, and ports. Nodes are HTML tables representing bloq instances.
Each cell in the table has a graphviz port and represents a soquet. Edges connect
node:port tuples representing connections between soquets.

Each node and port has a string identifier. We use the `_IDBuilder` helper class
to assign unique, readable IDs to each object.

Users should call <a href="../../qualtran/drawing/GraphDrawer.html#get_graph"><code>GraphDrawer.get_graph()</code></a> as the primary entry point. Other methods
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

<h3 id="get_dangle_node"><code>get_dangle_node</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L148-L150">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dangle_node(
    soq: <a href="../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> pydot.Node
</code></pre>

Overridable method to create a Node representing dangling Soquets.


<h3 id="add_dangles"><code>add_dangles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L152-L171">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_dangles(
    graph: pydot.Graph,
    signature: <a href="../../qualtran/Signature.html"><code>qualtran.Signature</code></a>,
    dangle: <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>
) -> pydot.Graph
</code></pre>

Add nodes representing dangling indices to the graph.

We wrap this in a subgraph to align (rank=same) the 'nodes'

<h3 id="soq_label"><code>soq_label</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L173-L175">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>soq_label(
    soq: <a href="../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> str
</code></pre>

Overridable method for getting label text for a Soquet.


<h3 id="get_thru_register"><code>get_thru_register</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L177-L186">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_thru_register(
    thru: <a href="../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> str
</code></pre>

Overridable method for generating a <TR> representing a THRU soquet.

This should have a `colspan="2"` to make sure there aren't separate left and right
cells / soquets.

<h3 id="get_binst_table_attributes"><code>get_binst_table_attributes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L242-L244">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_binst_table_attributes() -> str
</code></pre>

Overridable method to configure the desired table attributes for the bloq.


<h3 id="get_binst_header_text"><code>get_binst_header_text</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L246-L248">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_binst_header_text(
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>
) -> str
</code></pre>

Overridable method returning the text used for the header cell of a bloq.


<h3 id="add_binst"><code>add_binst</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L250-L307">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_binst(
    graph: pydot.Graph,
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>
) -> pydot.Graph
</code></pre>

Process and add a bloq instance to the Graph.


<h3 id="cxn_label"><code>cxn_label</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L309-L311">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_label(
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> str
</code></pre>

Overridable method to return labels for connections.


<h3 id="cxn_edge"><code>cxn_edge</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L313-L315">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_edge(
    left_id: str,
    right_id: str,
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> pydot.Edge
</code></pre>

Overridable method to style a pydot.Edge for connecionts.


<h3 id="add_cxn"><code>add_cxn</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L317-L339">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_cxn(
    graph: pydot.Graph,
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> pydot.Graph
</code></pre>

Process and add a connection to the Graph.

Connections are specified using a `:` delimited set of ids. The first element
is the node (bloq instance). For most bloq instances, the second element is
the port (soquet). The final element is the compass direction of where exactly
the connecting line should be anchored.

For DangleT nodes, there aren't any Soquets so the second element is omitted.

<h3 id="get_graph"><code>get_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L341-L357">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_graph() -> pydot.Graph
</code></pre>

Get the graphviz graph representing the Bloq.

This is the main entry-point to this class.

<h3 id="get_svg_bytes"><code>get_svg_bytes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L359-L361">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg_bytes() -> bytes
</code></pre>

Get the SVG code (as bytes) for drawing the graph.


<h3 id="get_svg"><code>get_svg</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/graphviz.py#L363-L365">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg() -> IPython.display.SVG
</code></pre>

Get an IPython SVG object displaying the graph.




