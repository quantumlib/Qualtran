# GraphvizCounts
`qualtran.drawing.GraphvizCounts`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L29-L138">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This class turns a bloqs count graph into Graphviz objects and drawings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.bloq_counts_graph.GraphvizCounts`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.GraphvizCounts(
    g: nx.DiGraph
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`g`<a id="g"></a>
: The counts graph.




## Methods

<h3 id="get_id"><code>get_id</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L45-L51">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_id(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> str
</code></pre>




<h3 id="get_node_title"><code>get_node_title</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L53-L58">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_title(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as a title of a node.

Override this method for complete control over the titles of nodes.

<h3 id="abbreviate_field_list"><code>abbreviate_field_list</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L60-L79">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>abbreviate_field_list(
    name_vals: Iterable[Tuple[str, Any]],
    max_field_val_len: int = 12,
    max_detail_fields=5
)
</code></pre>

Helper function for abbreviating a list of key=value representations.

This is used by the default `get_node_details`.

<h3 id="get_node_details"><code>get_node_details</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L81-L96">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_details(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as details for a node.

Override this method for complete control over the details attached to nodes.

<h3 id="get_node_properties"><code>get_node_properties</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L98-L110">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_properties(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Get graphviz properties for a bloq node representing `b`.


<h3 id="add_nodes"><code>add_nodes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L112-L116">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_nodes(
    graph: pydot.Graph
)
</code></pre>

Helper function to add nodes to the pydot graph.


<h3 id="add_edges"><code>add_edges</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L118-L123">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edges(
    graph: pydot.Graph
)
</code></pre>

Helper function to add edges to the pydot graph.


<h3 id="get_graph"><code>get_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L125-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_graph()
</code></pre>

Get the pydot graph.


<h3 id="get_svg_bytes"><code>get_svg_bytes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L132-L134">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg_bytes() -> bytes
</code></pre>

Get the SVG code (as bytes) for drawing the graph.


<h3 id="get_svg"><code>get_svg</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L136-L138">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg() -> IPython.display.SVG
</code></pre>

Get an IPython SVG object displaying the graph.




