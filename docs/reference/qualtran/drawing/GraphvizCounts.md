# GraphvizCounts
`qualtran.drawing.GraphvizCounts`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L105-L170">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Draw a bloq call graphs using Graphviz.

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

Each node is a bloq with a `bloq.pretty_name()` label and an automatically-determined
"details" string based on the bloqs attributes. For non-attrs classes, classes with
a large number of fields, or classes where the fields' string representations are long;
the details string will be abbreviated.

Each edge is labeled with the number of times the "caller" (predecessor) bloq calls the
"callee" (successor) bloq.

<h2 class="add-link">Args</h2>

`g`<a id="g"></a>
: The call graph, from e.g. <a href="../../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a>.




<h2 class="add-link">See Also</h2>




## Methods

<h3 id="get_node_title"><code>get_node_title</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L130-L131">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_title(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as a title of a node.

Override this method for complete control over the titles of nodes.

<h3 id="abbreviate_field_list"><code>abbreviate_field_list</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L133-L154">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L156-L170">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_details(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as details for a node.

Override this method for complete control over the details attached to nodes.



