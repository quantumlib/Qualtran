# GraphvizCallGraph
`qualtran.drawing.GraphvizCallGraph`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L173-L212">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Draw a bloq call graph using Graphviz with additional data.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.bloq_counts_graph.GraphvizCallGraph`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.GraphvizCallGraph(
    g: nx.DiGraph, bloq_data: Optional[Dict['Bloq', Dict[Any, Any]]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Each edge is labeled with the number of times the "caller" (predecessor) bloq calls the
"callee" (successor) bloq.

This class follows the behavior described in https://github.com/quantumlib/Qualtran/issues/791
and will replace `GraphvizCounts` when all bloqs have been migrated to use `__str__()`.

<h2 class="add-link">Args</h2>

`g`<a id="g"></a>
: The call graph, from e.g. <a href="../../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a>.

`bloq_data`<a id="bloq_data"></a>
: A mapping from a bloq to a set of key, value pairs to include in a table
  in each node. The keys and values must support `str()`.




## Methods

<h3 id="get_node_title"><code>get_node_title</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L196-L197">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_title(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as a title of a node.

Override this method for complete control over the titles of nodes.

<h3 id="get_node_properties"><code>get_node_properties</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L199-L212">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_properties(
    b: 'Bloq'
)
</code></pre>

Get graphviz properties for a bloq node representing `b`.

By default, this will craft a label from `get_node_title` and `get_node_details`,
and a rectangular node shape. Override this method to provide further customization.



