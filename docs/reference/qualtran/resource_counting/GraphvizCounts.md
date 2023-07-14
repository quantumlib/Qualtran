# GraphvizCounts
`qualtran.resource_counting.GraphvizCounts`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L218-L274">
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
<p>`qualtran.resource_counting.bloq_counts.GraphvizCounts`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.GraphvizCounts(
    g: nx.DiGraph
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`g`<a id="g"></a>
: The counts graph.




## Methods

<h3 id="get_id"><code>get_id</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L230-L236">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_id(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> str
</code></pre>




<h3 id="get_node_properties"><code>get_node_properties</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L238-L246">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_properties(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Get graphviz properties for a bloq node representing `b`.


<h3 id="add_nodes"><code>add_nodes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L248-L252">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_nodes(
    graph: pydot.Graph
)
</code></pre>

Helper function to add nodes to the pydot graph.


<h3 id="add_edges"><code>add_edges</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L254-L259">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edges(
    graph: pydot.Graph
)
</code></pre>

Helper function to add edges to the pydot graph.


<h3 id="get_graph"><code>get_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L261-L266">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_graph()
</code></pre>

Get the pydot graph.


<h3 id="get_svg_bytes"><code>get_svg_bytes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L268-L270">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg_bytes() -> bytes
</code></pre>

Get the SVG code (as bytes) for drawing the graph.


<h3 id="get_svg"><code>get_svg</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L272-L274">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_svg() -> IPython.display.SVG
</code></pre>

Get an IPython SVG object displaying the graph.




