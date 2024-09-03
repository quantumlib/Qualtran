# GraphvizCallGraph
`qualtran.drawing.GraphvizCallGraph`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L178-L352">
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

The constructor of this class assumes you have already generated the call graph as a networkx
graph and constructed any associated data. See the factory method
<a href="../../qualtran/drawing/GraphvizCallGraph.html#from_bloq"><code>GraphvizCallGraph.from_bloq()</code></a> to set up a call graph diagram from a bloq with sensible
defaults.

This class uses a bloq's `__str__` string to title the bloq. Arbitrary additional tabular
data can be provided with `bloq_data`.

This graph drawer is the successor to the `GraphvizCounts` existing drawer,
and will replace `GraphvizCounts` when all bloqs have been migrated to use `__str__()`.

<h2 class="add-link">Args</h2>

`g`<a id="g"></a>
: The call graph, from e.g. <a href="../../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a>.

`bloq_data`<a id="bloq_data"></a>
: A mapping from a bloq to a set of key, value pairs to include in a table
  in each node. The keys and values must support `str()`.




## Methods

<h3 id="format_qubit_count"><code>format_qubit_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L209-L219">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>format_qubit_count(
    val: <a href="../../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>
) -> Dict[str, str]
</code></pre>

Format `QubitCount` cost values as a string.


Args

`val`
: The qubit count value, which should be an integer




Returns




<h3 id="format_qec_gates_cost"><code>format_qec_gates_cost</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L221-L258">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>format_qec_gates_cost(
    val: 'GateCounts', agg: Optional[str] = None
) -> Dict[str, str]
</code></pre>

Format `QECGatesCost` cost values as a string.


Args

`val`
: The qec gate costs value, which should be a `GateCounts` dataclass.

`agg`
: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
  (optionally) aggregate the gate counts. If not specified, the 'factored'
  approach is used where each type of gate is counted individually. See the
  methods on `GateCounts` for more information.




Returns




<h3 id="format_cost_data"><code>format_cost_data</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L260-L297">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>format_cost_data(
    cost_data: Dict['Bloq', Dict['CostKey', 'CostValT']],
    agg_gate_counts: Optional[str] = None
) -> Dict['Bloq', Dict[str, str]]
</code></pre>

Format `cost_data` as human-readable strings.


Args

`cost_data`
: The cost data, likely returned from a call to `query_costs()`. This
  class method will delegate to `format_qubit_count` and `format_qec_gates_cost`
  for `QubitCount` and `QECGatesCost` cost keys, respectively.

`agg_gate_counts`
: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
  (optionally) aggregate the gate counts. If not specified, the 'factored'
  approach is used where each type of gate is counted individually. See the
  methods on `GateCounts` for more information.




Returns




<h3 id="from_bloq"><code>from_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L299-L334">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_bloq(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    *,
    max_depth: Optional[int] = None,
    agg_gate_counts: Optional[str] = None
) -> 'GraphvizCallGraph'
</code></pre>

Draw a bloq call graph.

This factory method will generate a call graph from the bloq, query the `QECGatesCost`
and `QubitCount` costs, format the cost data, and merge it with the call graph
to create a call graph diagram with annotated costs.

For additional customization, users can construct the call graph and bloq data themselves
and use the normal constructor, or provide minor display customizations by
overriding the `format_xxx` class methods.

Args

`bloq`
: The bloq from which we construct the call graph and query the costs.

`max_depth`
: The maximum depth (from the root bloq) of the call graph to draw. Note
  that the cost computations will walk the whole call graph, but only the nodes
  within this depth will be drawn.

`agg_gate_counts`
: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
  (optionally) aggregate the gate counts. If not specified, the 'factored'
  approach is used where each type of gate is counted individually. See the
  methods on `GateCounts` for more information.




Returns




<h3 id="get_node_title"><code>get_node_title</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L336-L337">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_title(
    b: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>

Return text to use as a title of a node.

Override this method for complete control over the titles of nodes.

<h3 id="get_node_properties"><code>get_node_properties</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/bloq_counts_graph.py#L339-L352">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_node_properties(
    b: 'Bloq'
)
</code></pre>

Get graphviz properties for a bloq node representing `b`.

By default, this will craft a label from `get_node_title` and `get_node_details`,
and a rectangular node shape. Override this method to provide further customization.



