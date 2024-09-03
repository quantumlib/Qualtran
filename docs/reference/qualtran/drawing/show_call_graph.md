# show_call_graph


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/_show_funcs.py#L95-L125">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Display a graph representation of the call graph.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.show_call_graph(
    item,
    /,
    *,
    max_depth: Optional[int] = None,
    agg_gate_counts: Optional[str] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`item`<a id="item"></a>
: Either a networkx graph or a bloq. If a networkx graph, it should be a "call graph"
  which is passed verbatim to the graph drawer and the additional arguments to this
  function are ignored. If it is a bloq, the factory
  method <a href="../../qualtran/drawing/GraphvizCallGraph.html#from_bloq"><code>GraphvizCallGraph.from_bloq()</code></a> is used to construct the call graph, compute
  relevant costs, and display the call graph annotated with the costs.

`max_depth`<a id="max_depth"></a>
: The maximum depth (from the root bloq) of the call graph to draw. Note
  that the cost computations will walk the whole call graph, but only the nodes
  within this depth will be drawn.

`agg_gate_counts`<a id="agg_gate_counts"></a>
: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
  (optionally) aggregate the gate counts. If not specified, the 'factored'
  approach is used where each type of gate is counted individually.


