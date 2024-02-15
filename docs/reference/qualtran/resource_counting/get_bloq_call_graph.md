# get_bloq_call_graph


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/bloq_counts.py#L181-L227">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recursively build the bloq call graph and call totals.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.resource_counting.bloq_counts.get_bloq_call_graph`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.get_bloq_call_graph(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
    ssa: Optional[<a href="../../qualtran/resource_counting/SympySymbolAllocator.html"><code>qualtran.resource_counting.SympySymbolAllocator</code></a>] = None,
    keep: Optional[Callable[[Bloq], bool]] = None,
    max_depth: Optional[int] = None
) -> Tuple[nx.DiGraph, Dict[Bloq, Union[int, sympy.Expr]]]
</code></pre>



<!-- Placeholder for "Used in" -->

See <a href="../../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a> as a convenient way of calling this function.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to count sub-bloqs.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each (sub)bloq to replace attributes
  that do not affect resource estimates with generic sympy symbols. If the function
  returns `None`, the bloq is omitted from the counts graph. If a sequence of
  generalizers is provided, each generalizer will be run in order.

`ssa`<a id="ssa"></a>
: a `SympySymbolAllocator` that will be passed to the <a href="../../qualtran/Bloq.html#build_call_graph"><code>Bloq.build_call_graph</code></a> method. If
  your `generalizer` function closes over a `SympySymbolAllocator`, provide it here as
  well. Otherwise, we will create a new allocator.

`keep`<a id="keep"></a>
: If this function evaluates to True for the current bloq, keep the bloq as a leaf
  node in the call graph instead of recursing into it.

`max_depth`<a id="max_depth"></a>
: If provided, build a call graph with at most this many layers.




<h2 class="add-link">Returns</h2>

`g`<a id="g"></a>
: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
  the number of times successor bloq is called via its predecessor.

`sigma`<a id="sigma"></a>
: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
  according to `keep` and `max_depth` (if provided) or if a bloq cannot be
  decomposed.


