# get_bloq_counts_graph


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L139-L177">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recursively gather bloq counts.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.resource_counting.bloq_counts.get_bloq_counts_graph`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.get_bloq_counts_graph(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    generalizer: Callable[[Bloq], Optional[Bloq]] = None,
    ssa: Optional[<a href="../../qualtran/resource_counting/SympySymbolAllocator.html"><code>qualtran.resource_counting.SympySymbolAllocator</code></a>] = None,
    keep: Optional[Sequence[Bloq]] = None
) -> Tuple[nx.DiGraph, Dict[Bloq, Union[int, sympy.Expr]]]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to count sub-bloqs.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each (sub)bloq to replace attributes
  that do not affect resource estimates with generic sympy symbols. If this function
  returns `None`, the bloq is ommitted from the counts graph.

`ssa`<a id="ssa"></a>
: a `SympySymbolAllocator` that will be passed to the <a href="../../qualtran/Bloq.html#bloq_counts"><code>Bloq.bloq_counts</code></a> methods. If
  your `generalizer` function closes over a `SympySymbolAllocator`, provide it here as
  well. Otherwise, we will create a new allocator.

`keep`<a id="keep"></a>
: Stop recursing and keep these bloqs as leaf nodes in the counts graph. Otherwise,
  leaf nodes are those without a decomposition.




<h2 class="add-link">Returns</h2>

`g`<a id="g"></a>
: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' counts
  how many of the successor bloqs are used in the decomposition of the predecessor
  bloq(s).

`sigma`<a id="sigma"></a>
: A mapping from leaf bloqs to their total counts.


