# get_bloq_callee_counts


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_call_graph.py#L89-L120">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get the direct callees of a bloq and the number of times they are called.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.get_bloq_callee_counts(
    bloq: 'Bloq',
    generalizer: Optional['GeneralizerT'] = None,
    ssa: Optional[<a href="../../qualtran/resource_counting/SympySymbolAllocator.html"><code>qualtran.resource_counting.SympySymbolAllocator</code></a>] = None
) -> List[<a href="../../qualtran/resource_counting/BloqCountT.html"><code>qualtran.resource_counting.BloqCountT</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

This calls `bloq.build_call_graph()` with the correct configuration options.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each callee to consolidate attributes
  that do not affect resource estimates. If the callable
  returns `None`, the bloq is omitted from the counts graph. If a sequence of
  generalizers is provided, each generalizer will be run in order.

`ssa`<a id="ssa"></a>
: A sympy symbol allocator that can be provided if one already exists in your
  computation.




<h2 class="add-link">Returns</h2>


