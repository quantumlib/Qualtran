# query_costs


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L221-L249">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Compute a selection of costs for a bloq and its callees.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.query_costs(
    bloq: 'Bloq',
    cost_keys: Iterable[<a href="../../qualtran/resource_counting/CostKey.html"><code>qualtran.resource_counting.CostKey</code></a>],
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None
) -> Dict['Bloq', Dict[CostKey, CostValT]]
</code></pre>



<!-- Placeholder for "Used in" -->

This function can be used to annotate a call graph diagram with multiple costs
for each bloq. Specifically, the return value of this function can be used as the
`bloq_data` argument to `GraphvizCallGraph`.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to seed the cost computation.

`cost_keys`<a id="cost_keys"></a>
: A sequence of CostKey that specifies which costs to compute.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each bloq in the call graph to dynamically
  modify attributes. If the function returns `None`, the bloq is ignored in the
  cost computation. If a sequence of generalizers is provided, each generalizer
  will be run in order.




<h2 class="add-link">Returns</h2>


