# get_cost_cache


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L184-L218">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Build a cache of cost values for the bloq and its callees.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.get_cost_cache(
    bloq: 'Bloq',
    cost_key: CostKey[CostValT],
    costs_cache: Optional[Dict['Bloq', CostValT]] = None,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None
) -> Dict['Bloq', CostValT]
</code></pre>



<!-- Placeholder for "Used in" -->

This can be useful to inspect how callees' costs flow upwards in a given cost computation.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to seed the cost computation.

`cost_key`<a id="cost_key"></a>
: A CostKey that specifies which cost to compute.

`costs_cache`<a id="costs_cache"></a>
: If provided, use this dictionary for initial cached cost values. Values in this
  dictionary will be preferred over computed values (even if they disagree). This
  dictionary will be mutated by the function. This dictionary will be returned by the
  function.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each bloq in the call graph to dynamically
  modify attributes. If the function returns `None`, the bloq is ignored in the
  cost computation. If a sequence of generalizers is provided, each generalizer
  will be run in order.




<h2 class="add-link">Returns</h2>


