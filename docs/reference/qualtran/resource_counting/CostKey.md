# CostKey
`qualtran.resource_counting.CostKey`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L44-L98">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract base class for different types of costs.

<!-- Placeholder for "Used in" -->

One important aspect of a bloq is the resources required to execute it on an error
corrected quantum computer. Since we're usually trying to minimize these resource requirements
we will generally use the catch-all term "costs".

There are a variety of different types or flavors of costs. Each is represented by an
instance of a sublcass of `CostKey`. For example, gate counts (including T-gate counts),
qubit requirements, and circuit depth are all cost metrics that may be of interest.

Each `CostKey` primarily encodes the behavior required to compute a cost value from a
bloq. Often, these costs are defined recursively: a bloq's costs is some combination
of the costs of the bloqs in its decomposition (i.e. the bloq 'callees'). Implementors
must override the `compute` method to define the cost computation.

Each cost key has an associated CostValT. For example, the CostValT of a "t count"
CostKey could be an integer. For a more complicated gateset, the value could be a mapping
from gate to count. This abstract base class is generic w.r.t. `CostValT`. Subclasses
should have a concrete value type. The `validate_val` method can optionally be overridden
to raise an exception if a bad value type is encountered. The `zero` method must return
the zero (additive identity) cost value of the correct type.

## Methods

<h3 id="compute"><code>compute</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L68-L87">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>compute(
    bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], CostValT]
) -> <a href="../../qualtran/resource_counting.html#CostValT"><code>qualtran.resource_counting.CostValT</code></a>
</code></pre>

Compute this type of cost.

When implementing a new CostKey, this method must be overridden.
Users should not call this method directly. Instead: use the <a href="../../qualtran/resource_counting.html"><code>qualtran.resource_counting</code></a>
functions like `get_cost_value`, `get_cost_cache`, or `query_costs`. These provide
caching, logging, generalizers, and support for static costs.

For recursive computations, use the provided callable to recurse.

Args

`bloq`
: The bloq to compute the cost of.

`get_callee_cost`
: A qualtran-provided function for computing costs for "callees"
  of the bloq; i.e. bloqs in the decomposition. Use this function to accurately
  cache intermediate cost values and respect bloqs' static costs.




Returns




<h3 id="zero"><code>zero</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L89-L91">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>zero() -> <a href="../../qualtran/resource_counting.html#CostValT"><code>qualtran.resource_counting.CostValT</code></a>
</code></pre>

The value corresponding to zero cost.


<h3 id="validate_val"><code>validate_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_costing.py#L93-L98">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_val(
    val: <a href="../../qualtran/resource_counting.html#CostValT"><code>qualtran.resource_counting.CostValT</code></a>
)
</code></pre>

Assert that `val` is a valid `CostValT`.

This method can be optionally overridden to raise an error if an invalid value
is encountered. By default, no validation is performed.



