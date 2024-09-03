# RotationCostModel
`qualtran.surface_code.RotationCostModel`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/rotation_cost_model.py#L24-L33">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Analytical estimate of the complexity of approximating a rotation given an error budget.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.rotation_cost_model.RotationCostModel`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="rotation_cost"><code>rotation_cost</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/rotation_cost_model.py#L27-L29">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>rotation_cost(
    error_budget: float
) -> <a href="../../qualtran/resource_counting/GateCounts.html"><code>qualtran.resource_counting.GateCounts</code></a>
</code></pre>

Cost of a single rotation.


<h3 id="preparation_overhead"><code>preparation_overhead</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/rotation_cost_model.py#L31-L33">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>preparation_overhead(
    error_budget
) -> <a href="../../qualtran/resource_counting/GateCounts.html"><code>qualtran.resource_counting.GateCounts</code></a>
</code></pre>

Cost of preparation circuit.




