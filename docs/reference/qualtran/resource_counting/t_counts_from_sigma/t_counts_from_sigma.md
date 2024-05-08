# t_counts_from_sigma


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/t_counts_from_sigma.py#L41-L55">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.t_counts_from_sigma.t_counts_from_sigma(
    sigma: Mapping['Bloq', SymbolicInt],
    rotation_types: Optional[Tuple[Type['_HasEps'], ...]] = None
) -> <a href="../../../qualtran/cirq_interop/t_complexity_protocol/SymbolicInt.html"><code>qualtran.cirq_interop.t_complexity_protocol.SymbolicInt</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
