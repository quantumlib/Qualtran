# t_counts_from_sigma


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/t_counts_from_sigma.py#L43-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.t_counts_from_sigma.t_counts_from_sigma(
    sigma: Dict['Bloq', Union[int, 'sympy.Expr']],
    rotation_types: Optional[Tuple['_HasEps', ...]] = None
) -> <a href="../../../qualtran/resource_counting/symbolic_counting_utils/SymbolicInt.html"><code>qualtran.resource_counting.symbolic_counting_utils.SymbolicInt</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
