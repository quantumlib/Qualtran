# assert_equivalent_bloq_example_counts


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L433-L504">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Assert that the BloqExample has consistent bloq counts.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_equivalent_bloq_example_counts(
    bloq_ex: <a href="../../qualtran/BloqExample.html"><code>qualtran.BloqExample</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Bloq counts can be annotated directly via the <a href="../../qualtran/Bloq.html#build_call_graph"><code>Bloq.build_call_graph</code></a> override.
They can be inferred from a bloq's decomposition. This function asserts that both
data sources are present and that they produce the same values.

If both sources are present, and they disagree, that results in a `FAIL`. If only one source
is present, an `UNVERIFIED` exception is raised. If neither are present, a `MISSING` result
is raised.

<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>


