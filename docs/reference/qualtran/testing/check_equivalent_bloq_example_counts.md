# check_equivalent_bloq_example_counts


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L500-L522">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that the BloqExample has consistent bloq counts.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.check_equivalent_bloq_example_counts(
    bloq_ex: <a href="../../qualtran/BloqExample.html"><code>qualtran.BloqExample</code></a>
) -> Tuple[<a href="../../qualtran/testing/BloqCheckResult.html"><code>qualtran.testing.BloqCheckResult</code></a>, str]
</code></pre>



<!-- Placeholder for "Used in" -->

Bloq counts can be annotated directly via the <a href="../../qualtran/Bloq.html#build_call_graph"><code>Bloq.build_call_graph</code></a> override.
They can be inferred from a bloq's decomposition. This function checks that both
data sources are present and that they produce the same values.

If both sources are present, and they disagree, that results in a `FAIL`. If only one source
is present, an `UNVERIFIED` result is returned. If neither are present, a `MISSING` result
is returned.

<h2 class="add-link">Returns</h2>

`result`<a id="result"></a>
: The `BloqCheckResult`.

`msg`<a id="msg"></a>
: A message providing details from the check.


