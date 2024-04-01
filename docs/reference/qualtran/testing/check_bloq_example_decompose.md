# check_bloq_example_decompose


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L373-L391">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that the BloqExample has a valid decomposition.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.check_bloq_example_decompose(
    bloq_ex: <a href="../../qualtran/BloqExample.html"><code>qualtran.BloqExample</code></a>
) -> Tuple[<a href="../../qualtran/testing/BloqCheckResult.html"><code>qualtran.testing.BloqCheckResult</code></a>, str]
</code></pre>



<!-- Placeholder for "Used in" -->

This will use `assert_valid_decomposition` which has a variety of sub-checks. A failure
in any of those checks will result in `FAIL`. `DecomposeTypeError` results in a
not-applicable `NA` status. `DecomposeNotImplementedError` results in a `MISSING` status.

<h2 class="add-link">Returns</h2>

`result`<a id="result"></a>
: The `BloqCheckResult`.

`msg`<a id="msg"></a>
: A message providing details from the check.


