# check_bloq_example_qtyping


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L663-L686">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that the bloq example has valid quantum data types throughout its decomposition.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.check_bloq_example_qtyping(
    bloq_ex: <a href="../../qualtran/BloqExample.html"><code>qualtran.BloqExample</code></a>
) -> Tuple[<a href="../../qualtran/testing/BloqCheckResult.html"><code>qualtran.testing.BloqCheckResult</code></a>, str]
</code></pre>



<!-- Placeholder for "Used in" -->

If the bloq has no decomposition, this check is not applicable. Otherwise: we check the
`connections` in the decomposed bloq with increasing levels of type checking severity.
First, we check loose type checking (allowing conversions between numeric types). A
failure here is returned as a FAIL.

Then `QDTypeCheckingSeverity.ANY` checking (allowing just conversions to and from QAny) and
finally strict checking are performed. Currently, these are coded as an UNVERIFIED bloq
check result.

<h2 class="add-link">Returns</h2>

`result`<a id="result"></a>
: The `BloqCheckResult`.

`msg`<a id="msg"></a>
: A message providing details from the check.


