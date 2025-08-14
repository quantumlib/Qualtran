# check_dtypes_consistent


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1186-L1226">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check if two types are consistent given our current definition on consistent types.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.check_dtypes_consistent(
    dtype_a: <a href="../qualtran/QCDType.html"><code>qualtran.QCDType</code></a>,
    dtype_b: <a href="../qualtran/QCDType.html"><code>qualtran.QCDType</code></a>,
    type_checking_severity: <a href="../qualtran/QDTypeCheckingSeverity.html"><code>qualtran.QDTypeCheckingSeverity</code></a> = <a href="../qualtran/QDTypeCheckingSeverity.html#LOOSE"><code>qualtran.QDTypeCheckingSeverity.LOOSE</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`dtype_a`<a id="dtype_a"></a>
: The dtype to check against the reference.

`dtype_b`<a id="dtype_b"></a>
: The reference dtype.

`type_checking_severity`<a id="type_checking_severity"></a>
: Severity of type checking to perform.




<h2 class="add-link">Returns</h2>


