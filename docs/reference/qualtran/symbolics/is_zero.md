# is_zero


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/symbolics/math_funcs.py#L318-L327">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



check if a symbolic integer is zero


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.symbolics.math_funcs.is_zero`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.symbolics.is_zero(
    x: <a href="../../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->


If it returns True, then the value is definitely 0.
If it returns False, then the value is either non-zero,
or could not be symbolically symplified to a zero.