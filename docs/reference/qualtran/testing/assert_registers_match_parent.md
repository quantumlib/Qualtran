# assert_registers_match_parent


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L37-L55">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that the registers following decomposition match those of the original bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_registers_match_parent(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This is a strict condition of the `decompose_bloq()` protocol. A decomposition is only
valid if it takes exactly the same inputs and outputs.

This returns the decomposed bloq for further checking.