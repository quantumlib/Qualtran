# assert_soquets_used_exactly_once


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L158-L180">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that all soquets are used once and only once.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_soquets_used_exactly_once(
    cbloq: <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

Each bloq's register produces prod(reg.shape) soquets which must be consumed
once and only once.