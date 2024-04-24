# assert_soquets_belong_to_registers


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L153-L173">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that all soquet's registers make sense.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_soquets_belong_to_registers(
    cbloq: <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

We check that any indexed soquets fit within the bounds of the register and that the
register actually exists on the bloq.