# assert_registers_match_dangling


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L62-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that connections to LeftDangle and RightDangle match the declared registers.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_registers_match_dangling(
    cbloq: <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

All Soquets must be consumed exactly once by a subsequent subbloq or connected explicitly
to either `LeftDangle` or `RightDangle` to indicate the soquet's status as an input
or output, respectively.