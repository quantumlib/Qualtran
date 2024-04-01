# assert_valid_bloq_decomposition


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L192-L202">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check the validity of a bloq decomposition.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_valid_bloq_decomposition(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Importantly, this does not do any correctness checking -- for that you likely
need to use the simulation utilities provided by the library.

This returns the decomposed, composite bloq on which you can do further testing.