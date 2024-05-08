# assert_bloq_example_serializes


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L549-L585">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Assert that the BloqExample has consistent serialization.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_bloq_example_serializes(
    bloq_ex: <a href="../../qualtran/BloqExample.html"><code>qualtran.BloqExample</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

This function asserts that the given bloq can be serialized to a proto format and the
corresponding proto can be deserialized back to a bloq which is equal to the original
bloq.

If the given Bloq cannot be serialized / deserialized OR if the deserialized Bloq is not
equal to the given Bloq, then the result is `FAIL`. If the roundtrip succeeds, the result
is `PASS`.

<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>


