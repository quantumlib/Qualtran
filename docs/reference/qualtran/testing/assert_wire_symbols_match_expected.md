# assert_wire_symbols_match_expected


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L230-L258">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Assert a bloq's wire symbols match the expected ones.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_wire_symbols_match_expected(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    expected_ws: List[Union[str, WireSymbol]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

For multi-dimensional registers (with a shape), this will iterate
through the register indices (see numpy.ndindices for iteration order).

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: the bloq whose wire symbols we want to check.

`expected_ws`<a id="expected_ws"></a>
: A list of the expected wire symbols or their associated text.


