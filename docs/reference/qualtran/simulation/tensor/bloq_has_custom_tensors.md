# bloq_has_custom_tensors


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_flattening.py#L18-L26">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Whether this bloq declares custom tensors by overriding `.add_my_tensors(...)`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.bloq_has_custom_tensors(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

This is a heuristic that checks that the method is overriden. This is used as the
flattening predicate in `flatten_for_tensor_contraction`.