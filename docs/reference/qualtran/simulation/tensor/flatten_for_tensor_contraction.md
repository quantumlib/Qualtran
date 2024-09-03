# flatten_for_tensor_contraction


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_flattening.py#L29-L43">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.flatten_for_tensor_contraction(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    full_flatten: bool = True
) -> <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to flatten.

`full_flatten`<a id="full_flatten"></a>
: Whether to completely flatten the bloq into the smallest possible
  bloqs. Otherwise, stop flattening if custom tensors are encountered.


