# bloq_to_dense


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_dense.py#L134-L158">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a contracted, dense ndarray representing the composite bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.bloq_to_dense(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    full_flatten: bool = True
) -> <a href="../../../qualtran/testing/NDArray.html"><code>qualtran.testing.NDArray</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This function is also available as the <a href="../../../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a> method.

This function decomposes and flattens a given bloq into a factorized CompositeBloq,
turns that composite bloq into a Quimb tensor network, and contracts it into a dense
matrix.

The returned array will be 0-, 1- or 2- dimensional with indices arranged according to the
bloq's signature. In the case of a 2-dimensional matrix, we follow the
quantum computing / matrix multiplication convention of (right, left) order of dimensions.

For fine-grained control over the tensor contraction, use
`cbloq_to_quimb` and `TensorNetwork.to_dense` directly.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq

`full_flatten`<a id="full_flatten"></a>
: Whether to completely flatten the bloq into the smallest possible
  bloqs. Otherwise, stop flattening if custom tensors are encountered.


