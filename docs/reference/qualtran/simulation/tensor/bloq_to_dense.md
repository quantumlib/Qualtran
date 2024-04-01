# bloq_to_dense


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_dense.py#L45-L66">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a contracted, dense ndarray representing the composite bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.bloq_to_dense(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> <a href="../../../qualtran/drawing/musical_score/NDArray.html"><code>qualtran.drawing.musical_score.NDArray</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

The public version of this function is available as the <a href="../../../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a>
method.

This constructs a tensor network and then contracts it according to the cbloq's registers,
i.e. the dangling indices. The returned array will be 0-, 1- or 2- dimensional. If it is
a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
of (right, left) indices.

For more fine grained control over the final shape of the tensor, use
`cbloq_to_quimb` and `TensorNetwork.to_dense` directly.