# bloq_to_dense


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_dense.py#L132-L178">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a contracted, dense ndarray representing the composite bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.bloq_to_dense(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    full_flatten: bool = True,
    superoperator: bool = False
) -> <a href="../../../qualtran/testing/NDArray.html"><code>qualtran.testing.NDArray</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This function is also available as the <a href="../../../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a> method.

This function decomposes and flattens a given bloq into a factorized CompositeBloq,
turns that composite bloq into a Quimb tensor network, and contracts it into a dense
ndarray.

The returned array will be 0-, 1-, 2-, or 4-dimensional with indices arranged according to the
bloq's signature and the type of simulation requested via the `superoperator` flag.

If `superoperator` is set to False (the default), a pure-state tensor network will be
constructed.
 - If `bloq` has all thru-registers, the dense tensor will be 2-dimensional with shape `(n, n)`
   where `n` is the number of bits in the signature. We follow the linear algebra convention
   and order the indices as (right, left) so the matrix-vector product can be used to evolve
   a state vector.
 - If `bloq` has all left- or all right-registers, the tensor will be 1-dimensional with
   shape `(n,)`. Note that we do not distinguish between 'row' and 'column' vectors in this
   function.
 - If `bloq` has no external registers, the contracted form is a 0-dimensional complex number.

If `superoperator` is set to True, an open-system tensor network will be constructed.
 - States result in a 2-dimensional density matrix with indices (right_forward, right_backward)
   or (left_forward, left_backward) depending on whether they're input or output states.
 - Operations result in a 4-dimensional tensor with indices (right_forward, right_backward,
   left_forward, left_backward).

For fine-grained control over the tensor contraction, use
`cbloq_to_quimb` and `TensorNetwork.to_dense` directly.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq

`full_flatten`<a id="full_flatten"></a>
: Whether to completely flatten the bloq into the smallest possible
  bloqs. Otherwise, stop flattening if custom tensors are encountered.

`superoperator`<a id="superoperator"></a>
: If toggled to True, do an open-system simulation. This supports
  non-unitary operations like measurement, but is more costly and results in
  higher-dimension resultant tensors.


