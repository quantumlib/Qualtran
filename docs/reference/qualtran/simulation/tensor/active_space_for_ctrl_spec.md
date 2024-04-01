# active_space_for_ctrl_spec


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_tensor_data_manipulation.py#L62-L80">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the "active" subspace corresponding to `signature` and `ctrl_spec`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.active_space_for_ctrl_spec(
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>,
    ctrl_spec: <a href="../../../qualtran/CtrlSpec.html"><code>qualtran.CtrlSpec</code></a>
) -> Tuple[Union[int, slice], ...]
</code></pre>



<!-- Placeholder for "Used in" -->

Assumes first n-registers for `signature` are control registers corresponding to `ctrl_spec`.
Returns a tuple of indices/slices that can be used to address into the ndarray, representing
tensor data of shape `tensor_shape_from_signature(signature)`, and access the active subspace.