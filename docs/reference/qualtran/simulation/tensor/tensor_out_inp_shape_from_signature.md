# tensor_out_inp_shape_from_signature


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_tensor_data_manipulation.py#L25-L42">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a tuple for tensor data corresponding to signature.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.tensor_out_inp_shape_from_signature(
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]
</code></pre>



<!-- Placeholder for "Used in" -->

Tensor data for a bloq with a given `signature` can be expressed as a ndarray of
shape `out_indices_shape + inp_indices_shape` where

 1. out_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
     corresponding to the RIGHT registers in signature.
 2. inp_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
     corresponding to the LEFT registers in signature.

This method returns a tuple of (out_indices_shape, inp_indices_shape).