# tensor_data_from_unitary_and_signature


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_tensor_data_manipulation.py#L94-L136">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns tensor data respecting `signature` corresponding to `unitary`


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.tensor_data_from_unitary_and_signature(
    unitary: np.ndarray,
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


For a given input unitary, we extract the action of the unitary on a subspace where
input qubits corresponding to LEFT registers and output qubits corresponding to RIGHT
registers in `signature` are 0.

The input unitary is assumed to act on `_n_qubits(signature)`, and thus is of shape
`(2 ** _n_qubits(signature), 2 ** _n_qubits(signature))` where `_n_qubits(signature)`
is `sum(reg.total_bits() for reg in signature)`.

The shape of the returned tensor matches `tensor_shape_from_signature(signature)`.