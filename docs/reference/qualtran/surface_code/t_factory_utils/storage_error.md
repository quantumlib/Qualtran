# storage_error


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/t_factory_utils.py#L63-L100">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates several channels each applying the requested pauli error to a single qubit.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.t_factory_utils.storage_error(
    kind: str, probabilities: Sequence[float], qubits: Sequence[cirq.Qid]
) -> Sequence[cirq.Operation]
</code></pre>



<!-- Placeholder for "Used in" -->

Each returned operation is a channel that applies the requested error with
probability `probabilities[i]` to the ith qubit.

The ith qubit gets transformed as

$$
    \rho_i \xrightarrow (1 - p_i) \rho_i + p_i E \rho_i E^\dagger
$$

where $E$ is the requested error (one of X or Z).

<h2 class="add-link">Args</h2>

`kind`<a id="kind"></a>
: The pauli error to apply, one of 'Z' or 'X'.

`probabilities`<a id="probabilities"></a>
: The list probabilities of the channels.

`qubits`<a id="qubits"></a>
: The qubits to apply the error to.




<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>

`ValueError`<a id="ValueError"></a>
: if kind is not 'Z' or 'X'.


