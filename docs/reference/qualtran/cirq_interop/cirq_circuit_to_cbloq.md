# cirq_circuit_to_cbloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L104-L128">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a Cirq circuit into a `CompositeBloq`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.cirq_circuit_to_cbloq(
    circuit: cirq.Circuit
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper. The
resultant composite bloq will represent a unitary with one thru-register
named "qubits" of shape `(n_qubits,)`.