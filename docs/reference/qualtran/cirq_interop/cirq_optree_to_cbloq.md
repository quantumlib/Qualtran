# cirq_optree_to_cbloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L104-L175">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.cirq_optree_to_cbloq(
    optree: cirq.OP_TREE,
    *,
    signature: Optional[<a href="../../qualtran/Signature.html"><code>qualtran.Signature</code></a>] = None
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper.

If `signature` is not None, the signature of the resultant CompositeBloq is `signature`. For
multi-dimensional registers and registers with > 1 bitsize, this function automatically
splits the input soquets into a flat list and joins the output soquets into the correct shape
to ensure compatibility with the flat API expected by Cirq.

If `signature` is None, the resultant composite bloq will have one thru-register named "qubits"
of shape `(n_qubits,)`.