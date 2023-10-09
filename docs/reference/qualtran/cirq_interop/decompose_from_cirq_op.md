# decompose_from_cirq_op


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L353-L380">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a CompositeBloq constructed using Cirq operations obtained via `bloq.as_cirq_op`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.decompose_from_cirq_op(
    bloq: 'Bloq'
) -> 'CompositeBloq'
</code></pre>



<!-- Placeholder for "Used in" -->

This method first checks whether `bloq.signature` is parameterized. If yes, it raises a
NotImplementedError. If not, it uses `cirq_optree_to_cbloq` to wrap the operations obtained
from `bloq.as_cirq_op` into a `CompositeBloq` which has the same signature as `bloq` and returns
the corresponding `CompositeBloq`.