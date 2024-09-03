# cirq_gate_to_bloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L324-L420">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



For a given Cirq gate, return an equivalent bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.cirq_gate_to_bloq(
    gate: cirq.Gate
) -> <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This will try to find the idiomatically correct bloq to return. If there is no equivalent
Qualtran bloq for the given Cirq gate, we wrap it in the `CirqGateAsBloq` wrapper class.