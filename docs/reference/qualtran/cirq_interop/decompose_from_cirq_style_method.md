# decompose_from_cirq_style_method


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L537-L582">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a `CompositeBloq` decomposition using a cirq-style decompose method.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.decompose_from_cirq_style_method(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    method_name: str = &#x27;decompose_from_registers&#x27;
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

The bloq must have a method with the given name (by default: "decompose_from_registers") that
satisfies the following function signature:

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:

This must yield a list of `cirq.Operation`s using `cirq.Gate.on(...)`, <a href="../../qualtran/Bloq.html#on"><code>Bloq.on(...)</code></a>,
<a href="../../qualtran/GateWithRegisters.html#on_registers"><code>GateWithRegisters.on_registers(...)</code></a>, or <a href="../../qualtran/Bloq.html#on_registers"><code>Bloq.on_registers(...)</code></a>.

If <a href="../../qualtran/Bloq.html#on"><code>Bloq.on()</code></a> is used, the bloqs will be retained in their native form in the returned
composite bloq. If `cirq.Gate.on()` is used, the gates will be wrapped in `CirqGateAsBloq`.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to decompose.

`method_name`<a id="method_name"></a>
: The string name of the method that can be found on the bloq that
  yields the cirq-style decomposition.


