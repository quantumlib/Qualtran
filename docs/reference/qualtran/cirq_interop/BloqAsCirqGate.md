# BloqAsCirqGate
`qualtran.cirq_interop.BloqAsCirqGate`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_bloq_to_cirq.py#L74-L191">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A shim for using bloqs in a Cirq circuit.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.BloqAsCirqGate(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to wrap.






<h2 class="add-link">Attributes</h2>

`bloq`<a id="bloq"></a>
: The bloq we're wrapping.




## Methods

<h3 id="bloq_on"><code>bloq_on</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_bloq_to_cirq.py#L96-L119">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>bloq_on(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    cirq_quregs: Dict[str, 'CirqQuregT'],
    qubit_manager: cirq.QubitManager
) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]
</code></pre>

Shim `bloq` into a cirq gate and call it on `cirq_quregs`.

This is used as a default implementation for <a href="../../qualtran/Bloq.html#as_cirq_op"><code>Bloq.as_cirq_op</code></a> if a native
cirq conversion is not specified.

Args

`bloq`
: The bloq to be wrapped with `BloqAsCirqGate`

`cirq_quregs`
: The cirq qubit registers on which we call the gate. Should correspond to
  registers in `self.bloq.signature.lefts()`.

`qubit_manager`
: A `cirq.QubitManager` to allocate new qubits.




Returns

`op`
: A cirq operation whose gate is the `BloqAsCirqGate`-wrapped version of `bloq`.

`cirq_quregs`
: The output cirq qubit registers.




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_bloq_to_cirq.py#L167-L177">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power, modulo=None
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_bloq_to_cirq.py#L179-L182">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="num_qubits"><code>num_qubits</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.




