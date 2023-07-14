# CirqGateAsBloq
`qualtran.cirq_interop.CirqGateAsBloq`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L51-L101">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A Bloq wrapper around a `cirq.Gate`.

Inherits From: [`Bloq`](../../qualtran/Bloq.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.CirqGateAsBloq(
    gate
)
</code></pre>



<!-- Placeholder for "Used in" -->

This bloq has one thru-register named "qubits", which is a 1D array of soquets
representing individual qubits.



<h2 class="add-link">Attributes</h2>

`gate`<a id="gate"></a>
: &nbsp;

`n_qubits`<a id="n_qubits"></a>
: &nbsp;

`signature`<a id="signature"></a>
: &nbsp;




## Methods

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L61-L62">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="short_name"><code>short_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L64-L66">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>short_name() -> str
</code></pre>




<h3 id="add_my_tensors"><code>add_my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L76-L92">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_my_tensors(
    tn: qtn.TensorNetwork,
    tag: Any,
    *,
    incoming: Dict[str, 'SoquetT'],
    outgoing: Dict[str, 'SoquetT']
)
</code></pre>

Override this method to support native quimb simulation of this Bloq.

This method is responsible for adding a tensor corresponding to the unitary, state, or
effect of the bloq to the provided tensor network `tn`. Often, this method will add
one tensor for a given Bloq, but some bloqs can be represented in a factorized form
requiring the addition of more than one tensor.

If this method is not overriden, the default implementation will try to use the bloq's
decomposition to find a dense representation for this bloq.

Args

`tn`
: The tensor network to which we add our tensor(s)

`tag`
: An arbitrary tag that must be forwarded to `qtn.Tensor`'s `tag` attribute.

`incoming`
: A mapping from register name to SoquetT to order left indices for
  the tensor network.

`outgoing`
: A mapping from register name to SoquetT to order right indices for
  the tensor network.




<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L94-L98">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', qubits: 'CirqQuregT'
) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]
</code></pre>

Override this method to support conversion to a Cirq operation.

If this method is not overriden, the default implementation will wrap this bloq
in a `BloqAsCirqGate` shim.

Args

`qubit_manager`
: A `cirq.QubitManager` for allocating `cirq.Qid`s.

`**cirq_quregs`
: kwargs mapping from this bloq's left register names to an ndarray of
  `cirq.Qid`. The final dimension of this array corresponds to the registers
  `bitsize` size. Any additional dimensions come first and correspond to the
  register `shape` sizes.




Returns

`op`
: A cirq operation corresponding to this bloq acting on the provided cirq qubits or
  None. This method should return None if and only if the bloq instance truly should
  not be included in the Cirq circuit (e.g. for reshaping bloqs). A bloq with no cirq
  equivalent should raise an exception instead.

`cirq_quregs`
: A mapping from this bloq's right register of the same format as the
  `cirq_quregs` argument. The returned dictionary corresponds to the output qubits.




<h3 id="t_complexity"><code>t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/_cirq_interop.py#L100-L101">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'cirq_ft.TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class CirqGateAsBloq.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class CirqGateAsBloq.




