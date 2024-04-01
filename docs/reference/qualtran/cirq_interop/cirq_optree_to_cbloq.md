# cirq_optree_to_cbloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L365-L476">
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
    signature: Optional[<a href="../../qualtran/Signature.html"><code>qualtran.Signature</code></a>] = None,
    in_quregs: Optional[Dict[str, 'CirqQuregT']] = None,
    out_quregs: Optional[Dict[str, 'CirqQuregT']] = None
) -> <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

 Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper.
 The signature of the resultant CompositeBloq is `signature`, if provided. Otherwise, use
 one thru-register named "qubits" of shape `(n_qubits,)`.

 For multi-dimensional registers and registers with bitsize>1, this function automatically
 splits the input soquets and joins the output soquets to ensure compatibility with the
 flat-list-of-qubits expected by Cirq.

 When specifying a signature, users must also specify the `in_quregs` & `out_quregs` arguments,
 which are mappings of cirq qubits used in the OP-TREE corresponding to the `LEFT` & `RIGHT`
 registers in `signature`. If `signature` has registers with entry

    - `Register('x', QAny(bitsize=2), shape=(3, 4), side=Side.THRU)`
    - `Register('y', QBit(), shape=(10, 20), side=Side.LEFT)`
    - `Register('z', QBit(), shape=(10, 20), side=Side.RIGHT)`

then `in_quregs` should have one entry corresponding to registers `x` and `y` as follows:

    - key='x'; value=`np.array(cirq_qubits_used_for_x, shape=(3, 4, 2))` and
    - key='y'; value=`np.array(cirq_qubits_used_for_y, shape=(10, 20, 1))`.
and `out_quregs` should have one entry corresponding to registers `x` and `z` as follows:

    - key='x'; value=`np.array(cirq_qubits_used_for_x, shape=(3, 4, 2))` and
    - key='z'; value=`np.array(cirq_qubits_used_for_z, shape=(10, 20, 1))`.

Any qubit in `optree` which is not part of `in_quregs` and `out_quregs` is considered to be
allocated & deallocated inside the CompositeBloq and does not show up in it's signature.