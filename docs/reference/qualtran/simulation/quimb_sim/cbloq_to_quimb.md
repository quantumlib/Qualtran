# cbloq_to_quimb


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/simulation/quimb_sim.py#L65-L141">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a composite bloq into a Quimb tensor network.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.quimb_sim.cbloq_to_quimb(
    cbloq: <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    pos: Optional[Dict[BloqInstance, Tuple[float, float]]] = None
) -> Tuple[qtn.TensorNetwork, Dict]
</code></pre>



<!-- Placeholder for "Used in" -->

External indices are the dangling soquets of the compute graph.

<h2 class="add-link">Args</h2>

`cbloq`<a id="cbloq"></a>
: The composite bloq. A composite bloq is a container class analogous to a
  `TensorNetwork`. This function will simply add the tensor(s) for each Bloq
  that constitutes the `CompositeBloq`.

`pos`<a id="pos"></a>
: Optional mapping of each `binst` to (x, y) coordinates which will be converted
  into a `fix` dictionary appropriate for `qtn.TensorNetwork.draw()`.




<h2 class="add-link">Returns</h2>

`tn`<a id="tn"></a>
: The `qtn.TensorNetwork` representing the quantum graph. This is constructed
  by delegating to each bloq's `add_my_tensors` method.

`fix`<a id="fix"></a>
: A version of `pos` suitable for `TensorNetwork.draw()`


