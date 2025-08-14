# cbloq_to_superquimb


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_quimb.py#L203-L266">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a composite bloq into a superoperator tensor network.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.cbloq_to_superquimb(
    cbloq: <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    friendly_indices: bool = False
) -> qtn.TensorNetwork
</code></pre>



<!-- Placeholder for "Used in" -->

This simulation strategy can handle non-unitary dynamics, but is more costly.

This function will call <a href="../../../qualtran/Bloq.html#my_tensors"><code>Bloq.my_tensors</code></a> on each subbloq in the composite bloq to add
tensors to a quimb tensor network. This uses ths system+environment strategy for modeling
open system dynamics. In contrast to `cbloq_to_quimb`, each bloq will have
its tensors added twice: once to the part of the network representing the "forward"
wavefunction, and its conjugate added to the part of the network representing the "backward"
part of the wavefunction. If the bloq returns a sentinel value of the `DiscardInd` class,
that particular index is *traced out*: the forward and backward copies of the index are joined.
This corresponds to removing the qubit from the computation and integrating over its possible
values. Arbitrary non-unitary dynamics can be modeled by unitary interaction of the 'system'
with an 'environment' that is traced out.

If a bloq returns a value of type `DiscardInd` in its tensors, this function must be
used. The ordinary `cbloq_to_quimb` will raise an error.

<h2 class="add-link">Args</h2>

`cbloq`<a id="cbloq"></a>
: The composite bloq.

`friendly_indices`<a id="friendly_indices"></a>
: If set to True, the outer indices of the tensor network will be renamed
  from their Qualtran-computer-readable form to human-friendly strings. This may be
  useful if you plan on manually manipulating the resulting tensor network but will
  preclude any further processing by Qualtran functions. The indices are named
  {soq.reg.name}{soq.idx}_{j}{side}{direction}, where j is the individual bit index,
  side is 'l' or 'r' for left or right (resp.), and direction is 'f' or 'b' for the
  forward or backward (adjoint) wavefunctions.


