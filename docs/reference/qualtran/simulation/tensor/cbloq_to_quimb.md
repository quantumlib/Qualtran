# cbloq_to_quimb


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_quimb.py#L40-L89">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a composite bloq into a tensor network.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.cbloq_to_quimb(
    cbloq: <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    friendly_indices: bool = False
) -> qtn.TensorNetwork
</code></pre>



<!-- Placeholder for "Used in" -->

This function will call <a href="../../../qualtran/Bloq.html#my_tensors"><code>Bloq.my_tensors</code></a> on each subbloq in the composite bloq to add
tensors to a quimb tensor network. This method has no default fallback, so you likely want to
call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
smallest form first. The small bloqs that result from a flattening 1) likely already have
their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
path.

<h2 class="add-link">Args</h2>

`cbloq`<a id="cbloq"></a>
: The composite bloq.

`friendly_indices`<a id="friendly_indices"></a>
: If set to True, the outer indices of the tensor network will be renamed
  from their Qualtran-computer-readable form to human-friendly strings. This may be
  useful if you plan on manually manipulating the resulting tensor network but will
  preclude any further processing by Qualtran functions. The indices are named
  {soq.reg.name}{soq.idx}_{j}{side}, where j is the individual bit index and side is 'l'
  or 'r' for left or right, respectively.




<h2 class="add-link">Returns</h2>


