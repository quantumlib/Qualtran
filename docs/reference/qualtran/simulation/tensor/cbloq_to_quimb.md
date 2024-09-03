# cbloq_to_quimb


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_quimb.py#L36-L87">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a composite bloq into a tensor network.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.cbloq_to_quimb(
    cbloq: <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
) -> qtn.TensorNetwork
</code></pre>



<!-- Placeholder for "Used in" -->

This function will call <a href="../../../qualtran/Bloq.html#my_tensors"><code>Bloq.my_tensors</code></a> on each subbloq in the composite bloq to add
tensors to a quimb tensor network. This method has no default fallback, so you likely want to
call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
smallest form first. The small bloqs that result from a flattening 1) likely already have
their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
path.