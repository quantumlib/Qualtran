# cbloq_as_contracted_tensor


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_quimb.py#L117-L146">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



`add_my_tensors` helper for contracting `cbloq` and adding it as a dense tensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.cbloq_as_contracted_tensor(
    cbloq: <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    incoming: Dict[str, SoquetT],
    outgoing: Dict[str, SoquetT],
    tags
) -> qtn.Tensor
</code></pre>



<!-- Placeholder for "Used in" -->

First, we turn the composite bloq into a TensorNetwork with `cbloq_to_quimb`. Then
we contract it to a dense ndarray. This function returns the dense array as well as
the indices munged from `incoming` and `outgoing` to match the structure of the ndarray.