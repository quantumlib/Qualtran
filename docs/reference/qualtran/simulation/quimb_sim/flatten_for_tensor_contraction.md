# flatten_for_tensor_contraction


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/simulation/quimb_sim.py#L48-L57">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.quimb_sim.flatten_for_tensor_contraction(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    max_depth: int = 1000
) -> <a href="../../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

By default, bloqs without custom tensors will be contracted to a dense tensor using their
decomposition and then that dense tensor will be used in the enclosing tensor network.
To allow a more efficient contraction ordering, use this function to decompose-and-flatten
as much as possible before starting the tensor contraction.