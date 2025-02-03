# bloq_to_dense_via_classical_action


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_tensor_from_classical.py#L78-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a contracted, dense ndarray representing the bloq, using its classical action.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.bloq_to_dense_via_classical_action(
    bloq: 'Bloq'
) -> <a href="../../../qualtran/testing/NDArray.html"><code>qualtran.testing.NDArray</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq




<h2 class="add-link">Raises</h2>

`ValueError`<a id="ValueError"></a>
: if the bloq does not have a classical action.


