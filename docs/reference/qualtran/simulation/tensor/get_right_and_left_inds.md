# get_right_and_left_inds


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_dense.py#L67-L117">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return right and left tensor indices.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.get_right_and_left_inds(
    tn: 'qtn.TensorNetwork',
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>
) -> List[List[Soquet]]
</code></pre>



<!-- Placeholder for "Used in" -->

In general, this will be returned as a list of length-2 corresponding
to the right and left indices, respectively. If there *are no* right
or left indices, that entry will be omitted from the returned list.

Right indices come first to match the quantum computing / matrix multiplication
convention where U_tot = U_n ... U_2 U_1.

<h2 class="add-link">Args</h2>

`tn`<a id="tn"></a>
: The tensor network to fetch the outer indices, which won't necessarily be ordered.

`signature`<a id="signature"></a>
: The signature of the bloq used to order the indices.


