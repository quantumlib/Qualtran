# get_right_and_left_inds


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/simulation/quimb_sim.py#L126-L143">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return right and left indices.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.quimb_sim.get_right_and_left_inds(
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>
) -> List[List[Soquet]]
</code></pre>



<!-- Placeholder for "Used in" -->

In general, this will be returned as a list of length-2 corresponding
to the right and left indices, respectively. If there *are* no right
or left indices, that entry will be omitted from the returned list.

Right indices come first to match the quantum computing / matrix multiplication
convention where U_tot = U_n ... U_2 U_1.