# bits_to_ints


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/simulation/classical_sim.py#L39-L51">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the integer specified by the given big-endian bitstrings.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.bits_to_ints(
    bitstrings: Union[Sequence[int], NDArray[np.uint]]
) -> NDArray[np.uint]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bitstrings`<a id="bitstrings"></a>
: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.




<h2 class="add-link">Returns</h2>


