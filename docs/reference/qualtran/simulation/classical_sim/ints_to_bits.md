# ints_to_bits


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L55-L71">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the big-endian bitstrings specified by the given integers.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.ints_to_bits(
    x: Union[int, np.integer, Sequence[int], NDArray[np.integer]], w: int
) -> NDArray[np.uint8]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`x`<a id="x"></a>
: An integer or array of unsigned integers.

`w`<a id="w"></a>
: The bit width of the returned bitstrings.


