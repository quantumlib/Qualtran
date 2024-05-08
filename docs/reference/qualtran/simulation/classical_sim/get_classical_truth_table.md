# get_classical_truth_table


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L224-L258">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get a 'truth table' for a classical-reversible bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.get_classical_truth_table(
    bloq: 'Bloq'
) -> Tuple[List[str], List[str], List[Tuple[Sequence[Any], Sequence[Any]]]]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The classical-reversible bloq to create a truth table for.




<h2 class="add-link">Returns</h2>

`in_names`<a id="in_names"></a>
: The names of the left, input registers to serve as truth table headings for
  the input side of the truth table.

`out_names`<a id="out_names"></a>
: The names of the right, output registers to serve as truth table headings
  for the output side of the truth table.

`truth_table`<a id="truth_table"></a>
: A list of table entries. Each entry is a tuple of (in_vals, out_vals).
  The vals sequences are ordered according to the `in_names` and `out_names` return
  values.


