# do_phased_classical_simulation


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L540-L571">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Do a phased classical simulation of the bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.do_phased_classical_simulation(
    bloq: 'Bloq',
    vals: Mapping[str, 'ClassicalValT'],
    rng: Optional['np.random.Generator'] = None,
    fixed_random_vals: Optional[Dict[int, Any]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This provides a simple interface to `PhasedClassicalSimState`. Advanced users
may wish to use that class directly.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to simulate

`vals`<a id="vals"></a>
: A mapping from input register name to initial classical values. The initial phase is
  assumed to be 1.0.

`rng`<a id="rng"></a>
: A numpy random generator (e.g. from `np.random.default_rng()`). This function
  will use this generator to supply random values from certain phased-classical operations
  like `MeasX`. If not supplied, classical measurements will use a random value.

`fixed_random_vals`<a id="fixed_random_vals"></a>
: A dictionary of instance to values to perform fixed calculation
  for classical values.




<h2 class="add-link">Returns</h2>

`final_vals`<a id="final_vals"></a>
: A mapping of output register name to final classical values.

`phase`<a id="phase"></a>
: The final phase.


