# ClassicalSimState
`qualtran.simulation.classical_sim.ClassicalSimState`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L193-L389">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A mutable class for classically simulating composite bloqs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.ClassicalSimState(
    signature: 'Signature',
    binst_graph: nx.DiGraph,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    random_handler: '_ClassicalValHandler' = _BannedClassicalValHandler()
)
</code></pre>



<!-- Placeholder for "Used in" -->

Consider using the public method <a href="../../../qualtran/Bloq.html#call_classically"><code>Bloq.call_classically(...)</code></a> for a simple interface
for classical simulation.

The `.step()` and `.finalize()` methods provide fine-grained control over the progress
of the simulation; or the `.simulate()` method will step through the entire composite bloq.

<h2 class="add-link">Args</h2>

`signature`<a id="signature"></a>
: The signature of the composite bloq.

`binst_graph`<a id="binst_graph"></a>
: The directed-graph form of the composite bloq. Consider constructing
  this class with the `.from_cbloq` constructor method to correctly generate the
  binst graph.

`vals`<a id="vals"></a>
: A mapping of input register name to classical value to serve as inputs to the
  procedure.

`random_handler`<a id="random_handler"></a>
: The classical random number handler to use for use in
  measurement-based outcomes (e.g. MBUC).






<h2 class="add-link">Attributes</h2>

`soq_assign`<a id="soq_assign"></a>
: An assignment of soquets to classical values. We store the classical state
  of each soquet (wire connection point in the compute graph) for debugging and/or
  visualization. After stepping through each bloq instance, the right-dangling soquet
  are assigned the output classical values

`last_binst`<a id="last_binst"></a>
: A record of the last bloq instance we processed during simulation. This
  can be used in concert with `.step()` for debugging.




## Methods

<h3 id="from_cbloq"><code>from_cbloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L240-L255">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_cbloq(
    cbloq: 'CompositeBloq',
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]]
) -> 'ClassicalSimState'
</code></pre>

Initiate a classical simulation from a CompositeBloq.


Args

`cbloq`
: The composite bloq

`vals`
: A mapping of input register name to classical value to serve as inputs to the
  procedure.




Returns




<h3 id="step"><code>step</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L328-L357">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>step() -> 'ClassicalSimState'
</code></pre>

Advance the simulation by one bloq instance.

After calling this method, `self.last_binst` will contain the bloq instance that
was just simulated. `self.soq_assign` and any other state variables will be updated.

Returns




<h3 id="finalize"><code>finalize</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L359-L381">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>finalize() -> Dict[str, 'ClassicalValT']
</code></pre>

Finish simulating a composite bloq and extract final values.


Returns

`final_vals`
: The final classical values, keyed by the RIGHT register names of the
  composite bloq.




Raises




<h3 id="simulate"><code>simulate</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L383-L389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate() -> Dict[str, 'ClassicalValT']
</code></pre>

Simulate the composite bloq and return the final values.




