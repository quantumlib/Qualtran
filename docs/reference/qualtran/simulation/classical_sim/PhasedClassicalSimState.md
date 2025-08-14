# PhasedClassicalSimState
`qualtran.simulation.classical_sim.PhasedClassicalSimState`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L392-L501">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A mutable class for classically simulating composite bloqs with phase tracking.

Inherits From: [`ClassicalSimState`](../../../qualtran/simulation/classical_sim/ClassicalSimState.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.PhasedClassicalSimState(
    signature: 'Signature',
    binst_graph: nx.DiGraph,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    *,
    phase: complex = 1.0,
    random_handler: '_ClassicalValHandler'
)
</code></pre>



<!-- Placeholder for "Used in" -->

The convenience function `do_phased_classical_simulation` will simulate a bloq. Use this
class directly for more fine-grained control.

This simulation scheme supports a class of circuits containing only:
 - classical operations corresponding to permutation matrices in the computational basis
 - phase-like operations corresponding to diagonal matrices in the computational basis.

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

`phase`<a id="phase"></a>
: The initial phase. It must be a valid phase: a complex number with unit modulus.






<h2 class="add-link">Attributes</h2>

`soq_assign`<a id="soq_assign"></a>
: An assignment of soquets to classical values.

`last_binst`<a id="last_binst"></a>
: A record of the last bloq instance we processed during simulation.

`phase`<a id="phase"></a>
: The current phase of the simulation state.

`random_handler`<a id="random_handler"></a>
: The classical random number handler to use for use in
  measurement-based outcomes (e.g. MBUC).




## Methods

<h3 id="from_cbloq"><code>from_cbloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L434-L467">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_cbloq(
    cbloq: 'CompositeBloq',
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    rng: Optional['np.random.Generator'] = None,
    fixed_random_vals: Optional[Dict[int, Any]] = None
) -> 'PhasedClassicalSimState'
</code></pre>

Initiate a classical simulation from a CompositeBloq.


Args

`cbloq`
: The composite bloq

`vals`
: A mapping of input register name to classical value to serve as inputs to the
  procedure.

`rng`
: A random number generator to use for classical random values, such a np.random.

`fixed_random_vals`
: A dictionary of bloq instances to values to perform fixed calculation
  for classical values.




Returns






