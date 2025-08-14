# call_cbloq_classically


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L504-L532">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Propagate `on_classical_vals` calls through a composite bloq's contents.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.call_cbloq_classically(
    signature: <a href="../../../qualtran/Signature.html"><code>qualtran.Signature</code></a>,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    binst_graph: nx.DiGraph,
    random_handler: '_ClassicalValHandler' = _RandomClassicalValHandler(rng=np.random.default_rng())
) -> Tuple[Dict[str, ClassicalValT], Dict[Soquet, ClassicalValT]]
</code></pre>



<!-- Placeholder for "Used in" -->

While we're handling the plumbing, we also do error checking on the arguments; see
`_update_assign_from_vals`.

<h2 class="add-link">Args</h2>

`signature`<a id="signature"></a>
: The cbloq's signature for validating inputs

`vals`<a id="vals"></a>
: Mapping from register name to classical values

`binst_graph`<a id="binst_graph"></a>
: The cbloq's binst graph.

`random_handler`<a id="random_handler"></a>
: The classical random number handler to use for use in
  measurement-based outcomes (e.g. MBUC).




<h2 class="add-link">Returns</h2>

`final_vals`<a id="final_vals"></a>
: A mapping from register name to output classical values

`soq_assign`<a id="soq_assign"></a>
: An assignment from each soquet to its classical value. Soquets
  corresponding to thru registers will be mapped to the *output* classical
  value.


