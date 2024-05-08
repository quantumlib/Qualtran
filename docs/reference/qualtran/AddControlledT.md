# AddControlledT
`qualtran.AddControlledT`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L243-L263">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The signature for the `add_controlled` callback part of `ctrl_system`.

<!-- Placeholder for "Used in" -->

See <a href="../qualtran/Bloq.html#get_ctrl_system"><code>Bloq.get_ctrl_system</code></a> for details.

<h2 class="add-link">Args</h2>

`bb`<a id="bb"></a>
: A bloq builder to use for adding.

`ctrl_soqs`<a id="ctrl_soqs"></a>
: The soquets that represent the control lines. These must be compatible with
  the ControlSpec; specifically with the control registers implied
  by `activation_function_dtypes`.

`in_soqs`<a id="in_soqs"></a>
: The soquets that plug in to the normal, uncontrolled bloq.




<h2 class="add-link">Returns</h2>

`ctrl_soqs`<a id="ctrl_soqs"></a>
: The output control soquets.

`out_soqs`<a id="out_soqs"></a>
: The output soquets from the uncontrolled bloq.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L260-L263">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    bb: 'BloqBuilder',
    ctrl_soqs: Sequence['SoquetT'],
    in_soqs: Dict[str, 'SoquetT']
) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]
</code></pre>

Call self as a function.




