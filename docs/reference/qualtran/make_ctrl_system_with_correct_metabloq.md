# make_ctrl_system_with_correct_metabloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L677-L716">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The default fallback for `Bloq.make_ctrl_system.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.make_ctrl_system_with_correct_metabloq(
    bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
) -> Tuple['_ControlledBase', 'AddControlledT']
</code></pre>



<!-- Placeholder for "Used in" -->

This intelligently selects the correct implemetation of `_ControlledBase` based
on the control spec.

 - A 1-qubit, positive control (i.e. `CtrlSpec()`) uses `Controlled`, which uses a
   "total control" decomposition.
 - Complex quantum controls (i.e. `CtrlSpec(...)` with quantum data types) uses
   `ControlledViaAnd`, which computes the activation function once and re-uses it
   for each subbloq in the decomposition of `bloq`.