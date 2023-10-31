# DecomposeTypeError
`qualtran.DecomposeTypeError`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L52-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Raised if a decomposition does not make sense in this context.

<!-- Placeholder for "Used in" -->

In contrast to `DecomposeNotImplementedError`, a decomposition does not make sense
in this context. This can be raised if the bloq is "atomic" -- that is, considered part
of the compilation target gateset. This can be raised if certain bloq attributes do not
permit a decomposition, most commonly if an attribute is symbolic.

