# BloqError
`qualtran.BloqError`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/composite_bloq.py#L546-L552">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A value error raised when CompositeBloq conditions are violated.

<!-- Placeholder for "Used in" -->

This error is raised during bloq building using `BloqBuilder`, which checks
for the validity of registers and connections during the building process. This error is
also raised by the validity assertion functions provided in this module.

