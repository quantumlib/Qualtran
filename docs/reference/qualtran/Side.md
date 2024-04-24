# Side
`qualtran.Side`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L29-L42">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Denote LEFT, RIGHT, or THRU registers.

<!-- Placeholder for "Used in" -->

LEFT registers serve as input lines (only) to the Bloq. RIGHT registers are output
lines (only) from the Bloq. THRU registers are both input and output.

Traditional unitary operations will have THRU registers that operate on a collection of
qubits which are then made available to following operations. RIGHT and LEFT registers
imply allocation, deallocation, or reshaping of the registers.



<h2 class="add-link">Class Variables</h2>

LEFT<a id="LEFT"></a>
: `<Side.LEFT: 1>`

RIGHT<a id="RIGHT"></a>
: `<Side.RIGHT: 2>`


