# add_ints


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/classical_sim.py#L242-L269">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs addition modulo $2^\mathrm{num\_bits}$ of (un)signed in a reversible way.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.classical_sim.add_ints(
    a: int, b: int, *, num_bits: Optional[int] = None, is_signed: bool = False
) -> int
</code></pre>



<!-- Placeholder for "Used in" -->

Addition of signed integers can result in an overflow. In most classical programming languages (e.g. C++)
what happens when an overflow happens is left as an implementation detail for compiler designers. However,
for quantum subtraction, the operation should be unitary and that means that the unitary of the bloq should
be a permutation matrix.

If we hold `a` constant then the valid range of values of $b \in [-2^{\mathrm{num\_bits}-1}, 2^{\mathrm{num\_bits}-1})$
gets shifted forward or backward by `a`. To keep the operation unitary overflowing values wrap around. This is the same
as moving the range $2^\mathrm{num\_bits}$ by the same amount modulo $2^\mathrm{num\_bits}$. That is add
$2^{\mathrm{num\_bits}-1})$ before addition modulo and then remove it.

<h2 class="add-link">Args</h2>

`a`<a id="a"></a>
: left operand of addition.

`b`<a id="b"></a>
: right operand of addition.

`num_bits`<a id="num_bits"></a>
: optional num_bits. When specified addition is done in the interval [0, 2**num_bits) or
  [-2**(num_bits-1), 2**(num_bits-1)) based on the value of `is_signed`.

`is_signed`<a id="is_signed"></a>
: boolean whether the numbers are unsigned or signed ints. This value is only used when
  `num_bits` is provided.


