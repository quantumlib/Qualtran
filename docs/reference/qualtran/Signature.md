# Signature
`qualtran.Signature`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L141-L279">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An ordered sequence of `Register`s that follow the rules for a bloq signature.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.Signature(
    registers: Iterable[<a href="../qualtran/Register.html"><code>qualtran.Register</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->

<a href="../qualtran/Bloq.html#signature"><code>Bloq.signature</code></a> is a property of all bloqs, and should be an object of this type.
It is analogous to a function signature in traditional computing where we specify the
names and types of the expected inputs and outputs.

Each LEFT (including thru) register must have a unique name. Each RIGHT (including thru)
register must have a unique name.

<h2 class="add-link">Args</h2>

`registers`<a id="registers"></a>
: The registers comprising the signature.




## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L160-L170">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>build(
    **registers
) -> 'Signature'
</code></pre>

Construct a Signature comprised of simple thru registers given the register bitsizes.


Args

`registers`
: keyword arguments mapping register name to bitsize. All registers
  will be 0-dimensional and THRU.




<h3 id="build_from_dtypes"><code>build_from_dtypes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L172-L180">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>build_from_dtypes(
    **registers
) -> 'Signature'
</code></pre>

Construct a Signature comprised of simple thru registers given the register dtypes.


Args

`registers`
: keyword arguments mapping register name to QDType. All registers
  will be 0-dimensional and THRU.




<h3 id="lefts"><code>lefts</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L182-L184">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lefts() -> Iterable[<a href="../qualtran/Register.html"><code>qualtran.Register</code></a>]
</code></pre>

Iterable over all registers that appear on the LEFT as input.


<h3 id="rights"><code>rights</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L186-L188">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rights() -> Iterable[<a href="../qualtran/Register.html"><code>qualtran.Register</code></a>]
</code></pre>

Iterable over all registers that appear on the RIGHT as output.


<h3 id="get_left"><code>get_left</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L190-L192">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_left(
    name: str
) -> <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>
</code></pre>

Get a left register by name.


<h3 id="get_right"><code>get_right</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L194-L196">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_right(
    name: str
) -> <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>
</code></pre>

Get a right register by name.


<h3 id="groups"><code>groups</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L198-L207">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>groups() -> Iterable[Tuple[str, List[Register]]]
</code></pre>

Iterate over register groups by name.

Registers with shared names (but differing `side` attributes) can be implicitly grouped.

<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L209-L211">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Signature'
</code></pre>

Swap all RIGHT and LEFT registers in this collection.


<h3 id="n_qubits"><code>n_qubits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L213-L224">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_qubits() -> int
</code></pre>

The number of qubits in the signature.

If the signature has LEFT and RIGHT registers, the number of qubits in the signature
is taken to be the greater of the number of left or right qubits. A bloq with this
signature uses at least this many qubits.

Classical registers are ignored.

<h3 id="n_cbits"><code>n_cbits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L226-L235">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_cbits() -> int
</code></pre>

The number of classical bits in the signature.

If the signature has LEFT and RIGHT registers, the number of classical bits in the signature
is taken to be the greater of the number of left or right cbits. A bloq with this
signature uses at least this many classical bits.

<h3 id="n_bits"><code>n_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L237-L250">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_bits() -> int
</code></pre>

The number of quantum + classical bits in the signature.

If the signature has LEFT and RIGHT registers, the number of bits in the signature
is taken to be the greater of the number of left or right bits. A bloq with this
signature uses at least this many quantum + classical bits.

See Also




<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L263-L264">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key
)
</code></pre>




<h3 id="__contains__"><code>__contains__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L266-L267">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    item: <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>
) -> bool
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L269-L270">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[<a href="../qualtran/Register.html"><code>qualtran.Register</code></a>]
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L272-L273">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/registers.py#L278-L279">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.




