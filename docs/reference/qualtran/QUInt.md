# QUInt
`qualtran.QUInt`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L360-L454">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Unsigned integer of a given width bitsize which wraps around upon overflow.

Inherits From: [`QDType`](../qualtran/QDType.md), [`QCDType`](../qualtran/QCDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.QUInt(
    bitsize
)
</code></pre>



<!-- Placeholder for "Used in" -->

Any intended wrap around effect is expected to be handled by the developer, similar
to an unsigned integer type in C.

Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
bit is at index 0.



<h2 class="add-link">Attributes</h2>

`bitsize`<a id="bitsize"></a>
: The number of qubits used to represent the integer.

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L380-L381">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this dtype is parameterized with symbolic objects.


<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L383-L384">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L386-L389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x: int
) -> List[int]
</code></pre>

Yields individual bits corresponding to binary representation of x


<h3 id="to_bits_array"><code>to_bits_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L391-L412">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits_array(
    x_array: NDArray[np.integer]
) -> NDArray[np.uint8]
</code></pre>

Returns the big-endian bitstrings specified by the given integers.


Args

`x_array`
: An integer or array of unsigned integers.




<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L414-L416">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
) -> int
</code></pre>

Combine individual bits to form x


<h3 id="from_bits_array"><code>from_bits_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L418-L435">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits_array(
    bits_array: NDArray[np.uint8]
) -> NDArray[np.integer]
</code></pre>

Returns the integer specified by the given big-endian bitstrings.


Args

`bits_array`
: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.




Returns




<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L437-L443">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val(
    val: int, debug_str: str = &#x27;val&#x27;
)
</code></pre>

Raises an exception if `val` is not a valid classical value for this type.


Args

`val`
: A classical value that should be in the domain of this QDType.

`debug_str`
: Optional debugging information to use in exception messages.




<h3 id="assert_valid_classical_val_array"><code>assert_valid_classical_val_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L445-L451">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val_array(
    val_array: NDArray[np.integer], debug_str: str = &#x27;val&#x27;
)
</code></pre>

Raises an exception if `val_array` is not a valid array of classical values for this type.

Often, validation on an array can be performed faster than validating each element
individually.

Args

`val_array`
: A numpy array of classical values. Each value should be in the domain
  of this QDType.

`debug_str`
: Optional debugging information to use in exception messages.




<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Check equality and either forward a NotImplemented or return the result negated.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class QUInt.




