# BQUInt
`qualtran.BQUInt`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L457-L557">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Unsigned integer whose values are bounded within a range.

Inherits From: [`QDType`](../qualtran/QDType.md), [`QCDType`](../qualtran/QCDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.BQUInt(
    bitsize, iteration_length=NOTHING
)
</code></pre>



<!-- Placeholder for "Used in" -->

LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
of values stored as a superposition over the `SELECT` register. Such (nested) coherent
for-loops can be represented using a `Tuple[Register(dtype=BQUInt),
...]` where the i'th entry stores the bitsize and iteration length of i'th
nested for-loop.

One useful feature when processing such nested for-loops is to flatten out a composite index,
represented by a tuple of indices (i, j, ...), one for each selection register into a single
integer that can be used to index a flat target register. An example of such a mapping
function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

<h2 class="add-link">For example</h2>






<h2 class="add-link">Attributes</h2>

`bitsize`<a id="bitsize"></a>
: The number of qubits used to represent the integer.

`iteration_length`<a id="iteration_length"></a>
: The length of the iteration range.

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L519-L520">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this dtype is parameterized with symbolic objects.


<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L526-L529">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L531-L537">View source</a>

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




<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L539-L542">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x: int
) -> List[int]
</code></pre>

Yields individual bits corresponding to binary representation of x


<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L544-L546">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
) -> int
</code></pre>

Combine individual bits to form x


<h3 id="assert_valid_classical_val_array"><code>assert_valid_classical_val_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L548-L554">View source</a>

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

Method generated by attrs for class BQUInt.




