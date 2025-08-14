# QFxp
`qualtran.QFxp`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L560-L790">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Fixed point type to represent real numbers.

Inherits From: [`QDType`](../qualtran/QDType.md), [`QCDType`](../qualtran/QCDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.QFxp(
    bitsize, num_frac, signed=attr_dict[&#x27;signed&#x27;].default
)
</code></pre>



<!-- Placeholder for "Used in" -->

A real number can be approximately represented in fixed point using `num_int`
bits for the integer part and `num_frac` bits for the fractional part. If the
real number is signed we store negative values in two's complement form. The first
bit can therefore be treated as the sign bit in such cases (0 for +, 1 for -).
In total there are `bitsize = (num_int + num_frac)` bits used to represent the number.
E.g. Using `(bitsize = 8, num_frac = 6, signed = False)` then
$\pi \approx 3.140625 = 11.001001$, where the . represents the decimal place.

We can specify a fixed point real number by the tuple bitsize, num_frac and
signed, with num_int determined as `(bitsize - num_frac)`.


### Classical Simulation

To hook into the classical simulator, we use fixed-width integers to represent
values of this type. See `to_fixed_width_int` for details.
In particular, the user should call <a href="../qualtran/QFxp.html#to_fixed_width_int"><code>QFxp.to_fixed_width_int(float_value)</code></a>
before passing a value to `bloq.call_classically`.

The corresponding raw qdtype is either an QUInt (when `signed=False`) or
QInt (when `signed=True`) of the same bitsize. This is the data type used
to represent classical values during simulation, and convert to and from bits
for intermediate values.

For example, QFxp(6, 4) has 2 int bits and 4 frac bits, and the corresponding
int type is QUInt(6). So a true classical value of `10.0011` will have a raw
integer representation of `100011`.

See https://github.com/quantumlib/Qualtran/issues/1219 for discussion on alternatives
and future upgrades.




<h2 class="add-link">Attributes</h2>

`bitsize`<a id="bitsize"></a>
: The total number of qubits used to represent the integer and
  fractional part combined.

`num_frac`<a id="num_frac"></a>
: The number of qubits used to represent the fractional part of the real number.

`signed`<a id="signed"></a>
: Whether the number is signed or not.

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_int`<a id="num_int"></a>
: Number of bits for the integral part.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L625-L626">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this dtype is parameterized with symbolic objects.


<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L636-L641">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[int]
</code></pre>

Use the classical domain for the underlying raw integer type.

See class docstring section on "Classical Simulation" for more details.

<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L643-L648">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x
) -> List[int]
</code></pre>

Use the underlying raw integer type.

See class docstring section on "Classical Simulation" for more details.

<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L650-L655">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
)
</code></pre>

Use the underlying raw integer type.

See class docstring section on "Classical Simulation" for more details.

<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L657-L662">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val(
    val: int, debug_str: str = &#x27;val&#x27;
)
</code></pre>

Verify using the underlying raw integer type.

See class docstring section on "Classical Simulation" for more details.

<h3 id="to_fixed_width_int"><code>to_fixed_width_int</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L664-L690">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_fixed_width_int(
    x: Union[float, Fxp],
    *,
    require_exact: bool = False,
    complement: bool = True
) -> int
</code></pre>

Returns the interpretation of the binary representation of `x` as an integer.

See class docstring section on "Classical Simulation" for more details on
the choice of this representation.

The returned value is an integer equal to `round(x * 2**self.num_frac)`.
That is, the input value `x` is converted to a fixed-point binary value
of `self.num_int` integral bits and `self.num_frac` fractional bits,
and then re-interpreted as an integer by dropping the decimal point.

For example, consider `QFxp(6, 4).to_fixed_width_int(1.5)`. As `1.5` is `0b01.1000`
in this representation, the returned value would be `0b011000` = 24.

For negative values, we use twos complement form. So in
`QFxp(6, 4, signed=True).to_fixed_width_int(-1.5)`, the input is `0b10.1000`,
which is interpreted as `0b101000` = -24.

Args

`x`
: input floating point value

`require_exact`
: Raise `ValueError` if `x` cannot be exactly represented.

`complement`
: Use twos-complement rather than sign-magnitude representation of negative values.




<h3 id="float_from_fixed_width_int"><code>float_from_fixed_width_int</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L692-L701">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>float_from_fixed_width_int(
    x: int
) -> float
</code></pre>

Helper to convert from the fixed-width-int representation to a true floating point value.

Here `x` is the internal value used by the classical simulator.
See `to_fixed_width_int` for conventions.

See class docstring section on "Classical Simulation" for more details on
the choice of this representation.

<h3 id="fxp_dtype_template"><code>fxp_dtype_template</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L709-L745">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fxp_dtype_template() -> Fxp
</code></pre>

A template of the `Fxp` data type for classical values.

To construct an `Fxp` with this config, one can use:
`Fxp(float_value, like=QFxp(...).fxp_dtype_template)`,
or given an existing value `some_fxp_value: Fxp`:
`some_fxp_value.like(QFxp(...).fxp_dtype_template)`.

The following Fxp configuration is used:
 - op_sizing='same' and const_op_sizing='same' ensure that the returned
   object is not resized to a bigger fixed point number when doing
   operations with other Fxp objects.
 - shifting='trunc' ensures that when shifting the Fxp integer to
   left / right; the digits are truncated and no rounding occurs
 - overflow='wrap' ensures that when performing operations where result
   overflows, the overflowed digits are simply discarded.

Support for `fxpmath.Fxp` is experimental, and does not hook into the classical
simulator protocol. Once the library choice for fixed-point classical real
values is finalized, the code will be updated to use the new functionality
instead of delegating to raw integer values (see above).

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

Method generated by attrs for class QFxp.




