# BloqBuilder
`qualtran.BloqBuilder`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L713-L1130">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A builder class for constructing a `CompositeBloq`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.BloqBuilder(
    add_registers_allowed: bool = True
)
</code></pre>



<!-- Placeholder for "Used in" -->

Users may instantiate this class directly or use its methods by
overriding <a href="../qualtran/Bloq.html#build_composite_bloq"><code>Bloq.build_composite_bloq</code></a>.

When overriding `build_composite_bloq`, the Bloq class will ensure that the bloq under
construction has the correct registers: namely, those of the decomposed bloq and parent
bloq are the same. This affords some additional error checking.
Initial soquets are passed as **kwargs (by register name) to the `build_composite_bloq` method.

When using this class directly, you must call `add_register` to set up the composite bloq's
registers. When adding a LEFT or THRU register, the method will return soquets to be
used when adding more bloqs. Adding a THRU or RIGHT register can enable more checks during
`finalize()`.

<h2 class="add-link">Args</h2>

`add_registers_allowed`<a id="add_registers_allowed"></a>
: Whether we allow the addition of registers during bloq building.
  This affords some additional error checking if set to `False` but you must specify
  all registers ahead-of-time.




## Methods

<h3 id="add_register_from_dtype"><code>add_register_from_dtype</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L750-L790">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_register_from_dtype(
    reg: Union[str, <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>],
    dtype: Optional[<a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a>] = None
) -> Union[None, <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>]
</code></pre>

Add a new typed register to the composite bloq being built.

If this bloq builder was constructed with `add_registers_allowed=False`,
this operation is not allowed.

Args

`reg`
: Either the register or a register name. If this is a register, then `bitsize`
  must also be provided and a default THRU register will be added.

`dtype`
: If `reg` is a register name, this is the quantum data type for the added register.
  Otherwise, this must not be provided.




Returns




<h3 id="add_register"><code>add_register</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L800-L821">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_register(
    reg: Union[str, <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>],
    bitsize: Optional[int] = None
) -> Union[None, <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>]
</code></pre>

Add a new register to the composite bloq being built.

If this bloq builder was constructed with `add_registers_allowed=False`,
this operation is not allowed.

Args

`reg`
: Either the register or a register name. If this is a register, then `bitsize`
  must also be provided and a default THRU register will be added.

`bitsize`
: If `reg` is a register name, this is the bitsize for the added register.
  Otherwise, this must not be provided.




Returns




<h3 id="from_signature"><code>from_signature</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L823-L847">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_signature(
    signature: <a href="../qualtran/Signature.html"><code>qualtran.Signature</code></a>,
    add_registers_allowed: bool = False
) -> Tuple['BloqBuilder', Dict[str, SoquetT]]
</code></pre>

Construct a BloqBuilder with a pre-specified signature.

This is safer if e.g. you're decomposing an existing Bloq and need the signatures
to match. This constructor is used by <a href="../qualtran/Bloq.html#decompose_bloq"><code>Bloq.decompose_bloq()</code></a>.

<h3 id="map_soqs"><code>map_soqs</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L849-L867">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>map_soqs(
    soqs: Dict[str, SoquetT], soq_map: Iterable[Tuple[SoquetT, SoquetT]]
) -> Dict[str, SoquetT]
</code></pre>

Map `soqs` according to `soq_map`.

See <a href="../qualtran/CompositeBloq.html#iter_bloqsoqs"><code>CompositeBloq.iter_bloqsoqs</code></a> for example code.

Args

`soqs`
: A soquet dictionary mapping register names to Soquets or arrays
  of Soquets. The values of this dictionary will be mapped.

`soq_map`
: An iterable of (old_soq, new_soq) tuples that inform how to
  perform the mapping. Note that this is a list of tuples (not a dictionary)
  because `old_soq` may be an unhashable numpy array of Soquet.




Returns




<h3 id="add_t"><code>add_t</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L896-L916">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_t(
    bloq: <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    **in_soqs
) -> Tuple[SoquetT, ...]
</code></pre>

Add a new bloq instance to the compute graph and always return a tuple of soquets.

This method will always return a tuple of soquets. See <a href="../qualtran/BloqBuilder.html#add_d"><code>BloqBuilder.add_d(..)</code></a> for a
method that returns a dictionary of soquets. See <a href="../qualtran/BloqBuilder.html#add"><code>BloqBuilder.add(..)</code></a> for a return
type that depends on the arity of the bloq.

Args

`bloq`
: The bloq representing the operation to add.

`**in_soqs`
: Keyword arguments mapping the new bloq's register names to input
  `Soquet`s or an array thereof. This is likely the output soquets from a prior
  operation.




Returns




<h3 id="add_d"><code>add_d</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L918-L935">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_d(
    bloq: <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    **in_soqs
) -> Dict[str, SoquetT]
</code></pre>

Add a new bloq instance to the compute graph and return new soquets as a dictionary.

This method returns a dictionary of soquets. See <a href="../qualtran/BloqBuilder.html#add_t"><code>BloqBuilder.add_t(..)</code></a> for a method
that returns an ordered tuple of soquets. See <a href="../qualtran/BloqBuilder.html#add"><code>BloqBuilder.add(..)</code></a> for a return
type that depends on the arity of the bloq.

Args

`bloq`
: The bloq representing the operation to add.

`**in_soqs`
: Keyword arguments mapping the new bloq's register names to input
  `Soquet`s or an array thereof. This is likely the output soquets from a prior
  operation.




Returns




<h3 id="add"><code>add</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L937-L969">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add(
    bloq: <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    **in_soqs
)
</code></pre>

Add a new bloq instance to the compute graph.

This is the primary method for building a composite bloq. Each call to `add` adds a
new bloq instance to the compute graph, wires up the soquets from prior operations
into the new bloq, and returns new soquets to be used for subsequent bloqs.

This method will raise a `BloqError` if the addition is invalid. Soquets must be
used exactly once and soquets must match the `Register` specifications of the bloq.

See also `add_t` or `add_d` for versions of this function that return output soquets
in a structured way that may be more appropriate for programmatic adding of bloqs.

Args

`bloq`
: The bloq representing the operation to add.

`**in_soqs`
: Keyword arguments mapping the new bloq's register names to input
  `Soquet`s or an array thereof. This is likely the output soquets from a prior
  operation.




Returns




<h3 id="add_from"><code>add_from</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L995-L1027">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_from(
    bloq: <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    **in_soqs
) -> Tuple[SoquetT, ...]
</code></pre>

Add all the sub-bloqs from `bloq` to the composite bloq under construction.


Args

`bloq`
: Where to add from. If this is a composite bloq, use its contents directly.
  Otherwise, we call `decompose_bloq()` first.

`in_soqs`
: Input soquets for `bloq`; used to connect its left-dangling soquets.




Returns




<h3 id="finalize"><code>finalize</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1029-L1066">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>finalize(
    **final_soqs
) -> <a href="../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>
</code></pre>

Finish building a CompositeBloq and return the immutable CompositeBloq.

This method is similar to calling `add()` but instead of adding a new Bloq,
it configures the final "dangling" soquets that serve as the outputs for
the composite bloq as a whole.

If `self.add_registers_allowed` is set to `True`, additional register
names passed to this function will be added as RIGHT registers. Otherwise,
this method validates the provided `final_soqs` against our list of RIGHT
(and THRU) registers.

Args

`**final_soqs`
: Keyword arguments mapping the composite bloq's register names to
  final`Soquet`s, e.g. the output soquets from a prior, final operation.




<h3 id="allocate"><code>allocate</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1093-L1098">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>allocate(
    n: Union[int, sympy.Expr] = 1,
    dtype: Optional[<a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a>] = None
) -> <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
</code></pre>




<h3 id="free"><code>free</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1100-L1106">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>free(
    soq: <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> None
</code></pre>




<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1108-L1115">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    soq: <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> NDArray[Soquet]
</code></pre>

Add a Split bloq to split up a register.


<h3 id="join"><code>join</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1117-L1130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>join(
    soqs: NDArray[Soquet],
    dtype: Optional[<a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a>] = None
) -> <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
</code></pre>






