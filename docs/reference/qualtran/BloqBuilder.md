# BloqBuilder
`qualtran.BloqBuilder`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L833-L1310">
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L870-L917">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_register_from_dtype(
    reg: Union[str, <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>],
    dtype: Optional[<a href="../qualtran/QCDType.html"><code>qualtran.QCDType</code></a>] = None
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L925-L960">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_register(
    reg: Union[str, <a href="../qualtran/Register.html"><code>qualtran.Register</code></a>],
    bitsize: Optional['SymbolicInt'] = None
) -> Union[None, <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>]
</code></pre>

Add a new register to the composite bloq being built.

If this bloq builder was constructed with `add_registers_allowed=False`,
this operation is not allowed.

Args

`reg`
: Either the register or a register name. If this is a register name, then `bitsize`
  must also be provided and a default THRU register will be added.

`bitsize`
: If `reg` is a register name, this is the bitsize for the added register.
  Otherwise, this must not be provided.




Returns




<h3 id="from_signature"><code>from_signature</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L962-L986">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L988-L1006">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1035-L1055">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1057-L1074">View source</a>

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




<h3 id="add_and_partition"><code>add_and_partition</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1076-L1104">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_and_partition(
    bloq: <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    partitions: Sequence[Tuple[Register, Sequence[Union[str, 'Unused']]]],
    left_only: bool = False,
    **in_soqs
)
</code></pre>

Add a new bloq instance to the compute graph by partitioning input and output soquets to fit the signature of the bloq.


Args

`bloq`
: The bloq representing the operation to add.

`partitions`
: A sequence of pairs specifying each register that is exposed in the external
  signature of the `AutoPartition` and the corresponding register names from `bloq`
  that concatenate to form the externally exposed register. See `AutoPartition`.

`left_only`
: If False, the output soquets will also follow `partition`.
  Otherwise, the output soquets will follow `bloq.signature.rights()`.
  This flag must be set to True if `bloq` does not have the same LEFT and RIGHT
  registers, as is required for the bloq to be fully wrapped on the left and right.

`**in_soqs`
: Keyword arguments mapping the new bloq's register names to input
  `Soquet`s. This is likely the output soquets from a prior operation.




Returns




<h3 id="add"><code>add</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1106-L1138">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1164-L1196">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1198-L1235">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1262-L1269">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>allocate(
    n: <a href="../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a> = 1,
    dtype: Optional[<a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a>] = None,
    dirty: bool = False
) -> <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
</code></pre>




<h3 id="free"><code>free</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1271-L1281">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>free(
    soq: <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>,
    dirty: bool = False
) -> None
</code></pre>




<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1283-L1294">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    soq: <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> NDArray[Soquet]
</code></pre>

Add a Split bloq to split up a register.


<h3 id="join"><code>join</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L1296-L1310">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>join(
    soqs: <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>,
    dtype: Optional[<a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a>] = None
) -> <a href="../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
</code></pre>






