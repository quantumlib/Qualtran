# BloqCheckException
`qualtran.testing.BloqCheckException`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L246-L293">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An exception raised by the `assert_bloq_example_xxx` functions in this module.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.BloqCheckException(
    check_result: <a href="../../qualtran/testing/BloqCheckResult.html"><code>qualtran.testing.BloqCheckResult</code></a>,
    msg: str
)
</code></pre>



<!-- Placeholder for "Used in" -->

These exceptions correspond to known failures due to assertion errors, non-applicable checks,
or unverified protocols.

Consider using the factory class methods `BloqCheckException.{fail, missing, na, unverified}`
for convenience.

<h2 class="add-link">Args</h2>

`check_result`<a id="check_result"></a>
: The BloqCheckResult.

`msg`<a id="msg"></a>
: A message providing details for the exception.






<h2 class="add-link">Attributes</h2>

`check_result`<a id="check_result"></a>
: The BloqCheckResult.

`msg`<a id="msg"></a>
: A message providing details for the exception.




## Methods

<h3 id="fail"><code>fail</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L275-L278">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>fail(
    msg: str
) -> 'BloqCheckException'
</code></pre>

Create an exception with a FAIL check result.


<h3 id="missing"><code>missing</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L280-L283">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>missing(
    msg: str
) -> 'BloqCheckException'
</code></pre>

Create an exception with a MISSING check result.


<h3 id="na"><code>na</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L285-L288">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>na(
    msg: str
) -> 'BloqCheckException'
</code></pre>

Create an exception with a NA check result.


<h3 id="unverified"><code>unverified</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L290-L293">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>unverified(
    msg: str
) -> 'BloqCheckException'
</code></pre>

Create an exception with an UNVERIFIED check result.




