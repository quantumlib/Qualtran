# TypeIs


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Special typing form used to annotate the return type of a user-defined type narrower function.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.symbolics.types.TypeIs(
    *args, **kwds
)
</code></pre>



<!-- Placeholder for "Used in" -->
  ``TypeIs`` only accepts a single type argument.
At runtime, functions marked this way should return a boolean.

``TypeIs`` aims to benefit *type narrowing* -- a technique used by static
type checkers to determine a more precise type of an expression within a
program's code flow.  Usually type narrowing is done by analyzing
conditional code flow and applying the narrowing to a block of code.  The
conditional expression here is sometimes referred to as a "type guard".

Sometimes it would be convenient to use a user-defined boolean function
as a type guard.  Such a function should use ``TypeIs[...]`` as its
return type to alert static type checkers to this intention.

Using  ``-> TypeIs`` tells the static type checker that for a given
function:

1. The return value is a boolean.
2. If the return value is ``True``, the type of its argument
is the intersection of the type inside ``TypeGuard`` and the argument's
previously known type.

For example::

    def is_awaitable(val: object) -> TypeIs[Awaitable[Any]]:
        return hasattr(val, '__await__')

    def f(val: Union[int, Awaitable[int]]) -> int:
        if is_awaitable(val):
            assert_type(val, Awaitable[int])
        else:
            assert_type(val, int)

``TypeIs`` also works with type variables.  For more information, see
PEP 742 (Narrowing types with TypeIs).