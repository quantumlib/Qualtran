Quantum Graph
-------------

## Data structure

We represent our quantum program as a directed graph of quantum variables passing through
operations called `Bloq`s. The container representing a program or subroutine is 
`CompositeBloq`. Since it is an immutable data structure, we use `BloqBuilder`
to help us build composite bloqs. We use the abbreviated variable name `cbloq` for
generic composite bloqs.

>>> bb = BloqBuilder()
>>> q = bb.allocate(1)       # allocate a quantum variable.
>>> q, = bb.add(H, qubit=q)  # wire it into a new `H` op, get new quantum variable back.
>>> bb.free(q)               # discard our quantum variable, get nothing back.
>>> cbloq = bb.finalize()    # finish building, turn into a `CompositeBloq`

## Components

`Bloq`s are analogous to functions. Any class attributes on a `Bloq` are analogous to
C++ template parameters. `Bloq`s follow value-equality with respect to these parameters.

>>> MultiCNOT(n=100) == MultiCNOT(n=100)

`BloqInstance`s are unique instantiations of a `Bloq` contained in a `CompositeBloq`, because
we could have many instances of the same `Bloq` and it matters (for data flow) which one
is which.

>>> binst1, (q,) = bb.add_2(H, q=q)
>>> binst2, (q,) = bb.add_2(H, q=q)
>>> binst1 != binst2  # one `H` comes before the other.

`FancyRegisters` are analogous to function (and return type) signatures and are a 
property of a `Bloq`. 

`Soquet`s are the "quantum variables" that follow linear typing rules (each must be used exactly
once) and are instantiations of registers. There is generally a 1-to-many relationship between
`FancyRegister` and `Soquet` because we support multidimensional quantum arrays.

`SoquetT` is a union type between `Soquet` and ndarray of `Soquet`. It means all the soquets
for a given register. When you see variable names like `soqdict`, `in_soqs`, or `final_soqs`
with type `Dict[str, SoquetT]`: this is a 1-to-1 mapping between register (by name) and 
all the `Soquet`s for that potentially-multidimensional register. When you see names 
like `idxed_soq`, it means an individual `Soquet` has been plucked
from a potentially-multidimensional array of them.

`Soquet`s are the main quantum variables a user-developer uses to wire up bloqs. The user
should assign meaningful names to the soquets or array of soquets and treat them as opaque
types.

## Gates

Bloqs support multi-dimensional, arbitrary-bitwidth, potentially asymmetric registers. As such,
bloqs can represent high-level computations like "ModularExponentiation" or 
"Quantum Phase Estimation".

Low-level `Bloq`s where all `FancyRegister`s have `bitsize=1`, `wireshape=(n,)`,
and `side=Side.THRU` we generally name "gates". Bloqs that satisfy these conditions
are similar to gates found in other quantum computing libraries.