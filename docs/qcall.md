# Supporting `qcall`

In Qualtran, quantum programs are traditionally constructed by adding subroutines (represented as
`Bloq` classes) using the imperative `bb.add(...)` method of `BloqBuilder`. While this is the
most literal mechanism for populating a composition's dataflow graph, it can be verbose.

To reduce some of the boilerplate, bloqs can choose to support the **`qcall` pattern**. By defining
a `qcall` method on a bloq class, authors enable users to compose bloqs in a manner more similar
to traditional function calls operating directly on quantum variables (`QVar` objects).

For example, instead of writing:
```python
q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)
```

Users can simply write:
```python
q0, q1 = CNOT.qcall(q0, q1)
```

Behind the scenes, operators such as inversion `~q` are also mapped to `qcall` methods (e.g. `~q`
maps to `XGate.qcall(q)` if `q` is a `QBit`), enabling even more concise usage of basic
subroutines.

## How `qcall` Works Under the Hood

During composite bloq building (such as within `build_composite_bloq` or standalone `BloqBuilder`
contexts), quantum variables are represented as `QVar` objects.

These provide a mutable handle that represents a quantum register of a specific `QCDType` (e.g.
`QBit`, `QUInt`). Crucially, every active `QVar` holds a reference to the active `BloqBuilder`
instance in its `.bb` property.

When a user executes `CNOT.qcall(q0, q1)`:
1. The `qcall` method retrieves the active `BloqBuilder` by accessing `q0.bb` or `q1.bb`.
2. It instantiates the `CNOT` bloq: `bloq = CNOT()`.
3. It adds the bloq instance to the builder: `bb.add(bloq, ctrl=q0, target=q1)`.
4. It returns the resulting `QVar`s returned by `bb.add(...)`.


## How to Add a `qcall` Method to Your Bloq Class

To add `qcall` support to your bloq class:

1. **Define a `@classmethod` named `qcall`** on your class. The method's signature should accept the
   input quantum variables as positional or keyword arguments.
2. **Extract the `BloqBuilder`** dynamically from one of the input `QVar` arguments using the `.bb`
   property (e.g., `bb = qvar.bb`).
3. **Construct the Bloq instance** (either parameter-less or parametrized based on the inputs) and
   add it using `bb.add(...)`.
4. **Return the output `QVar`s** returned by `bb.add(...)`.

If your bloq has attributes (such as datatypes, bitsizes, or constants) that are determined from the
quantum variable arguments at runtime, you should extract these parameters from the arguments'
`.dtype` or shapes:

```python
from typing import TYPE_CHECKING
from attrs import frozen
from qualtran import Bloq, Signature, QUInt

if TYPE_CHECKING:
    from qualtran import QVar

@frozen
class Add(Bloq):
    a_dtype: QUInt
    b_dtype: QUInt

    @property
    def signature(self) -> Signature:
        return Signature.build(a=self.a_dtype, b=self.b_dtype)

    @classmethod
    def qcall(cls, a: 'QVar', b: 'QVar') -> 'QVar':
        # 1. Infer the parameterized bloq fields from the QVar arguments
        bloq = cls(a_dtype=a.dtype, b_dtype=b.dtype)
        # 2. Retrieve the builder
        bb = a.bb
        # 3. Construct and add the bloq
        return bb.add(bloq, a=a, b=b)
```


### `BloqBuilder` helper methods

For the most common basic gates and bookkeeping bloqs (e.g., `CNOT`, `And`, `X`, `Z`, `H`), we also
expose a helper method on the `BloqBuilder` class. This allows users of the
builder to call them as `bb.CNOT(...)` or `bb.X(...)`.

