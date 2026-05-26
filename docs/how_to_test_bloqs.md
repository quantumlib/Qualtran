# How to Test Bloqs in Qualtran

Qualtran uses a multi-layered testing strategy to ensure that Bloqs are correct, have valid structures, and report accurate resource costs. This guide outlines the various tools and patterns available for testing Bloqs.

## 1. Overview

Testing a Bloq typically involves:
- **Automated testing** for standard properties (creation, decomposition, counts, serialization, typing).
- **Classical simulation** for verifying logical correctness on basis states.
- **Quantum simulation via Tensor contraction** for obtaining the full unitary matrix or state vector.
- **Manual assertions** for fine-grained control over structure and costs.

---

## 2. Automated Testing with `bloq_autotester`

The recommended way to test the standard properties of a Bloq is using the `bloq_autotester` pytest fixture. This fixture automatically runs a suite of checks on your Bloq instances.

### Step 1: Define a Bloq Example

To use the autotester, you must first define one or more examples of your Bloq in the source file (not the test file) using the `@bloq_example` decorator. This exposes the bloq for both testing and documentation.

```python
from qualtran import Bloq, bloq_example
from attrs import frozen

@frozen
class MyBloq(Bloq):
    # ... implementation ...
    pass

@bloq_example
def _my_bloq() -> MyBloq:
    return MyBloq()
```

### Step 2: Run Tests using the Fixture

In your corresponding test file (e.g., `my_bloq_test.py`), you use the `bloq_autotester` fixture and pass the example function.

```python
from qualtran.bloqs.my_bloq import _my_bloq

# Note: bloq_autotester is a pytest fixture and does not need to be imported.
def test_my_bloq(bloq_autotester):
    bloq_autotester(_my_bloq)
```

Because `bloq_autotester` is a parameterized fixture, it will automatically run the following 5 checks as separate test cases:
1. **`make`**: Verifies that the example instance can be created and is of the expected type.
2. **`decompose`**: Verifies that the graph structure after decomposition is valid (e.g., signature matches parent, no dangling soquets).
3. **`counts`**: Verifies that resource counts obtained via `build_call_graph` match counts derived from the decomposition.
4. **`serialize`**: Verifies that the Bloq can be serialized to protocol buffers and deserialized back without losing information.
5. **`qtyping`**: Verifies that quantum data types are consistent across all connections in the decomposition.

---

## 3. Classical Simulation

For Bloqs that represent reversible classical operations, you can verify correctness by simulating their action on computational basis states.

### Direct Execution with `call_classically`

You can use `bloq.call_classically()` to execute the Bloq on basis states. This method takes input values for the registers and returns a tuple of output values (ordered by the bloq's right registers).

```python
from qualtran.bloqs.basic_gates import TwoBitSwap

def test_two_bit_swap_call_classically():
    swap = TwoBitSwap()
    x, y = swap.call_classically(x=0, y=1)
    assert x == 1
    assert y == 0
```

### Consistency with Decomposition

To ensure that a Bloq's decomposition matches its high-level classical action, use `assert_consistent_classical_action`. This helper iterates over all combinations of provided input ranges.

```python
from qualtran.testing import assert_consistent_classical_action
from qualtran.bloqs.basic_gates import TwoBitSwap

def test_two_bit_swap_classical():
    assert_consistent_classical_action(TwoBitSwap(), x=[0, 1], y=[0, 1])
```

For Bloqs that introduce phases (like controlled-phase operations), use **`assert_consistent_phased_classical_action`** to track phase information as well.

---

## 4. Quantum Simulation via Tensor Contraction

For small Bloqs, `tensor_contract()` is the primary and recommended method to obtain the full unitary matrix or state vector by contracting the underlying tensor network. This is more general than Cirq-based simulation and is the standard way to verify quantum behavior.

### Verifying Unitaries

You can compare the contracted tensor with a hardcoded numpy array or a corresponding Cirq gate unitary.

```python
import numpy as np
import cirq
from qualtran.bloqs.basic_gates import Hadamard

def test_unitary_vs_cirq():
    h = Hadamard()
    unitary = h.tensor_contract()
    cirq_unitary = cirq.unitary(cirq.H)
    np.testing.assert_allclose(unitary, cirq_unitary)
```

---

## 5. Manual Structural and Cost Assertions

While `bloq_autotester` covers these, you can also perform these assertions manually for fine-grained control:

- **`assert_valid_bloq_decomposition(bloq)`**: Checks the validity of a Bloq's decomposition graph.
- **`assert_equivalent_bloq_counts(bloq)`**: Verifies that call graph annotations match the decomposition.
- **`bloq.t_complexity()`**: You can check the T-complexity directly against expected analytical formulas.

---

## 6. Advanced Topics

### Symbolics
Qualtran supports symbolic parameters (using Sympy). When testing bloqs with symbolic attributes, use `sympy.Symbol` in your `@bloq_example` to ensure they work correctly.

### Doctests
Qualtran relies on doctests in docstrings for quick verification. Ensure your docstrings contain valid, executable Python code demonstrating usage.

### Drawing
Use `assert_wire_symbols_match_expected` to verify the visual representation of your Bloq in diagrams.

### Notebooks
To ensure that documentation notebooks remain executable and correct, use `execute_notebook` (from `qualtran.testing`) in your tests.
