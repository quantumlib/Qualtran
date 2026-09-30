# QLT IR example `.qlt` files

This directory holds committed, **validated** QLT IR (`.qlt`) programs for a
hand-curated set of example bloqs from the `qualtran.bloqs` standard
library. They serve two purposes:

1. **Examples/references** for other consumers of the QLT IR text format.
2. **Golden files** for regression testing: `qualtran/qlt_ir/_roundtrip_test.py`
   asserts that each file is byte-for-byte reproducible by the current QLT IR
   compiler and that each round-trips (compile → print → parse → evaluate) with
   its structural properties preserved.

Every file here is guaranteed to round-trip cleanly under safe evaluation; the
generator refuses to write a file whose round trip does not preserve bloq keys,
signatures, object identity, and decomposition-graph topology.

## Regenerating

The curated example list lives in
[`qualtran/qlt_ir/_examples.py`](../_examples.py). To (re)generate these files after
a change to the compiler or the example list:

```shell
python dev_tools/generate-qlt-reference.py
```

To verify (without writing) that the committed files are up to date:

```shell
python dev_tools/generate-qlt-reference.py --check
```

Do not edit these files by hand.
