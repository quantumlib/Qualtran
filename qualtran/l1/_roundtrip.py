#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Utilities for round-trip testing the Qualtran-L1 intermediate representation.

The L1 pipeline has four stages:

```
Bloq  --compile-->  L1Module AST  --print-->  .qlt text
.qlt text  --parse-->  L1Module AST  --eval-->  {bloq_key: Bloq}
```

A perfectly lossless round trip from a Python `Bloq` back to an identical Python
`Bloq` is *not* a design goal of the `.qlt` representation. Some information is
deliberately discarded or transformed:

 - A leaf bloq is emitted as an `extern qdef` and re-linked by importing the
   Python class; the resulting object is equal to the original but the round trip
   does not exercise its decomposition.
 - A decomposed bloq is emitted as a `qdef` body and evaluated back into a
   `qualtran.CompositeBloq` (not the original bloq type). The original bloq is
   recorded on the `.decomposed_from` attribute.
 - Bookkeeping bloqs become `qcast`s which intentionally lose object identity.

Instead of exact equality, these utilities check that a well-defined set of
*properties* is preserved by the round trip. See `check_bloq_roundtrip`.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, TYPE_CHECKING, Union

import attrs
import networkx as nx

import qualtran as qlt

from . import nodes as qualtran_l1_nodes
from ._ast_to_code import L1ASTPrinter
from ._parse_eval import load_module
from ._to_l1 import L1ModuleBuilder, QDefWithContext

if TYPE_CHECKING:
    from .nodes import L1Nodes


@attrs.frozen
class RoundtripArtifacts:
    """The intermediate products of round-tripping a bloq through Qualtran-L1.

    Attributes:
        root_bloq_key: The bloq key of the root bloq that was compiled.
        original: The compiled qdefs (with Python-object context), keyed by bloq key.
            This is the "ground truth" against which the loaded bloqs are compared.
        l1_code: The textual `.qlt` representation.
        loaded: The bloqs recovered by parsing and evaluating `l1_code`, keyed by
            bloq key.
    """

    root_bloq_key: str
    original: Dict[str, QDefWithContext]
    l1_code: str
    loaded: Dict[str, 'qlt.Bloq']


def compile_bloq_to_l1(
    bloq: 'qlt.Bloq', *, extern_only_from: bool = False, nodes: 'L1Nodes' = qualtran_l1_nodes
) -> RoundtripArtifacts:
    """Compile a bloq to `.qlt` text and evaluate it back into bloqs.

    This performs the full round trip (compile → print → parse → eval) but does
    *not* run any property checks. Use `check_bloq_roundtrip` to additionally
    validate the result.

    Args:
        bloq: The bloq to compile.
        extern_only_from: Passed through to `L1ModuleBuilder.add_bloqs`.
        nodes: AST node factory module.

    Returns:
        A `RoundtripArtifacts` capturing every intermediate product.
    """
    # Use L1ModuleBuilder directly so we retain a record (`original`) of the
    # true subbloqs we serialized, alongside their Python objects.
    l1mb = L1ModuleBuilder(nodes=nodes)
    root_bloq_key = l1mb.add_bloqs(root=bloq, extern_only_from=extern_only_from)
    original = {qdef.bloq_key: qdef for qdef in l1mb.qdefs}

    l1_code = L1ASTPrinter().visit(l1mb.finalize())

    loaded = load_module(l1_code)

    return RoundtripArtifacts(
        root_bloq_key=root_bloq_key, original=original, l1_code=l1_code, loaded=loaded
    )


def check_bloq_keys(artifacts: RoundtripArtifacts) -> List[str]:
    """Check that the set of bloq keys is preserved by the round trip."""
    problems = []
    ref = set(artifacts.original.keys())
    tst = set(artifacts.loaded.keys())
    for missing in sorted(ref - tst):
        problems.append(f'BLOQ_KEYS: Missing {missing!r} after round trip')
    for extra in sorted(tst - ref):
        problems.append(f'BLOQ_KEYS: Superfluous {extra!r} after round trip')
    return problems


def check_signatures(artifacts: RoundtripArtifacts) -> List[str]:
    """Check that every bloq's signature is preserved by the round trip."""
    problems = []
    for bloq_key in artifacts.original.keys():
        if bloq_key not in artifacts.loaded:
            continue  # Missing keys are reported by `check_bloq_keys`.
        ref = artifacts.original[bloq_key]
        tst = artifacts.loaded[bloq_key]
        if ref.bloq.signature != tst.signature:
            problems.append(f"SIGNATURE: {bloq_key}: {ref.bloq.signature} != {tst.signature}")
    return problems


def check_no_placeholders(artifacts: RoundtripArtifacts) -> List[str]:
    """Check that every extern bloq was successfully re-linked.

    When an `extern qdef` cannot be resolved back to a `qualtran.Bloq` (e.g. it is
    not on the safe-import manifest), evaluation falls back to a
    `_PlaceholderBloq`. For a well-formed example this should never
    happen, so we flag it.
    """
    from ._eval import _PlaceholderBloq

    problems = []
    for bloq_key, tst in artifacts.loaded.items():
        if isinstance(tst, _PlaceholderBloq):
            problems.append(f"PLACEHOLDER: {bloq_key} failed to re-link to a Bloq")
    return problems


def check_bloq_object_identity(artifacts: RoundtripArtifacts) -> List[str]:
    """Check that re-linked/decomposed bloqs correspond to the originals.

    For each bloq key:
     - `QCast` bloqs are skipped: they intentionally lose object identity and are
       validated by `check_signatures` instead.
     - A loaded `CompositeBloq` must record the original bloq on `.decomposed_from`.
     - Any other loaded bloq (an externed leaf) must equal the original.
    """
    from qualtran.bloqs.bookkeeping.qcast import QCast

    problems = []
    for bloq_key in artifacts.original.keys():
        if bloq_key not in artifacts.loaded:
            continue
        ref = artifacts.original[bloq_key]
        tst = artifacts.loaded[bloq_key]

        if isinstance(tst, QCast):
            continue

        if isinstance(tst, qlt.CompositeBloq):
            if ref.bloq != tst.decomposed_from:
                problems.append(
                    f"BLOQ_OBJECT: {bloq_key}: decomposed_from "
                    f"{tst.decomposed_from!r} != {ref.bloq!r}"
                )
        elif ref.bloq != tst:
            problems.append(f'BLOQ_OBJECT: {bloq_key}: extern {tst!r} != {ref.bloq!r}')

    return problems


def _soquet_graph(cbloq: 'qlt.CompositeBloq') -> nx.DiGraph:
    """Build a soquet-level directed graph from a composite bloq."""
    graph = nx.DiGraph()
    graph.add_nodes_from((soq, {'soq': soq}) for soq in cbloq.all_soquets)
    graph.add_edges_from((cxn.left, cxn.right) for cxn in cbloq.connections)
    return graph


def check_soquet_graph_isomorphism(artifacts: RoundtripArtifacts) -> List[str]:
    """Check that decomposition graphs are isomorphic across the round trip.

    For each original bloq that decomposes, we compare the soquet-connectivity
    graph of the original decomposition with that of the loaded bloq. Nodes are
    matched by register, so this verifies the wiring topology is preserved.
    """
    problems = []
    for bloq_key in artifacts.original.keys():
        if bloq_key not in artifacts.loaded:
            continue
        ref = artifacts.original[bloq_key]
        tst = artifacts.loaded[bloq_key]

        try:
            ref_cbloq = ref.bloq.decompose_bloq()
        except (qlt.DecomposeTypeError, qlt.DecomposeNotImplementedError):
            continue

        if not isinstance(tst, qlt.CompositeBloq):
            problems.append(
                f'SOQUET_GRAPH: {bloq_key}: original decomposes but loaded '
                f'{type(tst).__name__} is not a CompositeBloq'
            )
            continue

        ref_sg = _soquet_graph(ref_cbloq)
        tst_sg = _soquet_graph(tst)

        def node_match(n1, n2):
            s1, s2 = n1['soq'], n2['soq']
            return s1.reg == s2.reg and s1.idx == s2.idx

        if not nx.is_isomorphic(tst_sg, ref_sg, node_match=node_match):
            problems.append(f'SOQUET_GRAPH: {bloq_key}: decomposition graphs not isomorphic')
    return problems


#: The default set of property checks run by `check_bloq_roundtrip`.
DEFAULT_CHECKS = (
    check_bloq_keys,
    check_signatures,
    check_no_placeholders,
    check_bloq_object_identity,
    check_soquet_graph_isomorphism,
)


def check_bloq_roundtrip(bloq: 'qlt.Bloq', *, extern_only_from: bool = False) -> List[str]:
    """Round-trip a bloq through Qualtran-L1 and return a list of problems.

    This compiles `bloq` to `.qlt`, parses and evaluates it back, and then runs
    every check in `DEFAULT_CHECKS`. An empty return value means the round trip
    preserved all checked properties.

    Args:
        bloq: The bloq to round-trip.
        extern_only_from: Passed through to `compile_bloq_to_l1`.

    Returns:
        A (possibly empty) list of human-readable problem descriptions.
    """
    artifacts = compile_bloq_to_l1(bloq, extern_only_from=extern_only_from)
    return check_artifacts(artifacts)


def check_artifacts(artifacts: RoundtripArtifacts) -> List[str]:
    """Run every check in `DEFAULT_CHECKS` against already-computed artifacts."""
    problems: List[str] = []
    for check in DEFAULT_CHECKS:
        problems.extend(check(artifacts))
    return problems


def assert_bloq_roundtrips(bloq: 'qlt.Bloq', *, extern_only_from: bool = False) -> None:
    """Assert that a bloq round-trips through Qualtran-L1 with no problems.

    Raises:
        AssertionError: If any checked property is not preserved, with all
            problems included in the message.
    """
    problems = check_bloq_roundtrip(bloq, extern_only_from=extern_only_from)
    if problems:
        joined = '\n  - '.join(problems)
        raise AssertionError(f"Round-trip of {bloq!r} failed:\n  - {joined}")


def save_bloq_qlt(
    bloq: 'qlt.Bloq',
    path: Union[str, 'os.PathLike[str]'],
    *,
    extern_only_from: bool = False,
    validate: bool = True,
) -> str:
    """Compile a bloq to `.qlt` and write it to `path` as a reference example.

    By default, the round trip is validated (via `DEFAULT_CHECKS`) *before* the
    file is written, so only known-good, round-trippable `.qlt` code is ever
    persisted. Parent directories are created as needed.

    Args:
        bloq: The bloq to serialize.
        path: Destination path for the `.qlt` file.
        extern_only_from: Passed through to `compile_bloq_to_l1`.
        validate: If `True` (default), assert the round trip preserves all checked
            properties before writing. If `False`, write unconditionally.

    Returns:
        The `.qlt` text that was written.

    Raises:
        AssertionError: If `validate` is `True` and the round trip fails.
    """
    artifacts = compile_bloq_to_l1(bloq, extern_only_from=extern_only_from)
    if validate:
        problems = check_artifacts(artifacts)
        if problems:
            joined = '\n  - '.join(problems)
            raise AssertionError(f"Refusing to save {bloq!r}: round trip failed:\n  - {joined}")

    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(artifacts.l1_code)
    return artifacts.l1_code


def validate_bloq(bloq: 'qlt.Bloq') -> None:
    """Validate a Python bloq before using it as a round-trip subject.

    Rejects malformed bloqs early, before they reach the round-trip machinery.
    Specifically it:

     - Accesses the signature (ensuring it is constructible).
     - If the bloq decomposes, runs `assert_valid_bloq_decomposition`.

    Raises:
        AssertionError: If the bloq or its decomposition is invalid.
    """
    import qualtran.testing as qlt_testing

    # Accessing the signature must succeed and yield a Signature.
    assert isinstance(bloq.signature, qlt.Signature), f"{bloq!r} has a bad signature"

    try:
        qlt_testing.assert_valid_bloq_decomposition(bloq)
    except (qlt.DecomposeTypeError, qlt.DecomposeNotImplementedError):
        # An atomic (non-decomposing) bloq is a valid round-trip subject; it will
        # simply be emitted as an `extern qdef`.
        pass
