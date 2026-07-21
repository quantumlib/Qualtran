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
"""Tests for the Qualtran-L1 round-trip utilities, examples, and reference files.

This module has three parts:

 1. Unit tests for the individual round-trip *checkers* (both their positive and
    negative behaviour). These make sure the checkers actually catch problems
    rather than silently passing.
 2. Parametrized round-trip tests over the curated `L1_EXAMPLES`.
 3. Tests for the `save_bloq_qlt` API and the committed `.qlt` reference files.
"""

import attrs
import pytest

import qualtran as qlt
import qualtran.dtype as qdt
import qualtran.l1._roundtrip as _rt
from qualtran.bloqs.arithmetic import BitwiseNot
from qualtran.bloqs.basic_gates import CNOT, CSwap, XGate
from qualtran.bloqs.bookkeeping.qcast import QCast
from qualtran.l1._eval import _PlaceholderBloq
from qualtran.l1._examples import L1_EXAMPLES, reference_path
from qualtran.l1._parse_eval import load_module
from qualtran.l1._roundtrip import (
    assert_bloq_roundtrips,
    check_artifacts,
    check_bloq_keys,
    check_bloq_object_identity,
    check_bloq_roundtrip,
    check_no_placeholders,
    check_signatures,
    check_soquet_graph_isomorphism,
    compile_bloq_to_l1,
    RoundtripArtifacts,
    save_bloq_qlt,
    validate_bloq,
)

# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

_EXAMPLE_PARAMS = [
    pytest.param(ex, id=ex.name, marks=[pytest.mark.slow] if ex.slow else []) for ex in L1_EXAMPLES
]


def _evolve_loaded(artifacts: RoundtripArtifacts, **changes) -> RoundtripArtifacts:
    """Return a copy of `artifacts` with entries of the `loaded` dict overridden."""
    loaded = dict(artifacts.loaded)
    loaded.update(changes)
    return attrs.evolve(artifacts, loaded=loaded)


# ---------------------------------------------------------------------------
# compile_bloq_to_l1
# ---------------------------------------------------------------------------


def test_compile_bloq_to_l1_fields():
    artifacts = compile_bloq_to_l1(CNOT())
    assert artifacts.root_bloq_key == 'CNOT'
    assert 'CNOT' in artifacts.original
    assert 'CNOT' in artifacts.loaded
    assert artifacts.l1_code.startswith('# Qualtran-L1')
    # The loaded bloq should be the re-linked extern.
    assert artifacts.loaded['CNOT'] == CNOT()


# ---------------------------------------------------------------------------
# check_bloq_keys
# ---------------------------------------------------------------------------


def test_check_bloq_keys_clean():
    assert check_bloq_keys(compile_bloq_to_l1(CNOT())) == []


def test_check_bloq_keys_missing():
    artifacts = compile_bloq_to_l1(CSwap(bitsize=2))
    broken = attrs.evolve(artifacts, loaded={})
    problems = check_bloq_keys(broken)
    assert problems
    assert all(p.startswith('BLOQ_KEYS: Missing') for p in problems)


def test_check_bloq_keys_superfluous():
    artifacts = compile_bloq_to_l1(CNOT())
    broken = _evolve_loaded(artifacts, ExtraBloq=CNOT())
    problems = check_bloq_keys(broken)
    assert problems == ["BLOQ_KEYS: Superfluous 'ExtraBloq' after round trip"]


# ---------------------------------------------------------------------------
# check_signatures
# ---------------------------------------------------------------------------


def test_check_signatures_clean():
    assert check_signatures(compile_bloq_to_l1(CNOT())) == []


def test_check_signatures_mismatch():
    artifacts = compile_bloq_to_l1(CNOT())
    # XGate has a different signature than CNOT.
    broken = _evolve_loaded(artifacts, CNOT=XGate())
    problems = check_signatures(broken)
    assert len(problems) == 1
    assert problems[0].startswith('SIGNATURE: CNOT:')


# ---------------------------------------------------------------------------
# check_no_placeholders
# ---------------------------------------------------------------------------


def test_check_no_placeholders_clean():
    assert check_no_placeholders(compile_bloq_to_l1(CNOT())) == []


def test_check_no_placeholders_detects_placeholder():
    artifacts = compile_bloq_to_l1(CNOT())
    placeholder = _PlaceholderBloq(signature=CNOT().signature)
    broken = _evolve_loaded(artifacts, CNOT=placeholder)
    problems = check_no_placeholders(broken)
    assert problems == ['PLACEHOLDER: CNOT failed to re-link to a Bloq']


# ---------------------------------------------------------------------------
# check_bloq_object_identity
# ---------------------------------------------------------------------------


def test_check_bloq_object_identity_clean():
    assert check_bloq_object_identity(compile_bloq_to_l1(CSwap(bitsize=2))) == []


def test_check_bloq_object_identity_extern_mismatch():
    artifacts = compile_bloq_to_l1(CNOT())
    broken = _evolve_loaded(artifacts, CNOT=XGate())
    problems = check_bloq_object_identity(broken)
    assert len(problems) == 1
    assert problems[0].startswith('BLOQ_OBJECT: CNOT: extern')


def test_check_bloq_object_identity_composite_mismatch():
    artifacts = compile_bloq_to_l1(CSwap(bitsize=2))
    root = artifacts.root_bloq_key
    composite = artifacts.loaded[root]
    assert isinstance(composite, qlt.CompositeBloq)
    # Drop the `decomposed_from` link; the composite no longer records its origin.
    broken = _evolve_loaded(artifacts, **{root: attrs.evolve(composite, decomposed_from=None)})
    problems = check_bloq_object_identity(broken)
    assert any(p.startswith(f'BLOQ_OBJECT: {root}:') for p in problems)


def test_check_bloq_object_identity_skips_qcast():
    # bitwise_not decomposes via Split/Join, which become QCasts. Those must be
    # skipped by the object-identity check (they intentionally lose identity).
    artifacts = compile_bloq_to_l1(BitwiseNot(qdt.QUInt(4)))
    assert any(isinstance(b, QCast) for b in artifacts.loaded.values())
    assert check_bloq_object_identity(artifacts) == []


# ---------------------------------------------------------------------------
# check_soquet_graph_isomorphism
# ---------------------------------------------------------------------------


def test_check_soquet_graph_isomorphism_clean():
    assert check_soquet_graph_isomorphism(compile_bloq_to_l1(CSwap(bitsize=2))) == []


def test_check_soquet_graph_isomorphism_not_composite():
    artifacts = compile_bloq_to_l1(CSwap(bitsize=2))
    root = artifacts.root_bloq_key
    # Replace the decomposed composite with a non-composite bloq.
    broken = _evolve_loaded(artifacts, **{root: XGate()})
    problems = check_soquet_graph_isomorphism(broken)
    assert any('not a CompositeBloq' in p for p in problems)


# ---------------------------------------------------------------------------
# check_artifacts / check_bloq_roundtrip / assert_bloq_roundtrips
# ---------------------------------------------------------------------------


def test_check_artifacts_and_roundtrip_agree():
    artifacts = compile_bloq_to_l1(CSwap(bitsize=2))
    assert check_artifacts(artifacts) == []
    assert check_bloq_roundtrip(CSwap(bitsize=2)) == []


def test_assert_bloq_roundtrips_passes():
    assert_bloq_roundtrips(CNOT())


def test_assert_bloq_roundtrips_raises_on_problems(monkeypatch):
    # The negative path is exercised independently of any particular bloq (and of
    # the safe-import allow-list): inject a synthetic problem and confirm the
    # assertion fires and surfaces it.
    monkeypatch.setattr(_rt, 'check_bloq_roundtrip', lambda bloq, **kw: ['synthetic problem'])
    with pytest.raises(AssertionError, match='synthetic problem'):
        assert_bloq_roundtrips(CNOT())


# ---------------------------------------------------------------------------
# validate_bloq
# ---------------------------------------------------------------------------


def test_validate_bloq_atomic():
    # An atomic bloq has no decomposition; this should still pass.
    validate_bloq(XGate())


def test_validate_bloq_composite():
    validate_bloq(CSwap(bitsize=3))


@pytest.mark.parametrize('example', _EXAMPLE_PARAMS)
def test_example_validate_bloq(example):
    validate_bloq(example.make())


# ---------------------------------------------------------------------------
# End-to-end example round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('example', _EXAMPLE_PARAMS)
def test_example_roundtrips(example):
    bloq = example.make()
    validate_bloq(bloq)
    assert_bloq_roundtrips(bloq)


# ---------------------------------------------------------------------------
# save_bloq_qlt
# ---------------------------------------------------------------------------


def test_save_bloq_qlt_writes_and_reloads(tmp_path):
    path = tmp_path / 'nested' / 'cswap.qlt'
    code = save_bloq_qlt(CSwap(bitsize=2), path)

    # File written with the returned content...
    assert path.read_text() == code
    # ...and identical to a direct compile.
    assert code == compile_bloq_to_l1(CSwap(bitsize=2)).l1_code
    # ...and the saved code re-loads into a module keyed by the root bloq.
    module = load_module(path.read_text())
    assert 'CSwap' in module


def test_save_bloq_qlt_refuses_invalid(tmp_path, monkeypatch):
    # A bloq whose round trip reports problems must not be written (default
    # validate=True). Inject a synthetic problem to stay independent of the
    # safe-import allow-list.
    monkeypatch.setattr(_rt, 'check_artifacts', lambda artifacts: ['synthetic problem'])
    path = tmp_path / 'bad.qlt'
    with pytest.raises(AssertionError, match='Refusing to save'):
        save_bloq_qlt(CNOT(), path)
    assert not path.exists()


def test_save_bloq_qlt_no_validate(tmp_path, monkeypatch):
    # With validation off, the round-trip checks are skipped entirely and the file
    # is written even though `check_artifacts` would report a problem.
    monkeypatch.setattr(_rt, 'check_artifacts', lambda artifacts: ['synthetic problem'])
    path = tmp_path / 'unvalidated.qlt'
    code = save_bloq_qlt(CNOT(), path, validate=False)
    assert path.read_text() == code


# ---------------------------------------------------------------------------
# Committed reference files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('example', _EXAMPLE_PARAMS)
def test_reference_file_up_to_date(example):
    path = reference_path(example.name)
    assert path.is_file(), (
        f'Missing reference file {path}. Regenerate with '
        f'`python dev_tools/generate-l1-reference.py`.'
    )

    bloq = example.make()
    validate_bloq(bloq)
    artifacts = compile_bloq_to_l1(bloq)
    assert check_artifacts(artifacts) == []
    assert path.read_text() == artifacts.l1_code, (
        f'Reference file {path} is out of date. Regenerate with '
        f'`python dev_tools/generate-l1-reference.py`.'
    )


@pytest.mark.parametrize('example', _EXAMPLE_PARAMS)
def test_reference_file_loadable(example):
    path = reference_path(example.name)
    module = load_module(path.read_text())
    assert example.make()  # constructible
    assert module  # non-empty
