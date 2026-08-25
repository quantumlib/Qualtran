#  Copyright 2025 Google LLC
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
import importlib

import pytest

from qualtran.bloqs.manifest import BLOQ_CLASS_NAMES

OPTIONAL_DEPS: dict[str, list[str]] = {
    "galois": ["qualtran.bloqs.gf_arithmetic.", "qualtran.bloqs.gf_poly_arithmetic."]
}


def _is_installed(pkg: str) -> bool:
    """Return True if *pkg* can be imported."""
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


# Build the set of prefixes to skip based on what's missing.
_SKIP_PREFIXES: list[str] = []
for _pkg, _prefixes in OPTIONAL_DEPS.items():
    if not _is_installed(_pkg):
        _SKIP_PREFIXES.extend(_prefixes)
_SKIP_PREFIXES_TUPLE = tuple(_SKIP_PREFIXES)  # for str.startswith()


def _should_skip(fqn: str) -> str | None:
    """If *fqn* requires a missing optional dep, return a skip reason."""
    for pkg, prefixes in OPTIONAL_DEPS.items():
        if any(fqn.startswith(p) for p in prefixes) and not _is_installed(pkg):
            return f"optional dependency {pkg!r} is not installed"
    return None


@pytest.mark.parametrize("fqn", BLOQ_CLASS_NAMES)
def test_import_manifest_class(fqn: str) -> None:
    """Each class in BLOQ_CLASS_NAMES must be importable with minimal deps."""
    reason = _should_skip(fqn)
    if reason:
        pytest.skip(reason)

    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None
