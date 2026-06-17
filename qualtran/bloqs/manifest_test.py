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


@pytest.mark.parametrize("fqn", BLOQ_CLASS_NAMES)
def test_import_manifest_class(fqn: str) -> None:
    """Each class in BLOQ_CLASS_NAMES must be importable with minimal deps."""
    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None
