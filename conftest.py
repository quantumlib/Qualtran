#  Copyright 2023 Google LLC
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

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--enable-slow-tests", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    # Let pytest handle markexpr if present.  Make an exception for
    # `pytest --co -m skip` so we can check test skipping rules below.
    markexpr_words = frozenset(config.option.markexpr.split())
    if not markexpr_words.issubset(["not", "skip"]):
        return  # pragma: no cover
    if config.option.enable_slow_tests:
        return  # pragma: no cover
    # Note: see Cirq/conftest.py for handling of several custom marks to be skipped
    skip_slow = pytest.mark.skip(reason="need --enable-slow-tests option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
