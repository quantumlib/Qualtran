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

from qualtran.bloqs.block_encoding import (
    _black_box_block_bloq,
    _black_box_prepare,
    _black_box_select,
)
from qualtran.testing import execute_notebook


def test_black_box_bloq(bloq_autotester):
    bloq_autotester(_black_box_block_bloq)


def test_black_box_prepare(bloq_autotester):
    bloq_autotester(_black_box_prepare)


def test_black_box_select(bloq_autotester):
    bloq_autotester(_black_box_select)


def test_notebook():
    execute_notebook('block_encoding')
