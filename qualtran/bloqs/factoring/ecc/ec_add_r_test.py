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

from qualtran.bloqs.factoring.ecc.ec_add_r import _ec_add_r, _ec_add_r_small, _ec_window_add


def test_ec_add_r(bloq_autotester):
    bloq_autotester(_ec_add_r)


def test_ec_add_r_small(bloq_autotester):
    bloq_autotester(_ec_add_r_small)


def test_ec_window_add(bloq_autotester):
    bloq_autotester(_ec_window_add)
