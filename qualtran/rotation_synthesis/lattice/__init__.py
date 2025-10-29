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

"""A package that provides methods for integer point enumeration."""

from qualtran.rotation_synthesis.lattice.geometry import Ellipse, Range, Rectangle
from qualtran.rotation_synthesis.lattice.grid_operators import GridOperator
from qualtran.rotation_synthesis.lattice.ipe import (
    enumerate_1d,
    enumerate_upright,
    get_overall_action,
    get_points_from_state,
)
from qualtran.rotation_synthesis.lattice.state import GridOperatorAction, SelingerState
