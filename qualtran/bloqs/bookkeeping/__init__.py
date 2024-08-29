#  Copyright 2024 Google LLC
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
"""Bloqs for virtual operations and register reshaping."""

from qualtran.bloqs.bookkeeping.allocate import Allocate
from qualtran.bloqs.bookkeeping.arbitrary_clifford import ArbitraryClifford
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition
from qualtran.bloqs.bookkeeping.cast import Cast
from qualtran.bloqs.bookkeeping.free import Free
from qualtran.bloqs.bookkeeping.join import Join
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.bookkeeping.split import Split
