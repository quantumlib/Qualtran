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
import time

from .execute_with_timeout import ExecuteWithTimeout


def a_long_function(n_seconds: int, cxn) -> None:
    time.sleep(n_seconds)
    cxn.send("Done")


def test_execute_with_timeout():
    exe = ExecuteWithTimeout(timeout=1, max_workers=1)

    for ns in [0.1, 100]:
        exe.submit(a_long_function, {'n_seconds': ns})

    results = []
    while exe.work_to_be_done:
        kwargs, result = exe.next_result()
        if result is None:
            results.append('Timeout')
        else:
            results.append(result)

    assert set(results) == {'Done', 'Timeout'}
