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
from dev_tools.qualtran_dev_tools.parse_docstrings import get_markdown_docstring_lines


class ClassWithDocstrings:
    """This class has some nifty docstrings.

    Parameters:
        x: The variable x
        y: The variable y used by `my_function`.

    References:
        [Google](www.google.com). Brin et. al. 1999.
    """


def test_get_markdown_docstring_lines():
    lines = get_markdown_docstring_lines(ClassWithDocstrings)
    assert lines == [
        '## `ClassWithDocstrings`',
        'This class has some nifty docstrings.',
        '',
        '#### Parameters',
        ' - `x`: The variable x',
        ' - `y`: The variable y used by `my_function`. ',
        '',
        '#### References',
        ' - [Google](www.google.com). Brin et. al. 1999.',
        '',
    ]
