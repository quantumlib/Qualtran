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
import pytest

from .bloq_finder import get_bloq_classes
from .parse_docstrings import get_markdown_docstring_lines, get_references, UnparsedReference


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


@pytest.mark.slow
def test_parse_all_references():
    bloq_classes = get_bloq_classes()
    references = {bloq_cls: get_references(bloq_cls) for bloq_cls in bloq_classes}
    for bloq_cls, refs in references.items():
        for ref in refs:
            if isinstance(ref, UnparsedReference):
                raise AssertionError(
                    f"{bloq_cls.__module__}.{bloq_cls.__qualname__} has an incorrectly "
                    f"formatted reference:\n{ref.text}"
                )
