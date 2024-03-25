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

from typing import Optional

from attrs import frozen


@frozen
class Reference:
    """A reference to a source material.

    Attributes:
        url: A link to a paper/article (e.g. https://link-to-paper).
        page: A page number.
        comment: An explanation or a pointer to a specific part of the source (e.g. `table 2`, `Fig 1e`, `third paragraph`).
    """

    url: Optional[str] = None
    page: Optional[int] = None
    comment: Optional[str] = None

    def __str__(self) -> str:
        not_none = []
        if self.url is not None:
            not_none.append(f"url='{self.url}'")
        if self.page is not None:
            not_none.append(f'page={self.page}')
        if self.comment is not None:
            not_none.append(f"comment='{self.comment}'")
        return 'Reference(' + ', '.join(not_none) + ')'

    def __repr__(self) -> str:
        return self.__str__()
