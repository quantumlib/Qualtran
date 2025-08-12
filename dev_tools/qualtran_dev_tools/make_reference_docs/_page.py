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
from enum import Enum
from pathlib import Path
from typing import Union

import attrs
import griffe


@attrs.mutable
class MajorClassPage:
    """A doc page for a 'major' class.

    A major class is documented on its own page, whereas
    a minor class is briefly documented on its module's page.
    """

    obj: griffe.Class = None
    section: str = None
    pref_path: str = None

    def out_path(self, out_dir: Path) -> Path:
        segments = self.pref_path.split('.')
        return out_dir / '/'.join(segments[:-1]) / f'{segments[-1]}.md'


class MemberType(Enum):
    MODULE = 'module'
    MAJOR = 'major'
    MINOR = 'minor'


@attrs.mutable
class ModulePageMember:
    obj: griffe.Object
    pref_dotpath: str
    mytype: MemberType


@attrs.mutable
class ModulePage:
    """A doc page for a module."""

    obj: griffe.Module = None
    section: str = None
    pref_path: str = None
    members: list[ModulePageMember] = attrs.field(factory=list)

    def out_path(self, out_dir: Path) -> Path:
        segments = self.pref_path.split('.')
        return out_dir / '/'.join(segments[:-1]) / f'{segments[-1]}.md'


Page = Union[ModulePage, MajorClassPage]
