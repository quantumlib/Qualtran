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
from pathlib import Path
from typing import Dict, List, Tuple

import attrs

from ._linking_writer import LinkingWriter, Writable
from ._page import Page


@attrs.mutable
class RenderContext:
    """Context required to make links."""

    # todo .. document attributes
    refdoc_relpath: Path
    sections: List[str]
    aliases_d: Dict[str, str]
    link_d: Dict[str, Tuple[Page, str | None]]
    linkable_to_prefpath: Dict[str, str]

    def get_linking_writer(self, f: Writable) -> LinkingWriter:
        return LinkingWriter(
            f,
            link_aliases=self.linkable_to_prefpath,
            link_d=self.link_d,
            refdoc_relpath=self.refdoc_relpath,
        )
