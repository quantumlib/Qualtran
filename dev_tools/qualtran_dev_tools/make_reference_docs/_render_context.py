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
from typing import Dict, List, Optional, Tuple

import attrs

from ._linking_writer import LinkingWriter, Writable
from ._page import Page


@attrs.mutable
class RenderContext:
    """Context required to make links.


    # Walk the tree into our weird internal data structurs.
    pages, aliases_d, link_d = get_pages(root_mod=root_mod)

    # Merge user-provided and auto-found section titles.
    sections = list(top_sections) + sorted(
        set(mp.section_not_none for mp in pages) - set(top_sections) - set(fake_sections)
    )

    # Set up dictionary of "linkable" items.
    linkable_to_prefpath = dict(aliases_d)
    linkable_to_prefpath |= addtl_linkable

    # Return pages and a `RenderContext`.
    render_context = RenderContext(
        refdoc_relpath=refdoc_relpath,
        sections=sections,
        aliases_d=aliases_d,
        link_d=link_d,
        linkable_to_prefpath=linkable_to_prefpath,
    )
    return pages, render_context

    """

    refdoc_relpath: Path
    """Relative subdirectory of where the reference docs go (opposed to the other docs)"""

    sections: List[str]
    """Ordered list of section names (to organize pages)"""

    aliases_d: Dict[str, str]
    """Returned by `get_pages`, see `_PackageWalker.aliases_d`"""

    link_d: Dict[str, Tuple[Page, Optional[str]]]
    """Returned by `get_pages`, see `_PackageWalker.link_d`"""

    linkable_to_prefpath: Dict[str, str]
    """Similar to aliases_d, but can have arbitrary keys that map to preferred dotpaths."""

    def get_linking_writer(self, f: Writable) -> LinkingWriter:
        return LinkingWriter(
            f,
            link_aliases=self.linkable_to_prefpath,
            link_d=self.link_d,
            refdoc_relpath=self.refdoc_relpath,
        )
