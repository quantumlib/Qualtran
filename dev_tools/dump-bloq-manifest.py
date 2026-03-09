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

"""
Script to dump a manifest of all bloq classes and bloq examples.

This script finds all Bloq classes and examples in the library, serializes the examples
to their objectstring representation, and writes the list of class names and
example objectstrings to `qualtran/bloqs/manifest.py`.

See `qualtran.l1.load_objectstring()` to load bloq objects from these strings.
"""

from functools import cached_property
from typing import List, Tuple

from attrs import frozen
from qualtran_dev_tools.bloq_finder import get_bloq_classes, get_bloq_examples
from qualtran_dev_tools.git_tools import get_git_root

from qualtran import Bloq
from qualtran.l1 import eval_cvalue_node, parse_objectstring, to_cobject_node
from qualtran.l1.nodes import CArgNode, CObjectNode, LiteralNode

MAXLEN = 300
"""If the objectstring is too long, we make the executive decision to truncate it."""

COPYRIGHT_NOTICE = """#  Copyright 2026 Google LLC
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
#  limitations under the License."""


@frozen
class BloqExampleListItem:
    """Formatting a bloq example and its serialized form.

    Attributes:
        bloq: The bloq instance from the example.
        bloq_example_name: The name of the bloq example.
        cobject_node: The CObjectNode representing the serialized bloq.
    """

    bloq: Bloq
    bloq_example_name: str
    cobject_node: CObjectNode

    @cached_property
    def objectstring(self) -> str:
        """Returns the canonical string representation of the CObjectNode."""
        return self.cobject_node.canonical_str()

    def maybe_commented_out(self, be_column_width: int = 30) -> Tuple[str, str, str]:
        """Generates a string representation for the manifest, potentially commented out.

        This method checks if the object string is too long, unparsable, unloadable, or
        if the re-loaded bloq is unequal to the original. If any of these conditions are met,
        it returns a commented-out string with a reason. Otherwise, it returns the
        executable string representation.

        Args:
            be_column_width: The width of the column for the bloq example name for nicer formatting.

        Returns:
            manifest_entry_str: The string to be written to the manifest (possibly commented out).
            reason: A reason string if it was commented out (empty otherwise).
            reason_details: Detailed error message if applicable (empty otherwise).
        """
        quoted_be = f"'{self.bloq_example_name}',"
        be = f'{quoted_be:{be_column_width}}'
        be2 = f'{quoted_be:{be_column_width+2}}'

        if len(self.objectstring) > MAXLEN:
            trunc = repr(self.objectstring)[:120] + '...'
            return f'# ({be}{trunc}', 'too long', ''

        try:
            node = parse_objectstring(self.objectstring)
        except Exception as e:  # pylint: disable=broad-except
            return f'# ({be}{self.objectstring!r}),', 'unparsable', str(e)

        try:
            bloq = eval_cvalue_node(node, safe=False)
        except Exception as e:  # pylint: disable=broad-except
            return f'# ({be}{self.objectstring!r}),', 'unloadable', str(e)

        if bloq != self.bloq:
            return f'# ({be}{self.objectstring!r}),', 'unequal', ''

        return f'({be2}{self.objectstring!r}),', '', ''


def main():
    """Main entry point for the script.

    Finds all bloq classes and examples, processes them, and writes the
    `BLOQ_CLASS_NAMES` and `BLOQ_EXAMPLE_OBJECTSTRINGS` lists to
    `qualtran/bloqs/manifest.py`.
    """
    bcs = get_bloq_classes()
    names = sorted(bc._class_name_in_pkg_() for bc in bcs)

    bes = get_bloq_examples()
    items: List[BloqExampleListItem] = []
    for be in bes:
        bloq = be.make()
        try:
            cobject_node = to_cobject_node(bloq)
            assert isinstance(cobject_node, CObjectNode)
        except Exception as e:  # pylint: disable=broad-except
            cobject_node = CObjectNode(
                name=bloq._class_name_in_pkg_(), cargs=[CArgNode(None, LiteralNode(str(e)))]
            )
        items.append(
            BloqExampleListItem(bloq=bloq, bloq_example_name=be.name, cobject_node=cobject_node)
        )

    items = sorted(items, key=lambda x: x.objectstring)
    include_commented_out = True
    be_objectstrings = []
    for item in items:
        serstr, reason, details = item.maybe_commented_out()

        if (not reason) or include_commented_out:
            be_objectstrings.append(serstr)

        if reason:
            reason = f'({reason})'
            print(f"Skipping {reason:20s} {serstr}")

        if details:
            print(f'         {"":20s} ->', details)

    reporoot = get_git_root()
    with (reporoot / 'qualtran/bloqs/manifest.py').open('w') as f:
        f.write(COPYRIGHT_NOTICE)
        f.write('\n\n')
        f.write('# This file is autogenerated\n')
        f.write('# See dev_tools/dump-bloq-manifest.py\n')
        f.write('# fmt: off\n\n')
        f.write('BLOQ_CLASS_NAMES = [\n')
        f.write('\n'.join([f'    "{name}",' for name in names]))
        f.write('\n]\n')

        f.write('\n')

        f.write('BLOQ_EXAMPLE_OBJECTSTRINGS = [\n')
        f.write('\n'.join([f'    {objstr}' for objstr in be_objectstrings]))
        f.write('\n]\n')


if __name__ == '__main__':
    main()
