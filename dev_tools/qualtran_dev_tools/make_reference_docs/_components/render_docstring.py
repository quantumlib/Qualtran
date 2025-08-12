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
import re
import warnings
from typing import List, Literal, Tuple, TYPE_CHECKING

from griffe import (
    DocstringSection,
    DocstringSectionAdmonition,
    DocstringSectionAttributes,
    DocstringSectionExamples,
    DocstringSectionKind,
    DocstringSectionParameters,
    DocstringSectionRaises,
    DocstringSectionReturns,
    DocstringSectionText,
    DocstringSectionYields,
    DocstringYield,
)

from ...parse_docstrings import parse_references

if TYPE_CHECKING:
    import griffe

PARSER: Literal['google'] = 'google'


def split_docstring(obj: 'griffe.Object') -> Tuple[str, List[DocstringSection]]:
    """Extract the first, summary line of a docstring from an object.

    Returns:
        first_line: The first line comprising a summary of the object, as a string
        rest: A list of `DocstringSection`. These can be passed to `write_docstring_parts`.
    """

    if obj.docstring is None:
        first_line = ""
        rest: List[DocstringSection] = []
        return first_line, rest

    dp0, *dparts = obj.docstring.parse(PARSER)
    if dp0.kind is not DocstringSectionKind.text:
        raise ValueError(obj)

    first_line, *other_lines = re.split(r'\n{2,}', dp0.value, flags=re.MULTILINE)
    other_lines = '\n\n'.join(other_lines).strip()
    if not other_lines:
        return first_line, dparts

    dp0 = DocstringSectionText(other_lines)
    rest = [dp0] + dparts
    return first_line, rest


def _write_text(f, part: DocstringSectionText, level: int):
    f.write(part.value.strip())
    f.write('\n\n')


def _write_parameters(f, part: DocstringSectionParameters, level: int):
    lvl = '#' * level
    f.write(f'###{lvl} Args\n')
    for param in part.value:
        if param.annotation:
            f.write(f'{param.name}: `{param.annotation}`\n')
        else:
            f.write(f'{param.name}\n')
        f.write(f': {param.description}')  # myst syntax for definition lists in markdown
        f.write('\n\n')


def _write_attributes(f, part: DocstringSectionAttributes, level: int):
    lvl = '#' * level
    f.write(f'###{lvl} Attributes\n')
    for attr in part.value:
        if attr.annotation:
            f.write(f'{attr.name}: `{attr.annotation}`\n')
        else:
            f.write(f'{attr.name}\n')
        f.write(f': {attr.description}')
        f.write('\n\n')


def _write_returns(f, part: DocstringSectionReturns, level: int):
    lvl = '#' * level
    f.write(f'###{lvl} Returns\n')

    # One, unnamed return value
    if len(part.value) == 1 and not part.value[0].name:
        f.write(part.value[0].description)
        f.write('\n')

    # Multiple return values and/or named return values
    else:
        for param in part.value:
            # f.write(f'{param.name=} {param.value=} {param.annotation=} {param.description=}')
            pname = param.name if param.name else 'ret'
            f.write(f'{pname}\n')
            f.write(f': {param.description}')  # myst syntax for definition lists in markdown
            f.write('\n\n')


def _write_admonition(f, part: DocstringSectionAdmonition, level: int):
    lvl = '#' * level
    part = part.value
    if part.kind == 'see-also':
        f.write(f'###{lvl} See Also\n')
        f.write(part.description)
        f.write('\n\n')
    elif part.kind == 'references':
        f.write(f'###{lvl} References\n')
        refs = parse_references(part.contents)
        for ref in refs:
            f.write(f' - {ref.text}\n')
        f.write('\n')
    elif part.kind == 'note':
        f.write(f'**Note:** {part.contents}\n\n')
    else:
        warnings.warn(f"Unknown admonition type {part.kind}")


def _write_examples(f, part: DocstringSectionExamples, level: int):
    # todo? "examples" heading?
    part = part.value
    for subkind, subtext in part:
        if subkind is DocstringSectionKind.examples:
            f.write('```pycon\n')
            f.write(subtext)
            f.write('\n````')
        elif subkind is DocstringSectionKind.text:
            f.write(subtext)
        else:
            raise ValueError()
        f.write('\n')
    f.write('\n\n')


def _write_raises(f, part: DocstringSectionRaises, level: int):
    lvl = '#' * level
    f.write(f'###{lvl} Raises\n')
    for subpart in part.value:
        if not subpart.annotation:
            raise ValueError()
        f.write(f'{subpart.annotation}\n')
        f.write(f': {subpart.description}')
        f.write('\n\n')


def _write_yields(f, part: DocstringSectionYields, level: int):
    lvl = '#' * level
    f.write(f'###{lvl} Yields\n')

    # One, unnamed yield value
    if len(part.value) == 1 and not part.value[0].name:
        f.write(part.value[0].description)
        f.write('\n')

    else:
        subpart: DocstringYield
        for param in part.value:
            pname = param.name if param.name else 'yld'
            f.write(f'{pname}\n')
            f.write(f': {param.description}')  # myst syntax for definition lists in markdown
            f.write('\n\n')


def _write_unknown(f, part: DocstringSection, level: int):
    f.write(str(part))
    f.write('\n')
    warnings.warn(f"Unknown docstring {part}")


_DISPATCH = {
    DocstringSectionKind.text: _write_text,
    DocstringSectionKind.parameters: _write_parameters,
    DocstringSectionKind.returns: _write_returns,
    DocstringSectionKind.admonition: _write_admonition,
    DocstringSectionKind.examples: _write_examples,
    DocstringSectionKind.raises: _write_raises,
    DocstringSectionKind.yields: _write_yields,
    DocstringSectionKind.attributes: _write_attributes,
}


def write_docstring_parts(f, parts: List[DocstringSection], level: int):
    for part in parts:
        _write = _DISPATCH.get(part.kind, _write_unknown)
        _write(f, part, level)  # type: ignore
    f.write('\n\n')
