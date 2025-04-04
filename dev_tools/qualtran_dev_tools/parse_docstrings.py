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

import inspect
import re
from typing import List, Type, Union

from attrs import frozen
from sphinx.ext.napoleon import Config, GoogleDocstring


@frozen(kw_only=True)
class Reference:
    """A structured reference from the References section"""

    url: str
    title: str
    extra: str

    @property
    def text(self):
        return f'[{self.title}]({self.url}). {self.extra}'


@frozen
class UnparsedReference:
    """Fallback for a reference that couldn't be parsed."""

    text: str


ReferenceT = Union[Reference, UnparsedReference]


def parse_reference(ref_text: str) -> ReferenceT:
    ref_text = re.sub(r'\n', ' ', ref_text)

    # To match the title and accept as many printable ascii characters as possible
    # besides square brackets, I have modified the range approach from
    # https://stackoverflow.com/a/31740504.
    link_match = re.match(r'^\[([ -Z^-~]+)]\((https?://[\w\d./?=#\-\(\)]+)\)\.?\s?(.*)$', ref_text)
    if link_match:
        title = link_match.group(1)
        url = link_match.group(2)
        rest_of_ref = link_match.group(3)
        return Reference(title=title, url=url, extra=rest_of_ref)

    return UnparsedReference(ref_text)


class _GoogleDocstringToMarkdown(GoogleDocstring):
    """Subclass of sphinx's parser to emit Markdown from Google-style docstrings."""

    def __init__(self, *args, **kwargs):
        self.references: List[ReferenceT] = []
        super().__init__(*args, **kwargs)

    def _load_custom_sections(self) -> None:
        super()._load_custom_sections()
        self._sections['registers'] = self._parse_registers_section

    def _parse_parameters_section(self, section: str) -> List[str]:
        """Sphinx method to emit a 'Parameters' section."""

        def _template(name, desc_lines):
            desc = ' '.join(desc_lines)
            return f' - `{name}`: {desc}'

        return [
            '#### Parameters',
            *[_template(name, desc) for name, _type, desc in self._consume_fields()],
            '',
        ]

    def _parse_references_section(self, section: str) -> List[str]:
        """Sphinx method to emit a 'References' section."""
        lines = self._dedent(self._consume_to_next_section())

        full_reference_text = '\n'.join(lines)
        reference_texts = re.split(r'\n\n', full_reference_text)
        my_refs = []
        for ref_text in reference_texts:
            ref = parse_reference(ref_text)
            my_refs.append(ref)

        self.references.extend(my_refs)
        return ['#### References', '\n'.join(f' - {ref.text}' for ref in my_refs), '']

    def _parse_registers_section(self, section: str) -> List[str]:
        def _template(name, desc_lines):
            desc = ' '.join(desc_lines)
            return f' - `{name}`: {desc}'

        return [
            '#### Registers',
            *[_template(name, desc) for name, _type, desc in self._consume_fields()],
            '',
        ]


def get_markdown_docstring(cls: Type) -> List[str]:
    """From a class `cls`, return its docstring as Markdown lines."""

    # 1. Sphinx incantation
    config = Config()
    docstring = cls.__doc__ if cls.__doc__ else ""
    gds = _GoogleDocstringToMarkdown(inspect.cleandoc(docstring), config=config, what='class')

    # 2. Substitute restructured text inline-code blocks to markdown-style backticks.
    lines = [re.sub(r':py:func:`(\w+)`', r'`\1`', line) for line in gds.lines()]

    return lines


def get_markdown_docstring_lines(cls: Type) -> List[str]:
    """From a class `cls`, return its docstring as Markdown lines with a header."""

    # 1. Get documentation lines
    lines = get_markdown_docstring(cls)

    # 2. Pre-pend a header.
    lines = [f'## `{cls.__name__}`'] + lines

    return lines


def get_references(cls: Type) -> List[ReferenceT]:
    """Get reference information for a class from the References section of its docstring."""
    config = Config()
    docstring = cls.__doc__ if cls.__doc__ else ""
    gds = _GoogleDocstringToMarkdown(inspect.cleandoc(docstring), config=config, what='class')
    return gds.references
