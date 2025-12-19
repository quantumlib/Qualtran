#  Copyright 2023 Google LLC
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

"""A recursive-descent parser for bloq string representation."""
import json
import logging
import re
from typing import List

import attrs

logger = logging.getLogger(__name__)

from .nodes import CArgNode, CObjectNode, CValueNode, L1ASTNode, LiteralNode, TupleNode


@attrs.frozen
class Token:
    type: str
    value: str
    line: int
    column: int


def tokenize(code: str) -> List[Token]:
    """Turn a string into a list of tokens."""
    token_specification = [
        ('NUMBER', r'(\-)?\d+(\.\d*)?(e[+\-]\d+)?'),
        ('NAME', r'[A-Za-z_][A-Za-z_0-9]*'),
        ('STRING', r"'[^']*'|\"[^\"]*\""),
        ('RARROW', r'\-\>'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACK', r'\['),
        ('RBRACK', r'\]'),
        ('LCURLY', r'\{'),
        ('RCURLY', r'\}'),
        ('EQUALS', r'='),
        ('COMMA', r','),
        ('COMMENT', r'#.*'),
        ('DOT', r'\.'),
        ('COLON', r':'),
        ('PIPE', r'\|'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    tokens = []
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'STRING':
            tokens.append(Token(kind, value[1:-1], line_num, column))  # remove quotes
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind == 'SKIP':
            pass
        elif kind == 'COMMENT':
            pass
        elif kind == 'MISMATCH':
            raise ValueError(f'{value!r} unexpected on line {line_num}')
        else:
            tokens.append(Token(kind, value, line_num, column))

    tokens.append(Token('EOF', '', line_num, 0))
    return tokens


class QualtranL1Parser:
    """A recursive-descent parser for bloq strings."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        """Look at the next token without consuming it."""
        return self.tokens[self.pos]

    def prev(self) -> Token:
        return self.tokens[self.pos - 1]

    def check(self, tt: str) -> bool:
        return self.peek().type == tt

    def advance(self):
        token = self.peek()
        self.pos += 1
        return token

    def consume(self, tt: str, err_msg: str) -> Token:
        """Consume the next token."""
        if self.check(tt):
            return self.advance()
        raise ValueError(f"{err_msg} at {self.peek()}")

    def _advance_to_next_list_item(
        self, rbrack='RBRACK', delim='COMMA', *, consume_rbrack: bool = True, list_name: str
    ) -> bool:
        """Helper method for processing a delimited list of items with optional trailing comma.

        Specifically, we support lists like '[a, b, c]' and '[a, b, c,]'.

        This should be called after processing a list item. If empty lists are allowed,
        they must be handled as a special case before processing list items.

        Args:
            rbrack: The token type that indicates the end of the grouping, typically ')', ']', '}'.
            delim: The token type that indicates the delimiter between items, typically ','.
            consume_rbrack: Whether to consume the final end-grouping item.
            list_name: The name of the list (used during error reporting).

        Returns:
            whether we reached the end of the list.
        """
        if self.check(delim):
            # Assume we already checked for an empty list and parsed an item.
            # A delimiter after an entry is always valid.
            self.advance()
        else:
            # A missing delimiter demands end of list
            if not self.check(rbrack):
                raise ValueError(f"Extraneous elements in {list_name} at {self.peek()}")

        if self.check(rbrack):
            if consume_rbrack:
                self.advance()
            return True
        return False

    def parse_cobject_only(self) -> CObjectNode:
        ret = self.parse_cobject_node()
        self.consume('EOF', 'Expected EOF')
        return ret

    def parse_bloq_key(self) -> str:
        return self.parse_cobject_node().canonical_str()

    def parse_cobject_node(self) -> CObjectNode:
        """Parse a classical object 'FuncName(arg, v1, k2=v2)'.

        The ()-list of arguments is optional. 'FuncName' will also parse.

        cobject:
            | qualified_identifier
            | qualified_identifier '(' ','.carg* ')'
        """
        name = self.parse_qualified_identifier()

        if not self.check('LPAREN'):
            return CObjectNode(name=name, cargs=())
        self.advance()  # LPAREN

        if self.check('RPAREN'):
            # Empty arg list
            self.advance()  # RPAREN
            return CObjectNode(name=name, cargs=tuple())

        cargs = []
        while True:
            carg = self.parse_carg()
            cargs.append(carg)

            done = self._advance_to_next_list_item('RPAREN', list_name='classical args')
            if done:
                return CObjectNode(name=name, cargs=tuple(cargs))

    def parse_qualified_identifier(self) -> str:
        """Parse a dot-separated identifier.

        qualified_identifier:
            | qualified_identifier '.' NAME
            | NAME
        """
        parts = [self.consume('NAME', 'Expected a valid identifier').value]
        while self.check('DOT'):
            self.advance()  # DOT
            parts.append(self.consume('NAME', 'Expected a valid identifier').value)
        return '.'.join(parts)

    def parse_carg(self) -> CArgNode:
        """Parse a classical 'arg'.

        carg:
            | key '=' cvalue
            | cvalue

        key: NAME
        """
        if self.check('NAME'):
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'EQUALS':
                key = self.advance().value  # NAME
                self.advance()  # EQUALS
                value = self.parse_cvalue()
                return CArgNode(key=key, value=value)

        value = self.parse_cvalue()
        return CArgNode(key=None, value=value)

    def parse_int_literal(self, err_ctx: str = 'parsing') -> int:
        tok = self.consume('NUMBER', f'expected an integer literal in {err_ctx}.')
        try:
            return int(tok.value)
        except ValueError as e:
            raise ValueError(f"expected an integer literal in {err_ctx}, not {tok.value}") from e

    def parse_cvalue(self) -> CValueNode:
        """Parse a value (literal or ccall).

        cvalue:
            | tuple
            | literal
            | cobject

        tuple:
            | '(' ','.cvalue* ')'

        literal:
            literal_float
            literal_int
            literal_str
        """
        token = self.peek()

        if token.type == 'LPAREN':
            # The cvalue is a tuple/list
            self.advance()  # LPAREN

            # Check for empty list
            if self.check('RPAREN'):
                self.advance()  # RPAREN
                return TupleNode(items=())

            # Get values
            vals = []
            while True:
                val = self.parse_cvalue()
                vals.append(val)
                done = self._advance_to_next_list_item('RPAREN', list_name='cvalue list')
                if done:
                    return TupleNode(items=tuple(vals))

        if token.type == 'NUMBER':
            # The cvalue is a number
            self.advance()  # NUMBER
            if '.' in token.value or 'e' in token.value:
                return LiteralNode(value=float(token.value))
            return LiteralNode(value=int(token.value))
        if token.type == 'STRING':
            # The cvalue is a string
            self.advance()  # STRING
            return LiteralNode(value=str(token.value))
        if token.type == 'NAME':
            # The cvalue is a cobject
            return self.parse_cobject_node()

        raise ValueError(f"Unexpected token {token} when parsing value")


def parse_objectstring(objectstring: str) -> CObjectNode:
    tokens = tokenize(objectstring)
    parser = QualtranL1Parser(tokens)
    cval_node: CObjectNode = parser.parse_cobject_only()
    return cval_node


def _l1_to_json_dict(self):
    d = {'_l1_node': self.__class__.__name__}
    for field in attrs.fields(self.__class__):
        v = getattr(self, field.name)
        d[field.name] = v
    return d


def l1_ast_node_to_json(o: object):
    if isinstance(o, L1ASTNode):
        return _l1_to_json_dict(o)
    return o


def dump_ast(node: L1ASTNode, f):
    json.dump(node, f, indent=2, default=l1_ast_node_to_json)
