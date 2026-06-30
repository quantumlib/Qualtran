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
from typing import List, Optional, Tuple

import attrs

from . import nodes as qualtran_l1_nodes
from .nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    CValueNode,
    L1ASTNode,
    L1Module,
    L1Nodes,
    LiteralNode,
    LValueNode,
    NestedQArgValue,
    QArgNode,
    QArgValueNode,
    QCallNode,
    QCastNode,
    QDefExternNode,
    QDefImplNode,
    QDefNode,
    QDTypeNode,
    QReturnNode,
    QSignatureEntry,
    StatementNode,
    TupleNode,
)

logger = logging.getLogger(__name__)


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
        ('AT', r'@'),
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
        assert kind is not None
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

    def __init__(self, tokens: List[Token], nodes: L1Nodes = qualtran_l1_nodes):
        self.tokens = tokens
        self.pos = 0
        self.nodes = nodes

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
        # A missing delimiter demands end of list
        elif not self.check(rbrack):
            raise ValueError(f"Extraneous elements in {list_name} at {self.peek()}")

        if self.check(rbrack):
            if consume_rbrack:
                self.advance()
            return True
        return False

    def parse_annotation(self) -> Optional[CValueNode]:
        if self.check('AT'):
            self.advance()
            return self.parse_cvalue()
        return None

    def parse_module(self) -> L1Module:
        """Parse an L1 module consisting of multiple quantum definitions.

        A module is a sequence of `qdef` or `extern qdef` blocks representing
        a complete Qualtran-L1 file or program.

        Returns:
            An L1Module containing the parsed quantum definitions.
        """
        if self.check('EOF'):
            return self.nodes.L1Module(qdefs=[])

        qdefs = self.parse_qdefs()

        self.consume('EOF', "Expected EOF")
        return self.nodes.L1Module(qdefs=qdefs)

    def parse_qdefs(self) -> List[QDefNode]:
        """Parse a sequence of quantum function, or cast definitions.

        Reads zero or more `qcast`, `qdef` or `extern qdef` keywords and parses them
        into QDefNode instances.

        Returns:
            qdefs: A list of QDefNode (implementations, externs, or casts).
        """
        qdefs = []
        while True:
            if self.check('EOF'):
                break

            tok = self.consume('NAME', "Expected an identifier")
            if tok.value == 'qdef':
                qdef = self.parse_qdef()
                qdefs.append(qdef)
            elif tok.value == 'extern':
                tok = self.consume('NAME', "Expected 'extern qdef'")
                if tok.value != 'qdef':
                    raise ValueError(f"Expected 'extern qdef', not 'extern {tok}'")
                qdef = self.parse_qdef(extern=True)
                qdefs.append(qdef)
            elif tok.value == 'qcast':
                qcast = self.parse_qcast()
                qdefs.append(qcast)
            else:
                raise ValueError(f"Unexpected identifier {tok}")

        return qdefs

    def parse_qcast(self) -> QCastNode:
        """Parse a qcast (casting operation) definition.

        A qcast consists of a bloq key and a quantum signature.
        It has no ``from`` clause and no body.

        ```qlt
        qcast Split(QUInt(4))
        [reg: QUInt(4) -> QBit[4]]
        ```

        Returns:
            A QCastNode.
        """
        bloq_key = self.parse_bloq_key()
        qsig = self.parse_qdef_signature()
        return self.nodes.QCastNode(bloq_key=bloq_key, qsignature=qsig)

    def parse_qdef(self, extern: bool = False) -> QDefNode:
        """Parse a single quantum function definition.

        A qdef consists of a signature, an optional classical 'from' binding,
        and either a body (for implementations) or no body (for externs).

        Args:
            extern: Whether to parse this as an 'extern qdef' block.

        Returns:
            A QDefImplNode or QDefExternNode based on the `extern` flag.
        """
        bloq_key = self.parse_bloq_key()
        if self.check('NAME') and self.peek().value == 'from':
            bobj_cval = self.parse_qdef_from()
        else:
            bobj_cval = None

        qsig = self.parse_qdef_signature()

        if extern:
            return self.nodes.QDefExternNode(
                bloq_key=bloq_key, qsignature=qsig, cobject_from=bobj_cval
            )
        else:
            statements = self.parse_qdef_body()
            return self.nodes.QDefImplNode(
                bloq_key=bloq_key, qsignature=qsig, body=statements, cobject_from=bobj_cval
            )

    def parse_qdef_from(self) -> CObjectNode:
        """qdef key() from qualtran.bloqs.BloqCls(x, param=y) [...] {...}"""
        tok = self.consume('NAME', "qdef from must start with 'from'")
        if tok.value != 'from':
            raise ValueError("qdef from must start with 'from'")
        bobj_cval = self.parse_cobject_node()
        return bobj_cval

    def parse_qdef_signature(self) -> List[QSignatureEntry]:
        self.consume('LBRACK', "quantum signature must start with [")

        if self.check('RBRACK'):
            # Empty list is valid.
            self.advance()
            return []

        qsig: List[QSignatureEntry] = []
        while True:
            qsig_entry = self.parse_qsig_entry()
            qsig.append(qsig_entry)

            done = self._advance_to_next_list_item('RBRACK', list_name='quantum signature')
            if done:
                return qsig

    def parse_qsig_entry(self) -> QSignatureEntry:
        """A 'qvar: QDType(k=v)[2, 4]' entry in the quantum signature.

        The quantum signature encodes both quantum inputs and outputs. A simple entry
        'qvar: t' indicates a THRU-register of type 't'. For output-only (RIGHT) and
        intput-only (LEFT) entries, the signature is 'qvar: t -> |' and 'qvar: | -> t',
        respectively. For casting registers, the syntax is 'qvar: t1 -> t2'.
        """
        name_tok = self.consume('NAME', 'invalid identifier for quantum signature entry.')
        self.consume('COLON', 'missing colon-delimited datatype in quantum signature')

        # Per docstring, there are four valid formulations for the datatype portion of
        # the signature entry. First, we suss out the '| -> t' case.
        if self.check('PIPE'):
            # | -> t
            self.advance()
            self.consume('RARROW', 'output-only datatypes must be formulated | -> t')
            dtype = self.parse_qsig_dtype()
            annotation = self.parse_annotation()
            return self.nodes.QSignatureEntry(
                name=name_tok.value, dtype=(None, dtype), annotation=annotation
            )

        # Per docstring, there are 3 out of 4 possible formulations for the datatype portion of
        # the signature entry. 't', 't -> |', 't1 -> t2'. Each starts with a data type.
        t1 = self.parse_qsig_dtype()
        if self.check('RARROW'):
            self.advance()
            # Narrowed down to either 't -> |' or 't1 -> t2'
            if self.peek().type == 'PIPE':
                # t -> |
                self.advance()
                annotation = self.parse_annotation()
                return self.nodes.QSignatureEntry(
                    name=name_tok.value, dtype=(t1, None), annotation=annotation
                )
            # t1 -> t2
            t2 = self.parse_qsig_dtype()
            annotation = self.parse_annotation()
            return self.nodes.QSignatureEntry(
                name=name_tok.value, dtype=(t1, t2), annotation=annotation
            )

        # We've eliminated all possibilities except the basic 't' syntax.
        annotation = self.parse_annotation()
        return self.nodes.QSignatureEntry(name=name_tok.value, dtype=t1, annotation=annotation)

    def parse_qsig_dtype(self) -> QDTypeNode:
        """QDType(k=v)[2, 4]"""
        cls = self.parse_cobject_node()
        if self.check('LBRACK'):
            shape = self.parse_shape_list()
            return self.nodes.QDTypeNode(dtype=cls, shape=shape)
        return self.nodes.QDTypeNode(dtype=cls, shape=None)

    def parse_shape_list(self) -> List[int]:
        """[1, 2, 3]"""
        self.consume('LBRACK', "datatype shapes must begin with '['")
        if self.check('RBRACK'):
            self.advance()
            return []

        items: List[int] = []
        while True:
            item = self.parse_int_literal(err_ctx='datatype shape list')
            items.append(item)

            done = self._advance_to_next_list_item(list_name='datatype shape')
            if done:
                return items

    def parse_qdef_body(self) -> List[StatementNode]:
        """Curly-brace delimited sequence of statements."""
        self.consume('LCURLY', 'qdef body must start with {')
        if self.check('RCURLY'):
            # Empty body isn't super valid, but we'll parse it.
            self.advance()
            return []

        statements = []
        while True:
            statement = self.parse_statement()
            statements.append(statement)
            if self.check('RCURLY'):
                self.advance()
                return statements

    def parse_statement(self) -> StatementNode:
        if self.peek().type == 'NAME' and self.peek().value == 'return':
            return self.parse_return_statement()

        lvalues = self.parse_lvalues()
        self.consume('EQUALS', "Assignment operator '=' expected")
        bloq_key = self.parse_bloq_key()
        annotation = self.parse_annotation()
        if self.check('LBRACK'):
            qargs = self.parse_qargs()
            return self.nodes.QCallNode(
                bloq_key=bloq_key, lvalues=lvalues, qargs=qargs, annotation=annotation
            )
        else:
            if len(lvalues) != 1:
                raise ValueError(
                    f"Syntax error: during alias assignment, only one lvalue may be specified (at {self.peek()})"
                )
            if lvalues[0].annotation is not None:
                raise ValueError(f"Syntax error: alias assignment lvalue cannot have an annotation")
            alias = lvalues[0].name
            return self.nodes.AliasAssignmentNode(alias=alias, bloq_key=bloq_key)

    def parse_return_statement(self) -> QReturnNode:
        ret_tok = self.consume('NAME', "return statement must start with 'return'")
        if ret_tok.value != 'return':
            raise ValueError("return statement must start with 'return'")

        qargs = self.parse_qargs()
        return self.nodes.QReturnNode(qargs)

    def parse_lvalues(self) -> List[str]:
        """Parse a comma-separated list of l-values.

        L-values are the targets of an assignment statement, typically
        quantum variables or the special pipe character `|` denoting
        an empty list of targets.

        Returns:
            A list of string identifiers for the l-values.
        """
        if self.check('PIPE'):
            # Empty list of lvalues is specified with '|'.
            # .. shows up in e.g. | = globalphase[]
            self.advance()
            return []

        lvalues = []
        while True:
            ident_tok = self.consume('NAME', 'Expected identifier for lvalue')
            annotation = self.parse_annotation()
            lvalues.append(self.nodes.LValueNode(name=ident_tok.value, annotation=annotation))

            done = self._advance_to_next_list_item(
                'EQUALS', consume_rbrack=False, list_name='lvalues'
            )
            if done:
                return lvalues

    def parse_qargs(self) -> List[QArgNode]:
        """[ctrl=[qvar[0], qvar[1]], target=trg]"""
        self.consume('LBRACK', 'qargs must start with [')
        if self.check('RBRACK'):
            # Empty list is valid.
            self.advance()
            return []

        args = []
        while True:
            arg = self.parse_qarg()
            args.append(arg)

            done = self._advance_to_next_list_item(list_name='qargs')
            if done:
                return args

    def parse_qarg(self) -> QArgNode:
        """ctrl=[qvar[0], qvar[1]]"""
        key = self.consume('NAME', 'invalid qarg key').value

        # Support array indexing on key e.g. reg[0]=...
        if self.check('LBRACK'):
            self.advance()
            idx = self.parse_int_literal('qarg key index')
            self.consume('RBRACK', 'missing closing bracket for qarg key index')
            key = f"{key}[{idx}]"

        self.consume('EQUALS', 'invalid qargs k=v specification')

        val: NestedQArgValue
        if self.check('LBRACK'):
            # Start a list
            val = self.parse_qarg_value_list()
        else:
            val = self.parse_qarg_value()

        annotation = self.parse_annotation()
        return self.nodes.QArgNode(key=key, value=val, annotation=annotation)

    def parse_qarg_value_list(self) -> List[NestedQArgValue]:
        """[qvar[0], qvar[1]].

        Lists can be arbitrarily nested.
        """
        qarg_value_list: List[NestedQArgValue] = []
        self.consume('LBRACK', 'qarg value list must start with [')
        while True:
            if self.check('LBRACK'):
                sublist = self.parse_qarg_value_list()
                qarg_value_list.append(sublist)
            else:
                val = self.parse_qarg_value()
                qarg_value_list.append(val)

            done = self._advance_to_next_list_item(list_name='qarg_value_list')
            if done:
                return qarg_value_list

    def parse_qarg_value(self) -> QArgValueNode:
        """The value in a target=trg qargs assignment.

        Can be 'trg' or include an optional index 'qvar[5,6]'.
        """
        value_tok = self.consume('NAME', 'qarg value must start with a valid identifier')

        if self.check('LBRACK'):
            # Get the optional index
            self.advance()
            idx = []
            while True:
                i = self.parse_int_literal('qarg value index')
                idx.append(i)

                done = self._advance_to_next_list_item('RBRACK', list_name='qarg value index')
                if done:
                    return self.nodes.QArgValueNode(name=value_tok.value, idx=tuple(idx))
        else:
            return self.nodes.QArgValueNode(name=value_tok.value, idx=tuple())

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
            return self.nodes.CObjectNode(name=name, cargs=())
        self.advance()  # LPAREN

        if self.check('RPAREN'):
            # Empty arg list
            self.advance()  # RPAREN
            return self.nodes.CObjectNode(name=name, cargs=tuple())

        cargs = []
        while True:
            carg = self.parse_carg()
            cargs.append(carg)

            done = self._advance_to_next_list_item('RPAREN', list_name='classical args')
            if done:
                return self.nodes.CObjectNode(name=name, cargs=tuple(cargs))

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
                return self.nodes.CArgNode(key=key, value=value)

        value = self.parse_cvalue()
        return self.nodes.CArgNode(key=None, value=value)

    def parse_int_literal(self, err_ctx: str = 'parsing') -> int:
        """Parse an integer literal.

        Reads a 'NUMBER' token and attempts to convert it to an integer.

        Args:
            err_ctx: An error context string to include in error messages
                if parsing fails.

        Returns:
            The parsed integer value.

        Raises:
            ValueError: If the token is not a valid integer.
        """
        tok = self.consume('NUMBER', f'Expected an integer literal in {err_ctx}')
        try:
            return int(tok.value)
        except ValueError as e:
            raise ValueError(f"Expected an integer literal in {err_ctx}, not {tok.value}") from e

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
                return self.nodes.TupleNode(items=())

            # Get values
            vals = []
            while True:
                val = self.parse_cvalue()
                vals.append(val)
                done = self._advance_to_next_list_item('RPAREN', list_name='cvalue list')
                if done:
                    return self.nodes.TupleNode(items=tuple(vals))

        if token.type == 'NUMBER':
            # The cvalue is a number
            self.advance()  # NUMBER
            if '.' in token.value or 'e' in token.value:
                return self.nodes.LiteralNode(value=float(token.value))
            return self.nodes.LiteralNode(value=int(token.value))
        if token.type == 'STRING':
            # The cvalue is a string
            self.advance()  # STRING
            return self.nodes.LiteralNode(value=str(token.value))
        if token.type == 'NAME':
            # The cvalue is a cobject
            return self.parse_cobject_node()

        raise ValueError(f"Unexpected token {token} when parsing value")


def parse_objectstring(objectstring: str, *, nodes: L1Nodes = qualtran_l1_nodes) -> CObjectNode:
    """Parse a classical object string representing a Bloq instance.

    Args:
        objectstring: The string representation of the object, e.g., 'MyBloq(1)'.
        nodes: The module providing the AST node constructors.

    Returns:
        A CObjectNode representing the parsed classical object.
    """
    tokens = tokenize(objectstring)
    parser = QualtranL1Parser(tokens, nodes=nodes)
    cval_node: CObjectNode = parser.parse_cobject_only()
    return cval_node


def parse_module(l1_code: str, *, nodes: L1Nodes = qualtran_l1_nodes) -> L1Module:
    """Parse an entire L1 code string into an L1Module.

    Args:
        l1_code: The Qualtran-L1 source code as a string.
        nodes: The module providing the AST node constructors.

    Returns:
        An L1Module containing the root AST of the parsed code.
    """
    tokens = tokenize(l1_code)
    parser = QualtranL1Parser(tokens, nodes=nodes)
    return parser.parse_module()


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
