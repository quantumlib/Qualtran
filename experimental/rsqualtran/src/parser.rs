use crate::nodes::{
    AliasAssignmentNode, CArgNode, CObjectNode, CValueNode, L1Module, LiteralNode, LiteralVal,
    NestedQArgValue, QArgNode, QArgValueNode, QCallNode, QCastNode, QDTypeNode, QDefExternNode,
    QDefImplNode, QDefNode, QReturnNode, QSignatureEntry, SignatureDType, Span, StatementNode,
    TupleNode,
};
use regex::Regex;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {}, col {}",
            self.message, self.span.start_line, self.span.start_col
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Number,
    Name,
    String,
    RArrow,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LCurly,
    RCurly,
    Equals,
    Comma,
    Dot,
    Colon,
    Pipe,
    At,
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub value: String,
    pub span: Span,
}

/// Turn a string into a list of tokens.
pub fn tokenize(code: &str) -> (Vec<Token>, Vec<ParseError>) {
    // We keep strings for regex matching logic, but map to TokenKind
    let token_specification = vec![
        ("NUMBER", r"(\-)?\d+(\.\d*)?(e[+\-]\d+)?"),
        ("NAME", r"[A-Za-z_][A-Za-z_0-9]*"),
        ("STRING", r"'[^']*'|\x22[^\x22]*\x22"), // \x22 is double quote
        ("RARROW", r"->"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACK", r"\["),
        ("RBRACK", r"\]"),
        ("LCURLY", r"\{"),
        ("RCURLY", r"\}"),
        ("EQUALS", r"="),
        ("COMMA", r","),
        ("COMMENT", r"#.*"),
        ("DOT", r"\."),
        ("COLON", r":"),
        ("PIPE", r"\|"),
        ("AT", r"@"),
        ("NEWLINE", r"\n"),
        ("SKIP", r"[ \t]+"),
        ("MISMATCH", r"."),
    ];

    let mut tok_regex_parts = Vec::new();
    for (name, pattern) in token_specification {
        tok_regex_parts.push(format!("(?P<{}>{})", name, pattern));
    }
    let tok_regex_str = tok_regex_parts.join("|");
    let tok_regex = Regex::new(&tok_regex_str).unwrap();

    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let mut line_num = 1;
    let mut line_start = 0;

    for caps in tok_regex.captures_iter(code) {
        let (kind_str, value, start, end) = {
            let mut found = None;
            for (i, name) in tok_regex.capture_names().enumerate().skip(1) {
                if let Some(m) = caps.get(i) {
                    if let Some(n) = name {
                        found = Some((n, m.as_str(), m.start(), m.end()));
                        break;
                    }
                }
            }
            if found.is_none() {
                panic!(
                    "No group matched! Matched text: {:?}",
                    caps.get(0).map(|m| m.as_str())
                );
            }
            found.unwrap()
        };

        let column = start - line_start;
        let end_column = column + value.len();
        let span = Span {
            start_line: line_num,
            start_col: column,
            end_line: line_num,
            end_col: end_column,
        };

        match kind_str {
            "NUMBER" => tokens.push(Token {
                kind: TokenKind::Number,
                value: value.to_string(),
                span,
            }),
            "NAME" => tokens.push(Token {
                kind: TokenKind::Name,
                value: value.to_string(),
                span,
            }),
            "STRING" => {
                let s = &value[1..value.len() - 1];
                tokens.push(Token {
                    kind: TokenKind::String,
                    value: s.to_string(),
                    span,
                });
            }
            "RARROW" => tokens.push(Token {
                kind: TokenKind::RArrow,
                value: value.to_string(),
                span,
            }),
            "LPAREN" => tokens.push(Token {
                kind: TokenKind::LParen,
                value: value.to_string(),
                span,
            }),
            "RPAREN" => tokens.push(Token {
                kind: TokenKind::RParen,
                value: value.to_string(),
                span,
            }),
            "LBRACK" => tokens.push(Token {
                kind: TokenKind::LBrack,
                value: value.to_string(),
                span,
            }),
            "RBRACK" => tokens.push(Token {
                kind: TokenKind::RBrack,
                value: value.to_string(),
                span,
            }),
            "LCURLY" => tokens.push(Token {
                kind: TokenKind::LCurly,
                value: value.to_string(),
                span,
            }),
            "RCURLY" => tokens.push(Token {
                kind: TokenKind::RCurly,
                value: value.to_string(),
                span,
            }),
            "EQUALS" => tokens.push(Token {
                kind: TokenKind::Equals,
                value: value.to_string(),
                span,
            }),
            "COMMA" => tokens.push(Token {
                kind: TokenKind::Comma,
                value: value.to_string(),
                span,
            }),
            "DOT" => tokens.push(Token {
                kind: TokenKind::Dot,
                value: value.to_string(),
                span,
            }),
            "COLON" => tokens.push(Token {
                kind: TokenKind::Colon,
                value: value.to_string(),
                span,
            }),
            "PIPE" => tokens.push(Token {
                kind: TokenKind::Pipe,
                value: value.to_string(),
                span,
            }),
            "AT" => tokens.push(Token {
                kind: TokenKind::At,
                value: value.to_string(),
                span,
            }),
            "NEWLINE" => {
                line_start = end;
                line_num += 1;
            }
            "SKIP" | "COMMENT" => {}
            "MISMATCH" => {
                errors.push(ParseError {
                    message: format!("Unexpected character: {}", value),
                    span,
                });
            }
            _ => panic!("Unknown token kind: {}", kind_str),
        }
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        value: "".to_string(),
        span: Span {
            start_line: line_num,
            start_col: 0,
            end_line: line_num,
            end_col: 0,
        },
    });
    (tokens, errors)
}

/// A recursive-descent parser for Qualtran-L1 code.
pub struct QualtranL1Parser {
    tokens: Vec<Token>,
    pos: usize,
    pub errors: Vec<ParseError>,
}

impl QualtranL1Parser {
    pub fn new(tokens: Vec<Token>, initial_errors: Vec<ParseError>) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: initial_errors,
        }
    }

    /// Look at the next token without consuming it.
    fn peek(&self) -> &Token {
        if self.pos >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1] // Return EOF
        } else {
            &self.tokens[self.pos]
        }
    }

    fn check(&self, tt: &TokenKind) -> bool {
        &self.peek().kind == tt
    }

    fn advance(&mut self) -> &Token {
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
    }

    /// Try to consume a token of type `tt`.
    ///
    /// If successful, returns `Some(token)`.
    /// If not, pushes an error to `self.errors` and returns `None`.
    ///
    /// Does NOT recover automatically (caller logic).
    /// Useful when working with `Option` or manual failure handling.
    fn consume_opt(&mut self, tt: TokenKind, err_msg: &str) -> Option<&Token> {
        if self.check(&tt) {
            Some(self.advance())
        } else {
            let tok = self.peek();
            self.errors.push(ParseError {
                message: format!("{} (found {:?})", err_msg, tok.kind),
                span: tok.span,
            });
            None
        }
    }

    /// Try to consume a token of type `tt`.
    ///
    /// If successful, returns `Ok(token)`.
    /// If not, pushes an error to `self.errors` and returns `Err(())`.
    ///
    /// This is a wrapper around `consume_opt` designed for ergonomic usage with the `?` operator.
    fn consume(&mut self, tt: TokenKind, err_msg: &str) -> Result<&Token, ()> {
        self.consume_opt(tt, err_msg).ok_or(())
    }

    /// Recover from a parse error by skipping tokens until we find a likely synchronization point.
    ///
    /// This is used to stabilize the parser state after an error so we can continue parsing
    /// the rest of the file and report multiple errors at once.
    ///
    /// We keep consuming tokens until we see one of the `tokens` in the provided list
    /// or we hit `EOF`.
    fn synchronize(&mut self, tokens: &[TokenKind]) {
        while !self.check(&TokenKind::Eof) {
            if tokens.contains(&self.peek().kind) {
                return;
            }
            self.advance();
        }
    }

    /// Helper method for processing a delimited list of items with optional trailing comma.
    ///
    /// Specifically, we support lists like `[a, b, c]` and `[a, b, c,]`.
    ///
    /// This should be called *after* processing a list item. If empty lists are allowed,
    /// they must be handled as a special case before processing list items.
    fn advance_to_next_list_item(
        &mut self,
        rbrack: TokenKind,
        delim: TokenKind,
        consume_rbrack: bool,
        list_name: &str,
    ) -> bool {
        if self.check(&delim) {
            self.advance();
        } else {
            if !self.check(&rbrack) {
                let tok = self.peek();
                self.errors.push(ParseError {
                    message: format!(
                        "Extraneous elements in {} at {:?} (expected {:?} or {:?})",
                        list_name, tok.kind, delim, rbrack
                    ),
                    span: tok.span,
                });
            }
        }

        if self.check(&rbrack) {
            if consume_rbrack {
                self.advance();
            }
            return true;
        }
        false
    }

    // --- Helper to merge spans ---
    fn mk_span(&self, start: Span, end: Span) -> Span {
        Span {
            start_line: start.start_line,
            start_col: start.start_col,
            end_line: end.end_line,
            end_col: end.end_col,
        }
    }

    // --- Parsing Methods ---

    /// Parse an optional annotation `@ cvalue`.
    ///
    /// Annotations are parsed but not stored at this time.
    fn parse_annotation(&mut self) -> Option<CValueNode> {
        if self.check(&TokenKind::At) {
            self.advance(); // @
            match self.parse_cvalue() {
                Ok(v) => Some(v),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    pub fn parse_module(&mut self) -> L1Module {
        if self.check(&TokenKind::Eof) {
            return L1Module { qdefs: vec![] };
        }
        let qdefs = self.parse_qdefs();
        let _ = self.consume(TokenKind::Eof, "Expected EOF");
        L1Module { qdefs }
    }

    fn parse_qdefs(&mut self) -> Vec<QDefNode> {
        let mut qdefs = Vec::new();
        loop {
            if self.check(&TokenKind::Eof) {
                break;
            }
            // Recovery point: if we are not at qdef/extern, skip until we are.
            if !self.check(&TokenKind::Name) {
                // If we are completely lost, advance one token
                let tok = self.peek();
                self.errors.push(ParseError {
                    message: format!("Expected top-level item, found {:?}", tok.kind),
                    span: tok.span,
                });
                self.advance();
                // Try to sync to next qdef
                self.synchronize(&[TokenKind::Name]); // Synchronize to next NAME which might be qdef
                continue;
            }

            // Check value without consuming yet to decide
            let val = &self.peek().value;
            if val == "qdef" {
                self.advance(); // consume "qdef"
                if let Some(qdef) = self.parse_qdef(false) {
                    qdefs.push(qdef);
                } else {
                    self.synchronize(&[TokenKind::Name, TokenKind::Eof]); // Sync to next possible qdef
                }
            } else if val == "extern" {
                self.advance(); // consume "extern"
                if let Some(_) = self.consume_opt(TokenKind::Name, "Expected 'extern qdef'") {
                    if self.tokens[self.pos - 1].value != "qdef" {
                        self.errors.push(ParseError {
                            message: "Expected 'extern qdef'".to_string(),
                            span: self.tokens[self.pos - 1].span,
                        });
                    }
                    if let Some(qdef) = self.parse_qdef(true) {
                        qdefs.push(qdef);
                    } else {
                        self.synchronize(&[TokenKind::Name, TokenKind::Eof]);
                    }
                }
            } else if val == "qcast" {
                self.advance(); // consume "qcast"
                if let Some(qcast) = self.parse_qcast() {
                    qdefs.push(qcast);
                } else {
                    self.synchronize(&[TokenKind::Name, TokenKind::Eof]);
                }
            } else {
                let tok = self.peek();
                self.errors.push(ParseError {
                    message: format!("Unexpected token at top level: {}", val),
                    span: tok.span,
                });
                self.advance();
                self.synchronize(&[TokenKind::Name, TokenKind::Eof]);
            }
        }
        qdefs
    }

    /// Parse a `qcast` definition.
    ///
    /// A qcast consists of a bloq key and a quantum signature.
    /// It has no `from` clause and no body.
    fn parse_qcast(&mut self) -> Option<QDefNode> {
        // 'qcast' is already consumed by the caller.
        let start_span = self.tokens[self.pos - 1].span;
        let bloq_key = self.parse_bloq_key().ok()?;
        let qsignature = self.parse_qdef_signature().ok()?;
        let end_span = self.tokens[self.pos - 1].span;
        Some(QDefNode::Cast(QCastNode {
            bloq_key,
            qsignature,
            span: self.mk_span(start_span, end_span),
        }))
    }

    fn parse_qdef(&mut self, extern_: bool) -> Option<QDefNode> {
        // 'qdef' (and 'extern') are already consumed by the caller.
        let start_span = self.tokens[self.pos - 1].span;

        let bloq_key = self.parse_bloq_key().ok()?;

        let cobject_from = if self.check(&TokenKind::Name) && self.peek().value == "from" {
            Some(self.parse_qdef_from().ok()?)
        } else {
            None
        };

        let qsignature = self.parse_qdef_signature().ok()?;

        if extern_ {
            let co = match cobject_from {
                Some(c) => c,
                None => {
                    self.errors.push(ParseError {
                        message: "extern qdef must have a 'from' clause".to_string(),
                        span: start_span, // use start_span of qdef/extern
                    });
                    return None;
                }
            };
            let end_span = if !qsignature.is_empty() {
                qsignature.last().unwrap().span
            } else {
                // Should use RBRACK from signature
                // But we don't have access to it easily here.
                // Let's assume current pos-1 is RBRACK
                self.tokens[self.pos - 1].span
            };

            Some(QDefNode::Extern(QDefExternNode {
                bloq_key,
                qsignature,
                cobject_from: co,
                span: self.mk_span(start_span, end_span),
            }))
        } else {
            let body = self.parse_qdef_body().ok()?;
            // Body ends with RCURLY. pos-1 should be RCURLY
            let end_span = self.tokens[self.pos - 1].span;

            Some(QDefNode::Impl(QDefImplNode {
                bloq_key,
                qsignature,
                body,
                cobject_from,
                span: self.mk_span(start_span, end_span),
            }))
        }
    }

    /// Parse the `from` clause of a qdef.
    ///
    /// Example:
    ///     qdef bloq_key(arg=0) from qualtran.bloqs.BloqCls(x, param=y) [...] {...}
    ///                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
    fn parse_qdef_from(&mut self) -> Result<CObjectNode, ()> {
        self.consume(TokenKind::Name, "qdef from must start with 'from'")?;
        if self.tokens[self.pos - 1].value != "from" {
            self.errors.push(ParseError {
                message: format!(
                    "qdef from must start with 'from', not {}",
                    self.tokens[self.pos - 1].value
                ),
                span: self.tokens[self.pos - 1].span,
            });
            return Err(());
        }
        self.parse_ccall()
    }

    /// Parse the quantum signature of a qdef.
    ///
    /// Example:
    ///     qdef bloq_key(arg=0) from xyz [qvar: QDType(k=v)[2, 4]] {...}
    ///                                   ^^^^^^^^^^^^^^^^^^^^^^^^^
    fn parse_qdef_signature(&mut self) -> Result<Vec<QSignatureEntry>, ()> {
        self.consume(TokenKind::LBrack, "quantum signature must start with [")?;
        if self.check(&TokenKind::RBrack) {
            self.advance();
            return Ok(vec![]);
        }

        let mut qsig = Vec::new();
        loop {
            qsig.push(self.parse_qsig_entry()?);
            // If we fail inside qsig_entry, we propagate error.
            if self.advance_to_next_list_item(
                TokenKind::RBrack,
                TokenKind::Comma,
                true,
                "quantum signature",
            ) {
                return Ok(qsig);
            }
        }
    }

    /// A `qvar: QDType(k=v)[2, 4]` entry in the quantum signature.
    ///
    /// The quantum signature encodes both quantum inputs and outputs. A simple entry
    /// `qvar: t` indicates a THRU-register of type `t`. For output-only (RIGHT) and
    /// input-only (LEFT) entries, the signature is `qvar: t -> |` and `qvar: | -> t`,
    /// respectively. For casting registers, the syntax is `qvar: t1 -> t2`.
    fn parse_qsig_entry(&mut self) -> Result<QSignatureEntry, ()> {
        let name_tok = self
            .consume(
                TokenKind::Name,
                "invalid identifier for quantum signature entry.",
            )?
            .clone();
        let start_span = name_tok.span;
        self.consume(
            TokenKind::Colon,
            "missing colon-delimited datatype in quantum signature",
        )?;

        // We check for output-only | -> t
        // or input-only t
        // or input-output t -> t
        // or input-output t -> |

        if self.check(&TokenKind::Pipe) {
            self.advance(); // |
            self.consume(
                TokenKind::RArrow,
                "output-only datatypes must be formulated | -> t",
            )?;
            let dtype = self.parse_qsig_dtype()?;
            let _annotation = self.parse_annotation();
            let end_span = dtype.span;
            return Ok(QSignatureEntry {
                name: name_tok.value,
                dtype: SignatureDType::Pair((None, Some(dtype))),
                span: self.mk_span(start_span, end_span),
            });
        }

        let t1 = self.parse_qsig_dtype()?;
        if self.check(&TokenKind::RArrow) {
            self.advance(); // ->
            if self.check(&TokenKind::Pipe) {
                let pipe_span = self.advance().span; // |
                let _annotation = self.parse_annotation();
                return Ok(QSignatureEntry {
                    name: name_tok.value,
                    dtype: SignatureDType::Pair((Some(t1), None)),
                    span: self.mk_span(start_span, pipe_span),
                });
            }
            let t2 = self.parse_qsig_dtype()?;
            let _annotation = self.parse_annotation();
            let end_span = t2.span;
            return Ok(QSignatureEntry {
                name: name_tok.value,
                dtype: SignatureDType::Pair((Some(t1), Some(t2))),
                span: self.mk_span(start_span, end_span),
            });
        }

        let _annotation = self.parse_annotation();
        let end_span = t1.span;
        Ok(QSignatureEntry {
            name: name_tok.value,
            dtype: SignatureDType::Single(t1),
            span: self.mk_span(start_span, end_span),
        })
    }

    /// `QDType(k=v)[2, 4]`
    fn parse_qsig_dtype(&mut self) -> Result<QDTypeNode, ()> {
        let cls = self.parse_ccall()?;
        let cls_span = cls.span;
        let mut end_span = cls_span;
        let shape = if self.check(&TokenKind::LBrack) {
            let s = self.parse_shape_list()?;
            // End span is RBRACK which was consumed in parse_shape_list
            end_span = self.tokens[self.pos - 1].span;
            Some(s)
        } else {
            None
        };
        Ok(QDTypeNode {
            dtype: cls,
            shape,
            span: self.mk_span(cls_span, end_span),
        })
    }

    /// `[1, 2, 3]`
    fn parse_shape_list(&mut self) -> Result<Vec<i64>, ()> {
        self.consume(TokenKind::LBrack, "datatype shapes must begin with '['")?;
        if self.check(&TokenKind::RBrack) {
            self.advance();
            return Ok(vec![]);
        }
        let mut items = Vec::new();
        loop {
            items.push(self.parse_int_literal("datatype shape list")?);
            if self.advance_to_next_list_item(
                TokenKind::RBrack,
                TokenKind::Comma,
                true,
                "datatype shape",
            ) {
                return Ok(items);
            }
        }
    }

    /// Curly-brace delimited sequence of statements.
    fn parse_qdef_body(&mut self) -> Result<Vec<StatementNode>, ()> {
        self.consume(TokenKind::LCurly, "qdef body must start with {")?;
        if self.check(&TokenKind::RCurly) {
            self.advance();
            return Ok(vec![]);
        }
        let mut statements = Vec::new();
        loop {
            if let Ok(stmt) = self.parse_statement() {
                statements.push(stmt);
            } else {
                // Let's try to find next statement or close brace.
                self.synchronize(&[TokenKind::RCurly, TokenKind::Name]);
                if self.check(&TokenKind::Eof) {
                    return Err(());
                }
                // if sync found name but it was "return", next loop will parse it.
                // if sync found RCURLY, next check handles it.
            }
            if self.check(&TokenKind::RCurly) {
                self.advance();
                return Ok(statements);
            }
        }
    }

    fn parse_statement(&mut self) -> Result<StatementNode, ()> {
        if self.check(&TokenKind::Name) && self.peek().value == "return" {
            return Ok(StatementNode::Return(self.parse_return_statement()?));
        }

        let lvalues = self.parse_lvalues()?;
        self.consume(TokenKind::Equals, "Assignment operator '=' expected")?;
        let bloq_key = self.parse_bloq_key()?;
        let _annotation = self.parse_annotation();

        let start_span = self.peek().span;

        if self.check(&TokenKind::LBrack) {
            let qargs = self.parse_qargs()?;
            let end_span = self.tokens[self.pos - 1].span;
            Ok(StatementNode::Call(QCallNode {
                bloq_key,
                lvalues,
                qargs,
                span: self.mk_span(start_span, end_span),
            }))
        } else {
            if lvalues.len() != 1 {
                self.errors.push(ParseError {
                    message:
                        "Syntax error: during alias assignment, only one lvalue may be specified"
                            .to_string(),
                    span: start_span,
                });
                return Err(());
            }
            // For alias, end span is the bloq_key (which was a String).
            // We need to know where bloq_key ended.
            // `parse_bloq_key` calls `parse_ccall`.
            // `parse_ccall` consumes tokens.
            // So `self.tokens[self.pos-1].span` is the end.
            let end_span = self.tokens[self.pos - 1].span;

            Ok(StatementNode::Alias(AliasAssignmentNode {
                alias: lvalues[0].clone(),
                bloq_key,
                span: self.mk_span(start_span, end_span),
            }))
        }
    }

    fn parse_return_statement(&mut self) -> Result<QReturnNode, ()> {
        let ret_tok = self
            .consume(TokenKind::Name, "return statement must start with 'return'")?
            .clone();
        if ret_tok.value != "return" {
            self.errors.push(ParseError {
                message: "Expected 'return'".to_string(),
                span: ret_tok.span,
            });
            return Err(());
        }
        let qargs = self.parse_qargs()?;
        let end_span = self.tokens[self.pos - 1].span;
        Ok(QReturnNode {
            ret_mapping: qargs,
            span: self.mk_span(ret_tok.span, end_span),
        })
    }

    fn parse_lvalues(&mut self) -> Result<Vec<String>, ()> {
        if self.check(&TokenKind::Pipe) {
            self.advance();
            return Ok(vec![]);
        }

        let mut lvalues = Vec::new();
        loop {
            lvalues.push(
                self.consume(TokenKind::Name, "Expected identifier for lvalue")?
                    .value
                    .clone(),
            );
            let _annotation = self.parse_annotation();
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                if !self.check(&TokenKind::Equals) {
                    let tok = self.peek();
                    self.errors.push(ParseError {
                        message: "Extraneous elements in lvalues".to_string(),
                        span: tok.span,
                    });
                    // recover?
                    return Err(());
                }
            }
            if self.check(&TokenKind::Equals) {
                return Ok(lvalues);
            }
        }
    }

    /// `[ctrl=[qvar[0], qvar[1]], target=trg]`
    fn parse_qargs(&mut self) -> Result<Vec<QArgNode>, ()> {
        self.consume(TokenKind::LBrack, "qargs must start with [")?;
        if self.check(&TokenKind::RBrack) {
            self.advance();
            return Ok(vec![]);
        }
        let mut args = Vec::new();
        loop {
            args.push(self.parse_qarg()?);
            if self.advance_to_next_list_item(TokenKind::RBrack, TokenKind::Comma, true, "qargs") {
                return Ok(args);
            }
        }
    }

    /// `ctrl=[qvar[0], qvar[1]]`
    fn parse_qarg(&mut self) -> Result<QArgNode, ()> {
        let key_tok = self.consume(TokenKind::Name, "invalid qarg key")?.clone();

        // Support array indexing on key e.g. reg[0]=...
        let key = if self.check(&TokenKind::LBrack) {
            self.advance();
            let idx = self.parse_int_literal("qarg key index")?;
            self.consume(
                TokenKind::RBrack,
                "missing closing bracket for qarg key index",
            )?;
            format!("{}[{}]", key_tok.value, idx)
        } else {
            key_tok.value.clone()
        };

        self.consume(TokenKind::Equals, "invalid qargs k=v specification")?;
        let value = if self.check(&TokenKind::LBrack) {
            NestedQArgValue::List(self.parse_qarg_value_list()?)
        } else {
            NestedQArgValue::Leaf(self.parse_qarg_value()?)
        };
        let _annotation = self.parse_annotation();
        let end_span = self.tokens[self.pos - 1].span;
        Ok(QArgNode {
            key,
            value,
            span: self.mk_span(key_tok.span, end_span),
        })
    }

    /// `[qvar[0], qvar[1]]`.
    ///
    /// Lists can be arbitrarily nested.
    fn parse_qarg_value_list(&mut self) -> Result<Vec<NestedQArgValue>, ()> {
        // consumes LBRACK and RBRACK
        self.consume(TokenKind::LBrack, "qarg value list must start with [")?;
        let mut list = Vec::new();
        loop {
            if self.check(&TokenKind::LBrack) {
                list.push(NestedQArgValue::List(self.parse_qarg_value_list()?));
            } else {
                list.push(NestedQArgValue::Leaf(self.parse_qarg_value()?));
            }
            if self.advance_to_next_list_item(
                TokenKind::RBrack,
                TokenKind::Comma,
                true,
                "qarg_value_list",
            ) {
                return Ok(list);
            }
        }
    }

    /// The value in a `target=trg` qargs assignment.
    ///
    /// Can be `trg` or include an optional index `qvar[5,6]`.
    fn parse_qarg_value(&mut self) -> Result<QArgValueNode, ()> {
        let name_tok = self
            .consume(TokenKind::Name, "qarg value must start with ident")?
            .clone();
        let idx = if self.check(&TokenKind::LBrack) {
            self.advance();
            let mut indices = Vec::new();
            loop {
                indices.push(self.parse_int_literal("qarg value index")?);
                if self.advance_to_next_list_item(
                    TokenKind::RBrack,
                    TokenKind::Comma,
                    true,
                    "qarg value index",
                ) {
                    break indices;
                }
            }
        } else {
            vec![]
        };
        let end_span = self.tokens[self.pos - 1].span;
        Ok(QArgValueNode {
            name: name_tok.value,
            idx,
            span: self.mk_span(name_tok.span, end_span),
        })
    }

    fn parse_bloq_key(&mut self) -> Result<String, ()> {
        Ok(self.parse_ccall()?.to_string())
    }

    /// Parse a classical call `FuncName(arg, k=v, k2=v2)`.
    ///
    /// The `()`-list of arguments is optional. `FuncName` will also parse.
    fn parse_ccall(&mut self) -> Result<CObjectNode, ()> {
        let name_parts = self.parse_qualified_identifier()?;
        let name = name_parts.join(".");

        // A simple name consumes 1 token. A-dot-B consumes 3 tokens.

        let consumed = name_parts.len() * 2 - 1;
        let start_span = self.tokens[self.pos - consumed].span;

        if !self.check(&TokenKind::LParen) {
            return Ok(CObjectNode {
                name,
                cargs: vec![],
                span: start_span,
            }); // Single name object
        }
        self.advance(); // LPAREN
        if self.check(&TokenKind::RParen) {
            let rp = self.advance(); // RPAREN
            let rp_span = rp.span;
            return Ok(CObjectNode {
                name,
                cargs: vec![],
                span: self.mk_span(start_span, rp_span),
            });
        }

        let mut cargs = Vec::new();
        loop {
            cargs.push(self.parse_carg()?);
            if self.advance_to_next_list_item(
                TokenKind::RParen,
                TokenKind::Comma,
                true,
                "classical args",
            ) {
                let end_span = self.tokens[self.pos - 1].span;
                return Ok(CObjectNode {
                    name,
                    cargs,
                    span: self.mk_span(start_span, end_span),
                });
            }
        }
    }

    /// Parse a dot-separated identifier.
    fn parse_qualified_identifier(&mut self) -> Result<Vec<String>, ()> {
        let mut parts = vec![self
            .consume(TokenKind::Name, "Expected identifier")?
            .value
            .clone()];
        while self.check(&TokenKind::Dot) {
            self.advance();
            parts.push(
                self.consume(TokenKind::Name, "Expected identifier")?
                    .value
                    .clone(),
            );
        }
        Ok(parts)
    }

    fn parse_carg(&mut self) -> Result<CArgNode, ()> {
        let start_span = self.peek().span;
        if self.check(&TokenKind::Name) {
            // checking ahead for equals
            if self.pos + 1 < self.tokens.len()
                && self.tokens[self.pos + 1].kind == TokenKind::Equals
            {
                let key = self.advance().value.clone(); // NAME
                self.advance(); // EQUALS
                let value = self.parse_cvalue()?;
                let end_span = value.span();
                return Ok(CArgNode {
                    key: Some(key),
                    value,
                    span: self.mk_span(start_span, end_span),
                });
            }
        }
        let value = self.parse_cvalue()?;
        let end_span = value.span();
        Ok(CArgNode {
            key: None,
            value,
            span: end_span,
        }) // if no key, span is value's span?
    }

    fn parse_int_literal(&mut self, err_ctx: &str) -> Result<i64, ()> {
        let tok = self
            .consume(
                TokenKind::Number,
                &format!("expected integer in {}", err_ctx),
            )?
            .clone();
        match tok.value.parse::<i64>() {
            Ok(i) => Ok(i),
            Err(_) => {
                self.errors.push(ParseError {
                    message: format!("Failed to parse integer {}", tok.value),
                    span: tok.span,
                });
                Err(())
            }
        }
    }

    /// Parse a value (literal or ccall).
    /// Parse a value (literal or ccall).
    fn parse_cvalue(&mut self) -> Result<CValueNode, ()> {
        if self.check(&TokenKind::LParen) {
            let start_span = self.advance().span;
            if self.check(&TokenKind::RParen) {
                let end_span = self.advance().span;
                return Ok(CValueNode::Tuple(TupleNode {
                    items: vec![],
                    span: self.mk_span(start_span, end_span),
                }));
            }
            let mut vals = Vec::new();
            loop {
                vals.push(self.parse_cvalue()?);
                if self.advance_to_next_list_item(
                    TokenKind::RParen,
                    TokenKind::Comma,
                    true,
                    "cvalue list",
                ) {
                    let end_span = self.tokens[self.pos - 1].span;
                    return Ok(CValueNode::Tuple(TupleNode {
                        items: vals,
                        span: self.mk_span(start_span, end_span),
                    }));
                }
            }
        } else if self.check(&TokenKind::Number) {
            let tok = self.advance();
            if tok.value.contains('.') || tok.value.contains('e') {
                Ok(CValueNode::Literal(LiteralNode {
                    value: LiteralVal::Float(
                        tok.value
                            .parse::<f64>()
                            .expect("float parse verified by regex"),
                    ),
                    span: tok.span,
                }))
            } else {
                Ok(CValueNode::Literal(LiteralNode {
                    value: LiteralVal::Int(
                        tok.value
                            .parse::<i64>()
                            .expect("int parse verified by regex"),
                    ),
                    span: tok.span,
                }))
            }
        } else if self.check(&TokenKind::String) {
            let tok = self.advance();
            Ok(CValueNode::Literal(LiteralNode {
                value: LiteralVal::String(tok.value.clone()),
                span: tok.span,
            }))
        } else if self.check(&TokenKind::Name) {
            Ok(CValueNode::CObject(self.parse_ccall()?))
        } else {
            let tok = self.peek();
            self.errors.push(ParseError {
                message: format!("Unexpected token {:?} when parsing cvalue", tok.kind),
                span: tok.span,
            });
            Err(())
        }
    }
}

// Updated wrapper to return errors
pub fn parse_l1_module(code: &str) -> (L1Module, Vec<ParseError>) {
    let (tokens, lex_errors) = tokenize(code);
    let mut parser = QualtranL1Parser::new(tokens, lex_errors);
    let module = parser.parse_module();
    (module, parser.errors)
}

/// Parse a bloq_key string through the tokenizer and parser pipeline,
/// returning its canonical string representation.
///
/// This normalizes user-supplied bloq_key strings: e.g. `"Negate()"` becomes
/// `"Negate"`, while `"AddK(k=1)"` stays `"AddK(k=1)"`.
///
/// Returns `Err` if the string cannot be parsed as a valid bloq_key.
pub fn canonicalize_bloq_key(s: &str) -> Result<String, String> {
    let (tokens, lex_errors) = tokenize(s);
    if !lex_errors.is_empty() {
        return Err(format!(
            "Tokenization errors in bloq_key '{}': {:?}",
            s, lex_errors
        ));
    }
    let mut parser = QualtranL1Parser::new(tokens, vec![]);
    parser
        .parse_bloq_key()
        .map_err(|()| format!("Failed to parse bloq_key '{}': {:?}", s, parser.errors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{CValueNode, LiteralVal, NestedQArgValue, QDefNode, StatementNode};

    #[test]
    fn test_tokenize_complex() {
        let code = "reg: QAny(5) -> QBit[5] | var = 'str'";
        let (tokens, errors) = tokenize(code);
        assert!(errors.is_empty());
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind.clone()).collect();
        // remove EOF from comparison if needed or include it
        assert_eq!(kinds.last().unwrap(), &TokenKind::Eof);

        let values: Vec<&str> = tokens.iter().map(|t| t.value.as_str()).collect();
        assert_eq!(values[6], "->");
        assert_eq!(values[11], "|");
        assert_eq!(values[14], "str");
    }

    #[test]
    fn test_parse_literal() {
        let (tokens, _) = tokenize("10");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let val = p.parse_cvalue().unwrap();
        if let CValueNode::Literal(l) = val {
            if let LiteralVal::Int(i) = l.value {
                assert_eq!(i, 10);
            } else {
                panic!("Expected Int");
            }
        } else {
            panic!("Expected Literal");
        }

        let (tokens, _) = tokenize("3.14");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let val = p.parse_cvalue().unwrap();
        if let CValueNode::Literal(l) = val {
            if let LiteralVal::Float(f) = l.value {
                assert!((f - 3.14).abs() < 1e-6);
            } else {
                panic!("Expected Float");
            }
        } else {
            panic!("Expected Literal");
        }

        let (tokens, _) = tokenize("'hello'");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let val = p.parse_cvalue().unwrap();
        if let CValueNode::Literal(l) = val {
            if let LiteralVal::String(s) = l.value {
                assert_eq!(s, "hello");
            } else {
                panic!("Expected String");
            }
        } else {
            panic!("Expected Literal");
        }
    }

    #[test]
    fn test_parse_cobject() {
        // qualtran.QAny(5)
        let (tokens, _) = tokenize("qualtran.QAny(5)");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let obj = p.parse_ccall().unwrap();
        assert_eq!(obj.name, "qualtran.QAny");
        assert_eq!(obj.cargs.len(), 1);
        // cargs[0] should be 5
        let carg = &obj.cargs[0];
        assert!(carg.key.is_none());
    }

    #[test]
    fn test_parse_qarg() {
        // reg
        let (tokens, _) = tokenize("reg");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let qarg = p.parse_qarg_value().unwrap();
        assert_eq!(qarg.name, "reg");
        assert!(qarg.idx.is_empty());

        // reg[0, 1]
        let (tokens, _) = tokenize("reg[0, 1]");
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let qarg = p.parse_qarg_value().unwrap();
        assert_eq!(qarg.name, "reg");
        assert_eq!(qarg.idx, vec![0, 1]);
    }

    #[test]
    fn test_parse_statement_alias() {
        // reg = Split(5)
        let code = "reg = Split(5)";
        let (tokens, _) = tokenize(code);
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let stmt = p.parse_statement().unwrap();
        if let StatementNode::Alias(node) = stmt {
            assert_eq!(node.alias, "reg");
            assert_eq!(node.bloq_key, "Split(5)"); // string rep
        } else {
            panic!("Expected AliasAssignmentNode");
        }
    }

    #[test]
    fn test_parse_statement_call() {
        // c, x, y = CSwap [ctrl=c, x=x, y=y]
        let code = "c, x, y = CSwap [ctrl=c, x=x, y=y]";
        let (tokens, _) = tokenize(code);
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let stmt = p.parse_statement().unwrap();
        if let StatementNode::Call(node) = stmt {
            assert_eq!(node.lvalues, vec!["c", "x", "y"]);
            assert_eq!(node.bloq_key, "CSwap");
            assert_eq!(node.qargs.len(), 3);
            assert_eq!(node.qargs[0].key, "ctrl");
        } else {
            panic!("Expected CallNode");
        }
    }

    #[test]
    fn test_parse_nested_qargs() {
        // [reg=[x, y, z]]
        let code = "[reg=[x, y, z]]";
        let (tokens, _) = tokenize(code);
        let mut p = QualtranL1Parser::new(tokens, vec![]);
        let qargs = p.parse_qargs().unwrap();
        assert_eq!(qargs.len(), 1);
        assert_eq!(qargs[0].key, "reg");
        if let NestedQArgValue::List(list) = &qargs[0].value {
            assert_eq!(list.len(), 3);
        } else {
            panic!("Expected List");
        }
    }

    // Add more tests with updated signatures...
    #[test]
    fn test_parse_qdef_impl() {
        let code = r#"
        qdef TestBlock [] {
            x = Alloc() []
            return [x=x]
        }
        "#;
        let (p, errors) = parse_l1_module(code);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert_eq!(p.qdefs.len(), 1);
        if let QDefNode::Impl(node) = &p.qdefs[0] {
            assert_eq!(node.bloq_key, "TestBlock");
            assert_eq!(node.body.len(), 2);
        } else {
            panic!("Expected Impl");
        }
    }

    #[test]
    fn test_error_recovery() {
        let code = r#"
        qdef Good [] {}
        qdef Bad [ !!! ] {} # Syntax error
        qdef GoodAgain [] {}
        "#;
        let (p, errors) = parse_l1_module(code);
        // Should have errors
        assert!(!errors.is_empty());
        // Should have recovered 2 good blocks + 1 bad block (bad block parsed as valid structure after skipping '!')
        assert_eq!(
            p.qdefs.len(),
            3,
            "Should recover 3 blocks (Bad block is structurally valid after lexer skips '!')"
        );
        assert_eq!(p.qdefs[0].bloq_key(), "Good");
        assert_eq!(p.qdefs[1].bloq_key(), "Bad");
        assert_eq!(p.qdefs[2].bloq_key(), "GoodAgain");
    }
}

// Helpers for test
impl QDefNode {
    fn bloq_key(&self) -> &str {
        match self {
            QDefNode::Impl(n) => &n.bloq_key,
            QDefNode::Extern(n) => &n.bloq_key,
            QDefNode::Cast(n) => &n.bloq_key,
        }
    }
}
