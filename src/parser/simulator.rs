use super::ParseNode;
use crate::error::ParseError;
use crate::grammar::Tree;
use crate::lexer::{DfaSimulator, MultiDfa, Token};
use crate::parser::{LL1ParsingTable, ParserSymbol};
use std::io::{Bytes, Read};

/// Trait implementing a pull parser.
///
/// In a pull parser the parser automatically pull tokens from the scanner, until parsing is
/// finished.
pub trait PullParser {
    /// Generates a parsing tree using a pull parser.
    ///
    /// Requires the scanner/lexer and the input to be processed.
    fn parse<R: Read>(self, dfa: &MultiDfa, input: Bytes<R>)
        -> Result<Tree<ParseNode>, ParseError>;
}

/// Trait implementing a push parser.
///
/// In a push parser the parser stops whenever a new token from the scanner is required. Providing
/// the next token resumes the parsing.
pub trait PushParser {
    /// Generates a parsing tree using a push parser.
    ///
    /// This method will return None until parse is complete. Each invocation expects the next
    /// token retrieved from the lexer/scanner. If input reaches Eof, None should be passed as
    /// token.
    fn parse(&mut self, token: Option<Token>) -> Option<Result<Tree<ParseNode>, ParseError>>;
}

/// Struct used to perform LL(1) parsing.
///
/// Stores intermediate data required by a LL(1) parser.
pub struct LLParser {
    /// stck for the table-driven parser
    /// contains ID of the parent (in the parse tree) and current symbol being processed
    stack: Vec<(u32, ParserSymbol)>,
    /// table for the table-driven parser
    table: LL1ParsingTable,
    /// indexed nodes of the tree being built. Childless, children will be assigned at the end.
    /// First value is the parent_id, second value is the actual tree.
    built_nodes: Vec<(u32, Tree<ParseNode>)>,
}

impl LLParser {
    /// Creates a new LL(1) parser with the given parsing table.
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::{LL1ParsingTable, LLParser};
    /// let g = "sum: num PLUS num;
    ///          num: INT | REAL;
    ///          INT: [0-9]+;
    ///          REAL: [0-9]+ '.' [0-9]+;
    ///          PLUS: '+';";
    /// let grammar = Grammar::parse_bootstrap(g).unwrap();
    /// let table = LL1ParsingTable::new(&grammar, 0).unwrap();
    /// let parser = LLParser::new(table);
    /// ```
    pub fn new(table: LL1ParsingTable) -> Self {
        let stack = vec![
            (u32::MAX, ParserSymbol::Terminal(table.eof_val())),
            (u32::MAX, ParserSymbol::NonTerminal(table.start())),
        ];
        Self {
            stack,
            table,
            built_nodes: Vec::new(),
        }
    }
}

/// Algorithm for LL1 parsing, as listed in the dragon book
fn parse_ll1(
    status: &mut LLParser,
    token: Option<Token>,
) -> Option<Result<Tree<ParseNode>, ParseError>> {
    let token_val = token
        .map(|t| t.production)
        .unwrap_or_else(|| status.table.eof_val());
    while let Some((parent_id, top_stack)) = status.stack.pop() {
        match top_stack {
            ParserSymbol::Terminal(t) => {
                if token_val == t {
                    if token_val == status.table.eof_val() {
                        // parsing over. No need to record the last terminal.
                        return Some(Ok(build_parse_tree(status)));
                    } else {
                        status.built_nodes.push((
                            parent_id,
                            Tree::new_leaf(ParseNode::Terminal(token.unwrap())),
                        ));
                        // ask more tokens
                        return None;
                    }
                } else {
                    return Some(Err(ParseError::ParsingError {
                        message: "wrong token".to_string(),
                    }));
                }
            }
            ParserSymbol::NonTerminal(nt) => {
                if let Some(y) = status.table.entry(nt, token_val) {
                    let current_id = status.built_nodes.len() as u32;
                    status
                        .built_nodes
                        .push((parent_id, Tree::new_leaf(ParseNode::ParserRule(nt))));
                    status
                        .stack
                        .extend(y.iter().rev().map(|&x| (current_id, x)));
                } else {
                    return Some(Err(ParseError::ParsingError {
                        message: "halt".to_string(),
                    }));
                }
            }
            _ => (),
        }
    }
    panic!("Unreachable")
}

/// Transforms a flat list of nodes into a proper tree.
fn build_parse_tree(parser: &mut LLParser) -> Tree<ParseNode> {
    while parser.built_nodes.len() > 1 {
        let (parent_id, child) = parser.built_nodes.pop().unwrap();
        // don't bother adding empty leaves or epsilon nodes
        if child.children_len() > 0 || matches!(child.value(), ParseNode::Terminal(_)) {
            parser.built_nodes[parent_id as usize].1.add_child(child);
        }
    }
    parser.built_nodes.pop().unwrap().1
}

impl PushParser for LLParser {
    fn parse(&mut self, token: Option<Token>) -> Option<Result<Tree<ParseNode>, ParseError>> {
        parse_ll1(self, token)
    }
}

impl PullParser for LLParser {
    fn parse<R: Read>(
        mut self,
        dfa: &MultiDfa,
        input: Bytes<R>,
    ) -> Result<Tree<ParseNode>, ParseError> {
        let mut scanner = DfaSimulator::new(dfa, input);
        let mut token;
        let mut parse_tree = None;
        while parse_tree.is_none() {
            token = scanner.next_token()?;
            parse_tree = parse_ll1(&mut self, token);
        }
        parse_tree.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{LLParser, PullParser};
    use crate::lexer::MultiDfa;
    use crate::parser::tests::grammar_428;
    use crate::parser::LL1ParsingTable;
    use std::io::{BufReader, Read};

    #[test]
    fn accepting_ll1() {
        let g = grammar_428();
        let dfa = MultiDfa::new(&g);
        let ll1_table = LL1ParsingTable::new(&g, 0).unwrap();
        let input = "(3*(1+2))";
        let reader = BufReader::new(input.as_bytes());
        let simulator = LLParser::new(ll1_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_ok());
    }

    #[test]
    fn rejecting_ll1() {
        let g = grammar_428();
        let dfa = MultiDfa::new(&g);
        let ll1_table = LL1ParsingTable::new(&g, 0).unwrap();
        let input = "(3*(1+2)";
        let reader = BufReader::new(input.as_bytes());
        let simulator = LLParser::new(ll1_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_err());
    }
}
