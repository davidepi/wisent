use super::lr::ShiftReduceAction;
use super::ParseNode;
use crate::error::ParseError;
use crate::grammar::{Grammar, Tree};
use crate::lexer::{DfaSimulator, MultiDfa, Token};
use crate::parser::{LLParsingTable, LRParsingTable, ParserSymbol};
#[cfg(feature = "trace_parser")]
use std::fmt::Write;
use std::io::{Bytes, Read};

/// Trait implementing a pull parser.
///
/// In a pull parser the parser automatically pull tokens from the scanner,
/// until parsing is finished.
pub trait PullParser {
    /// Generates a parsing tree using a pull parser.
    ///
    /// Requires the scanner/lexer and the input to be processed.
    fn parse<R: Read>(self, dfa: &MultiDfa, input: Bytes<R>)
        -> Result<Tree<ParseNode>, ParseError>;
}

/// Trait implementing a push parser.
///
/// In a push parser the parser stops whenever a new token from the scanner is
/// required. Providing the next token resumes the parsing.
pub trait PushParser {
    /// Generates a parsing tree using a push parser.
    ///
    /// This method will return None until parse is complete. Each invocation
    /// expects the next token retrieved from the lexer/scanner. If input
    /// reaches Eof, None should be passed as token.
    fn parse(&mut self, token: Option<Token>) -> Option<Result<Tree<ParseNode>, ParseError>>;
}

/// Struct used to perform LL(1) parsing.
///
/// Stores intermediate data required by a LL(1) parser.
pub struct LLParser {
    /// stack for the table-driven parser
    /// contains ID of the parent (in the parse tree) and current symbol being
    /// processed
    stack: Vec<(u32, ParserSymbol)>,
    /// table for the table-driven parser
    table: LLParsingTable,
    /// indexed nodes of the tree being built. Childless, children will be
    /// assigned at the end. First value is the parent_id, second value is
    /// the actual tree.
    built_nodes: Vec<(u32, Tree<ParseNode>)>,
    /// Stores a grammar, to provide better debug informations.
    #[cfg(feature = "trace_parser")]
    grammar: Option<Grammar>,
}

impl LLParser {
    /// Creates a new LL(1) parser with the given parsing table.
    ///
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::{LLGrammar, LLParser};
    /// let g = "sum: num PLUS num;
    ///          num: INT | REAL;
    ///          INT: [0-9]+;
    ///          REAL: [0-9]+ '.' [0-9]+;
    ///          PLUS: '+';";
    /// let grammar = Grammar::parse_bootstrap(g).unwrap();
    /// let grammar_ll = LLGrammar::try_from(&grammar).unwrap();
    /// let table = grammar_ll.parsing_table().unwrap();
    /// let parser = LLParser::new(table);
    /// ```
    pub fn new(table: LLParsingTable) -> Self {
        let stack = vec![
            (u32::MAX, ParserSymbol::Terminal(table.eof_val())),
            (u32::MAX, ParserSymbol::NonTerminal(table.start())),
        ];
        Self {
            stack,
            table,
            built_nodes: Vec::new(),
            #[cfg(feature = "trace_parser")]
            grammar: None,
        }
    }

    /// Supplies the production names to provide better debugging.
    ///
    /// Normally, when logging, the production index would be used. If this
    /// method is called (in debug mode), the parser stores the grammar in
    /// order to provide the production name when logging.
    ///
    /// This method requires the feature **trace-parser** to be active.
    pub fn verbose_debug(&mut self, _grammar: Grammar) {
        #[cfg(feature = "trace_parser")]
        {
            self.grammar = Some(_grammar);
        }
    }
}

/// Struct used to perform LR parsing.
///
/// Stores intermediate data required by a LR parser.
pub struct LRParser {
    stack: Vec<u32>,
    tokens: Vec<Token>,
    tree: Vec<Tree<ParseNode>>,
    table: LRParsingTable,
    #[cfg(feature = "trace_parser")]
    grammar: Option<Grammar>,
}

impl LRParser {
    /// Creates a new LR Parser with the given parsing table.
    ///
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::{LRGrammar, LRParser};
    /// let g = "sum: num PLUS num;
    ///          num: INT | REAL;
    ///          INT: [0-9]+;
    ///          REAL: [0-9]+ '.' [0-9]+;
    ///          PLUS: '+';";
    /// let grammar = Grammar::parse_bootstrap(g).unwrap();
    /// let grammar_lr = LRGrammar::try_from(&grammar).unwrap();
    /// let table = grammar_lr.parsing_table().unwrap();
    /// let parser = LRParser::new(table);
    /// ```
    pub fn new(table: LRParsingTable) -> Self {
        Self {
            stack: vec![0],
            tokens: Vec::new(),
            tree: Vec::new(),
            table,
            #[cfg(feature = "trace_parser")]
            grammar: None,
        }
    }

    /// Supplies the production names to provide better debugging.
    ///
    /// Normally, when logging, the production index would be used. If this
    /// method is called (in debug mode), the parser stores the grammar in
    /// order to provide the production name when logging.
    ///
    /// This method requires the feature **trace-parser** to be active.
    pub fn verbose_debug(&mut self, _grammar: Grammar) {
        #[cfg(feature = "trace_parser")]
        {
            self.grammar = Some(_grammar);
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
    debug_print_token_ll(status, token);
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
                    debug_print_accept_ll(status, nt, y);
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

/// Print the token received by the parser. prints nothing without trace_parser
/// feature.
fn debug_print_token_ll(_parser: &LLParser, _tok: Option<Token>) {
    #[cfg(feature = "trace_parser")]
    {
        let token_name = if let Some(grammar) = &_parser.grammar {
            _tok.map(|x| {
                grammar
                    .iter_term()
                    .nth(x.production as usize)
                    .unwrap()
                    .head
                    .clone()
            })
            .unwrap_or_else(|| "EOF".to_string())
        } else {
            _tok.map(|x| format!("{}", x.production))
                .unwrap_or_else(|| "EOF".to_string())
        };
        println!("Next token is {token_name}");
    }
}

/// prints the production accepted by the parser. Prints nothing without
/// trace_parser feature.
fn debug_print_accept_ll(_parser: &LLParser, _lhs: u32, _rhs: &[ParserSymbol]) {
    #[cfg(feature = "trace_parser")]
    {
        if let Some(grammar) = &_parser.grammar {
            let mut string = format!(
                "{}:",
                &grammar.iter_nonterm().nth(_lhs as usize).unwrap().head
            );
            for symbol in _rhs {
                match symbol {
                    ParserSymbol::Terminal(t) => {
                        let name = &grammar
                            .iter_term()
                            .nth(*t as usize)
                            .map(|x| x.head.clone())
                            .unwrap_or_else(|| "EOF".to_string());
                        write!(string, " {name}")
                    }
                    ParserSymbol::NonTerminal(nt) => {
                        let name = &grammar.iter_nonterm().nth(*nt as usize).unwrap().head;
                        write!(string, " {name}")
                    }
                    ParserSymbol::Empty => write!(string, " ε"),
                }
                .unwrap();
            }
            println!("Accepting rule {string}");
        } else {
            println!("Accepting rule {_lhs}");
        }
    }
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

/// Algorithm for LR parsing, as listed in the dragon book.
/// Unlike the dragon book one, must be called multiple times, to read the new
/// token until EOF is reached.
fn parse_lr(
    status: &mut LRParser,
    token: Option<Token>,
) -> Option<Result<Tree<ParseNode>, ParseError>> {
    let token_val = token
        .map(|t| t.production)
        .unwrap_or_else(|| status.table.eof_val());
    debug_print_token_lr(status, token);
    loop {
        let s = *status.stack.last().unwrap();
        match status.table.action[s as usize][token_val as usize] {
            ShiftReduceAction::Shift(i) => {
                status.stack.push(i);
                status.tokens.push(token.unwrap());
                debug_print_shift_lr(i);
                return None;
            }
            ShiftReduceAction::Reduce((rule, prod)) => {
                let reduced = &status.table.nonterminals[rule as usize][prod as usize];
                let mut children = Vec::with_capacity(reduced.len());
                for symbol in reduced.iter().rev() {
                    match symbol {
                        ParserSymbol::Terminal(_) => children.push(Tree::new_leaf(
                            ParseNode::Terminal(status.tokens.pop().unwrap()),
                        )),
                        ParserSymbol::NonTerminal(_) => children.push(status.tree.pop().unwrap()),
                        ParserSymbol::Empty => (),
                    }
                    status.stack.pop();
                }
                children.reverse();
                let node = Tree::new_node(ParseNode::ParserRule(rule), children);
                status.tree.push(node);
                let t = *status.stack.last().unwrap();
                let gt = status.table.goto[t as usize][rule as usize].unwrap();
                debug_print_reduce_lr(status, rule, reduced, gt);
                status.stack.push(gt);
            }
            ShiftReduceAction::Accept => return Some(Ok(status.tree.pop().unwrap())),
            ShiftReduceAction::Error => {
                return Some(Err(ParseError::ParsingError {
                    message: "halt".to_string(),
                }))
            }
        }
    }
}

/// Print the token received by the parser. prints nothing without trace_parser
/// feature.
fn debug_print_token_lr(_parser: &LRParser, _tok: Option<Token>) {
    #[cfg(feature = "trace_parser")]
    {
        let token_name = if let Some(grammar) = &_parser.grammar {
            _tok.map(|x| {
                grammar
                    .iter_term()
                    .nth(x.production as usize)
                    .unwrap()
                    .head
                    .clone()
            })
            .unwrap_or_else(|| "EOF".to_string())
        } else {
            _tok.map(|x| format!("{}", x.production))
                .unwrap_or_else(|| "EOF".to_string())
        };
        println!("Next token is {}", token_name);
    }
}

fn debug_print_shift_lr(_state: u32) {
    #[cfg(feature = "trace_parser")]
    {
        println!("Shifting state {_state}")
    }
}

/// prints the production accepted by the parser. Prints nothing without
/// trace_parser feature.
fn debug_print_reduce_lr(_parser: &LRParser, _lhs: u32, _rhs: &[ParserSymbol], _gt: u32) {
    #[cfg(feature = "trace_parser")]
    {
        if let Some(grammar) = &_parser.grammar {
            let mut string = format!(
                "{}:",
                &grammar.iter_nonterm().nth(_lhs as usize).unwrap().head
            );
            for symbol in _rhs {
                match symbol {
                    ParserSymbol::Terminal(t) => {
                        let name = &grammar
                            .iter_term()
                            .nth(*t as usize)
                            .map(|x| x.head.clone())
                            .unwrap_or_else(|| "EOF".to_string());
                        write!(string, " {}", name)
                    }
                    ParserSymbol::NonTerminal(nt) => {
                        let name = &grammar.iter_nonterm().nth(*nt as usize).unwrap().head;
                        write!(string, " {}", name)
                    }
                    ParserSymbol::Empty => write!(string, " ε"),
                }
                .unwrap();
            }
            println!("Reducing rule {string}. Going to state {_gt}");
        } else {
            println!("Reducing rule {_lhs}. Going to state {_gt}");
        }
    }
}

impl PushParser for LRParser {
    fn parse(&mut self, token: Option<Token>) -> Option<Result<Tree<ParseNode>, ParseError>> {
        parse_lr(self, token)
    }
}

impl PullParser for LRParser {
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
            parse_tree = parse_lr(&mut self, token);
        }
        parse_tree.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{LLParser, LRParser, PullParser};
    use crate::lexer::MultiDfa;
    use crate::parser::tests::{grammar_428, grammar_440};
    use crate::parser::{LLGrammar, LRGrammar};
    use std::io::{BufReader, Read};

    #[test]
    fn accepting_ll1() {
        let g = grammar_428();
        let dfa = MultiDfa::new(&g);
        let ll1_table = LLGrammar::try_from(&g).unwrap().parsing_table().unwrap();
        let input = "3+1*2";
        let reader = BufReader::new(input.as_bytes());
        let simulator = LLParser::new(ll1_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_ok());
    }

    #[test]
    fn rejecting_ll1() {
        let g = grammar_428();
        let dfa = MultiDfa::new(&g);
        let ll1_table = LLGrammar::try_from(&g).unwrap().parsing_table().unwrap();
        let input = "(3*(1+2)";
        let reader = BufReader::new(input.as_bytes());
        let simulator = LLParser::new(ll1_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_err());
    }

    #[test]
    fn accepting_slr() {
        let g = grammar_440();
        let dfa = MultiDfa::new(&g);
        let slr_table = LRGrammar::try_from(&g)
            .unwrap()
            .slr_parsing_table()
            .unwrap();
        let input = "3*2+1";
        let reader = BufReader::new(input.as_bytes());
        let mut simulator = LRParser::new(slr_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_ok());
    }

    #[test]
    fn rejecting_slr() {
        let g = grammar_440();
        let dfa = MultiDfa::new(&g);
        let slr_table = LRGrammar::try_from(&g)
            .unwrap()
            .slr_parsing_table()
            .unwrap();
        let input = "(3*(2+1)";
        let reader = BufReader::new(input.as_bytes());
        let mut simulator = LRParser::new(slr_table);
        assert!(simulator.parse(&dfa, reader.bytes()).is_err());
    }
}
