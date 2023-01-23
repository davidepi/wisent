use crate::error::ParseError;
use crate::grammar::{
    Grammar, LexerOp, LexerProduction, LexerRuleElement, ParserProduction, ParserRuleElement, Tree,
};
use maplit::btreeset;
use std::collections::BTreeSet;

/// Manually written DFA and recursive descent parser, so it
/// can be used to parse simple BNF grammars without depending on the crate itself.

/// Possible accepted production of the lexer.
#[allow(clippy::upper_case_acronyms, non_camel_case_types)]
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Acc {
    /// A literal between two single quotes (i.e. 'literal' or 'int').
    /// Can also represents a SINGLE unicode character written as 'U+XXXXXX'
    LITERAL,
    /// A charset [abc], can not contain ranges or square brackets.
    CHARSET,
    /// A whitespace [ \r\n\t]+
    WS,
    /// An identifier in the form [a-z][a-zA-Z0-9_]*
    NAME_TERM,
    /// An identifier in the form [A-Z][a-zA-Z0-9_]*
    NAME_NONTERM,
    /// `:`
    ASSIGN,
    /// `;`
    SEMI,
    /// `|`
    BAR,
    /// `(`
    LPAR,
    /// `)`
    RPAR,
    /// `*`
    KLEENE,
    /// `?`
    QM,
    /// `+`
    PL,
    /// `~`
    NOT,
    /// `.`
    ANY,
    /// ..
    TWODOTS,
}

/// Token accepted by the lexer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Token<'a> {
    /// Accepted production.
    tp: Acc,
    /// Accepted text.
    val: &'a str,
}

// Equivalence classes for the lexer
//  0: not in language
//  1: space \r \n \t
//  2: uppercase_letter
//  3: lowercase_letter
//  4: ':'
//  5: ';'
//  6: '|'
//  7: '['
//  8: ']'
//  9: '('
// 10: ')'
// 11: '*'
// 12: '+'
// 13: '?'
// 14: '~'
// 15: `.`
// 16: '\'';
// 17: '"';
// 18: '_';
// 19: digit
// 20: any other printable symbol
const CHAR_CLASS: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 20, 17, 20, 20, 20, 20, 16, 9, 10, 11, 12, 20, 20, 15, 20, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 4, 5, 20, 20, 20, 13, 20, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 7, 20, 8, 20, 18, 20, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 20, 6, 20, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Starting state for the lexer
const DFA_START: u8 = 16;

/// Sinking state for the lexer
const DFA_SINK: u8 = 20;

/// Transition table for the lexer. Starts from 0, sink in 20.
const TRANSITION: [[u8; 21]; 21] = [
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 3, 3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3, 3, 20],
    [20, 20, 4, 4, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 4, 4, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 15, 20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    [20, 2, 4, 3, 5, 6, 7, 17, 20, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 20, 20],
    [20, 17, 17, 17, 17, 17, 17, 17, 1, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
    [20, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 18, 18, 18, 18],
    [20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 19, 19, 19],
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
];

/// Accepting states for the lexer.
const ACCEPT: [Option<Acc>; 21] = [
    Some(Acc::LITERAL),
    Some(Acc::CHARSET),
    Some(Acc::WS),
    Some(Acc::NAME_NONTERM),
    Some(Acc::NAME_TERM),
    Some(Acc::ASSIGN),
    Some(Acc::SEMI),
    Some(Acc::BAR),
    Some(Acc::LPAR),
    Some(Acc::RPAR),
    Some(Acc::KLEENE),
    Some(Acc::PL),
    Some(Acc::QM),
    Some(Acc::NOT),
    Some(Acc::ANY),
    Some(Acc::TWODOTS),
    None,
    None,
    None,
    None,
    None,
];

/// Recursive descent implementation used to parse a simil-ANTLR grammar for bootstrapping other
/// grammars.
/// Grammar starts with [`parse_rulelist`]
pub(crate) fn bootstrap_grammar(content: &str) -> Result<Grammar, ParseError> {
    let mut position = 0;
    let buf = content.as_bytes();
    let mut lookahead = None;
    let (terminals, non_terminals) = parse_rulelist(buf, &mut position, &mut lookahead)?;
    Ok(Grammar {
        terminals: vec![terminals],
        non_terminals,
        ..Default::default()
    })
}

// rulelist: [rule_nonterm | rule_term]*
fn parse_rulelist<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead: &mut Option<Token<'a>>,
) -> Result<(Vec<LexerProduction>, Vec<ParserProduction>), ParseError> {
    let mut term = Vec::new();
    let mut nonterm = Vec::new();
    while let Ok(la) = get_lookahead(buffer, position, lookahead, "") {
        match la.tp {
            Acc::NAME_NONTERM => {
                let rule = parse_rule_nonterm(buffer, position, lookahead)?;
                nonterm.push(rule);
            }
            Acc::NAME_TERM => {
                let rule = parse_rule_term(buffer, position, lookahead)?;
                term.push(rule);
            }
            _ => {
                return Err(ParseError::SyntaxError {
                    message: format!("Unexpected {:?}, expecting `;` ", la.tp),
                });
            }
        }
    }
    Ok((term, nonterm))
}

// rule_nonterm: NAME_NONTERM ASSIGN expression_nonterm SEMI
fn parse_rule_nonterm<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<ParserProduction, ParseError> {
    let lookahead = get_lookahead(buffer, position, lookahead_cache, "NAME_NONTERM")?;
    if lookahead.tp == Acc::NAME_NONTERM {
        consume_lookahead(lookahead_cache);
        let assign = get_lookahead(buffer, position, lookahead_cache, "`=`")?;
        if assign.tp == Acc::ASSIGN {
            consume_lookahead(lookahead_cache);
            let expression = parse_expression_nonterm(buffer, position, lookahead_cache)?;
            let semi = get_lookahead(buffer, position, lookahead_cache, "`;`")?;
            if semi.tp == Acc::SEMI {
                consume_lookahead(lookahead_cache);
                Ok(ParserProduction {
                    head: lookahead.val[..lookahead.val.len()].to_string(),
                    body: expression,
                })
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unexpected {:?}, expecting `;` ", semi.tp),
                })
            }
        } else {
            Err(ParseError::SyntaxError {
                message: format!("Unexpected {:?}, expecting `:` ", assign.tp),
            })
        }
    } else {
        Err(ParseError::SyntaxError {
            message: format!("Unexpected {:?}, expecting identifier", lookahead),
        })
    }
}

// rule_term: NAME_TERM ASSIGN expression_term SEMI
fn parse_rule_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<LexerProduction, ParseError> {
    let lookahead = get_lookahead(buffer, position, lookahead_cache, "NAME_TERM")?;
    if lookahead.tp == Acc::NAME_TERM {
        consume_lookahead(lookahead_cache);
        let assign = get_lookahead(buffer, position, lookahead_cache, "`:`")?;
        if assign.tp == Acc::ASSIGN {
            consume_lookahead(lookahead_cache);
            let expression = parse_expression_term(buffer, position, lookahead_cache)?;
            let semi = get_lookahead(buffer, position, lookahead_cache, "`;`")?;
            if semi.tp == Acc::SEMI {
                consume_lookahead(lookahead_cache);
                Ok(LexerProduction {
                    head: lookahead.val[..lookahead.val.len()].to_string(),
                    body: expression,
                    actions: BTreeSet::new(),
                })
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unexpected {:?}, expecting `;` ", semi.tp),
                })
            }
        } else {
            Err(ParseError::SyntaxError {
                message: format!("Unexpected {:?}, expecting `:` ", assign.tp),
            })
        }
    } else {
        Err(ParseError::SyntaxError {
            message: format!("Unexpected {:?}, expecting identifier", lookahead),
        })
    }
}

// expression_nonterm: list_nonterm [BAR list_nonterm]*
fn parse_expression_nonterm<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let mut lists = vec![parse_list_nonterm(buffer, position, lookahead_cache)?];
    while let Ok(lookahead) = get_lookahead(buffer, position, lookahead_cache, "") {
        match lookahead.tp {
            Acc::BAR => {
                consume_lookahead(lookahead_cache);
                lists.push(parse_expression_nonterm(buffer, position, lookahead_cache)?);
            }
            _ => break,
        }
    }
    let tree = if lists.len() > 1 {
        Tree::new_node(ParserRuleElement::Operation(LexerOp::Or), lists)
    } else {
        assert!(lists.len() == 1);
        lists.pop().unwrap()
    };
    Ok(tree)
}

// expression_term: list_term [BAR list_term]*
fn parse_expression_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<LexerRuleElement>, ParseError> {
    let mut lists = vec![parse_list_term(buffer, position, lookahead_cache)?];
    while let Ok(lookahead) = get_lookahead(buffer, position, lookahead_cache, "") {
        match lookahead.tp {
            Acc::BAR => {
                consume_lookahead(lookahead_cache);
                lists.push(parse_expression_term(buffer, position, lookahead_cache)?);
            }
            _ => break,
        }
    }
    let tree = if lists.len() > 1 {
        Tree::new_node(LexerRuleElement::Operation(LexerOp::Or), lists)
    } else {
        assert!(lists.len() == 1);
        lists.pop().unwrap()
    };
    Ok(tree)
}

//list_nonterm: term_nonterm*
fn parse_list_nonterm<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let mut terms = Vec::new();
    while let Ok(term) = parse_term_nonterm(buffer, position, lookahead_cache) {
        terms.push(term);
    }
    let tree = if terms.is_empty() {
        Tree::new_leaf(ParserRuleElement::Empty)
    } else if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        Tree::new_node(ParserRuleElement::Operation(LexerOp::And), terms)
    };
    Ok(tree)
}

//list_term: grouped_term+
fn parse_list_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<LexerRuleElement>, ParseError> {
    let mut terms = vec![parse_grouped_term(buffer, position, lookahead_cache)?];
    while let Ok(term) = parse_grouped_term(buffer, position, lookahead_cache) {
        terms.push(term);
    }
    let tree = if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        terms.reverse();
        let l = terms.pop().unwrap();
        let r = terms.pop().unwrap();
        let mut tree = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![l, r]);
        while let Some(n) = terms.pop() {
            tree = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![tree, n]);
        }
        tree
    };
    Ok(tree)
}

//grouped_term: (NOT? LPAR expression_term RPAR ((KLEENE|QM|PL) QM?)?) | (NOT? term_term ((KLEENE|QM|PL) QM?)?)
fn parse_grouped_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<LexerRuleElement>, ParseError> {
    let lookahead = get_lookahead(
        buffer,
        position,
        lookahead_cache,
        "NOT or LPAR or CHARSET or LITERAL or ANY",
    )?;
    let negated = if lookahead.tp == Acc::NOT {
        consume_lookahead(lookahead_cache);
        true
    } else {
        false
    };
    let lookahead = get_lookahead(
        buffer,
        position,
        lookahead_cache,
        "LPAR or CHARSET or LITERAL or ANY",
    )?;
    let mut main = if lookahead.tp == Acc::LPAR {
        consume_lookahead(lookahead_cache);
        let tree = parse_expression_term(buffer, position, lookahead_cache)?;
        if get_lookahead(buffer, position, lookahead_cache, "RPAR")?.tp != Acc::RPAR {
            return Err(ParseError::SyntaxError {
                message: format!("Unexpected {:?}, expecting RPAR", lookahead),
            });
        } else {
            consume_lookahead(lookahead_cache);
        }
        tree
    } else {
        parse_term_term(buffer, position, lookahead_cache)?
    };
    if negated {
        main = Tree::new_node(LexerRuleElement::Operation(LexerOp::Not), vec![main]);
    }
    // at least a semicolon is expected after this production, hence why the hint for SEMI
    let lookahead = get_lookahead(buffer, position, lookahead_cache, "SEMI")?;
    let search_nongreedy = match lookahead.tp {
        Acc::KLEENE => {
            consume_lookahead(lookahead_cache);
            main = Tree::new_node(LexerRuleElement::Operation(LexerOp::Kleene), vec![main]);
            true
        }
        Acc::QM => {
            consume_lookahead(lookahead_cache);
            main = Tree::new_node(LexerRuleElement::Operation(LexerOp::Qm), vec![main]);
            true
        }
        Acc::PL => {
            consume_lookahead(lookahead_cache);
            main = Tree::new_node(LexerRuleElement::Operation(LexerOp::Pl), vec![main]);
            true
        }
        _ => false,
    };
    if search_nongreedy {
        let ng = get_lookahead(buffer, position, lookahead_cache, "SEMI")?;
        if ng.tp == Acc::QM {
            consume_lookahead(lookahead_cache);
            main = Tree::new_node(LexerRuleElement::Operation(LexerOp::Qm), vec![main]);
        }
    }
    Ok(main)
}

// term: NAME_TERM | NAME_NONTERM
fn parse_term_nonterm<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let lookahead = get_lookahead(
        buffer,
        position,
        lookahead_cache,
        "NAME_TERM or NAME_NONTERM",
    )?;
    match lookahead.tp {
        Acc::NAME_TERM => {
            consume_lookahead(lookahead_cache);
            Ok(Tree::new_leaf(ParserRuleElement::Terminal(
                lookahead.val[..lookahead.val.len()].to_string(),
            )))
        }
        Acc::NAME_NONTERM => {
            consume_lookahead(lookahead_cache);
            Ok(Tree::new_leaf(ParserRuleElement::NonTerminal(
                lookahead.val[..lookahead.val.len()].to_string(),
            )))
        }
        _ => Err(ParseError::SyntaxError {
            message: format!(
                "Unexpected {:?}, expecting NAME_TERM or NAME_NONTERM",
                lookahead
            ),
        }),
    }
}

// term: CHARSET | LITERAL [TWODOTS LITERAL] | ANY
fn parse_term_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<LexerRuleElement>, ParseError> {
    let lookahead = get_lookahead(
        buffer,
        position,
        lookahead_cache,
        "CHARSET or LITERAL or ANY",
    )?;
    match lookahead.tp {
        Acc::CHARSET => {
            consume_lookahead(lookahead_cache);
            let set = lookahead.val[1..lookahead.val.len() - 1]
                .chars()
                .collect::<BTreeSet<_>>();
            if set.is_empty() {
                Err(ParseError::SyntaxError {
                    message: "Charset cannot be empty".to_string(),
                })
            } else {
                Ok(Tree::new_leaf(LexerRuleElement::CharSet(set)))
            }
        }
        Acc::ANY => {
            consume_lookahead(lookahead_cache);
            Ok(Tree::new_leaf(LexerRuleElement::AnyValue))
        }
        Acc::LITERAL => {
            consume_lookahead(lookahead_cache);
            let literal_len = lookahead.val.len() - 2;
            if literal_len == 1 || lookahead.val[1..].starts_with("U+") {
                let single = literal_single(lookahead)?;
                if let Ok(twodots) = get_lookahead(buffer, position, lookahead_cache, "") {
                    if twodots.tp == Acc::TWODOTS {
                        consume_lookahead(lookahead_cache);
                        let literal2 = get_lookahead(buffer, position, lookahead_cache, "LITERAL")?;
                        if literal2.tp == Acc::LITERAL {
                            consume_lookahead(lookahead_cache);
                            let literal2_len = literal2.val.len() - 2;
                            if literal2_len == 1 || literal2.val[1..].starts_with("U+") {
                                let single2 = literal_single(literal2)?;
                                let mut set = BTreeSet::new();
                                for glyph in single..=single2 {
                                    set.insert(glyph);
                                }
                                Ok(Tree::new_leaf(LexerRuleElement::CharSet(set)))
                            } else {
                                Err(ParseError::SyntaxError {
                                    message: "The second literal in a range must be a single value"
                                        .to_string(),
                                })
                            }
                        } else {
                            Err(ParseError::SyntaxError {
                                message: format!("Unexpected {:?}, expecting LITERAL", lookahead),
                            })
                        }
                    } else {
                        Ok(Tree::new_leaf(LexerRuleElement::CharSet(
                            btreeset! {single},
                        )))
                    }
                } else {
                    Ok(Tree::new_leaf(LexerRuleElement::CharSet(
                        btreeset! {single},
                    )))
                }
            } else if literal_len > 0 {
                Ok(literal_concat(lookahead))
            } else {
                Err(ParseError::SyntaxError {
                    message: "Literal cannot be empty".to_string(),
                })
            }
        }
        _ => Err(ParseError::SyntaxError {
            message: format!(
                "Unexpected {:?}, expecting CHARSET or LITERAL or ANY",
                lookahead
            ),
        }),
    }
}

/// Parse a literal in the form 'a' or 'U+0061'.
/// Assuming input validation is done elsewhere.
fn literal_single(token: Token) -> Result<char, ParseError> {
    if token.val.len() == 3 {
        Ok(token.val[1..token.val.len() - 1].chars().next().unwrap())
    } else {
        let code = &token.val[3..token.val.len() - 1];
        if let Ok(codepoint) = u32::from_str_radix(code, 16) {
            match char::from_u32(codepoint) {
                Some(ch) => Ok(ch),
                None => Err(ParseError::SyntaxError {
                    message: format!(
                        "literal {} can not be converted to a valid unicode codepoint",
                        &token.val[1..token.val.len() - 1]
                    ),
                }),
            }
        } else {
            Err(ParseError::SyntaxError {
                message: format!(
                    "literal {} can not be converted to a valid codepoint",
                    &token.val[1..token.val.len() - 1]
                ),
            })
        }
    }
}

/// Parse a literal composed of multiple ASCII characters concatenated.
fn literal_concat(token: Token) -> Tree<LexerRuleElement> {
    let mut iter = token.val[1..token.val.len() - 1].chars();
    let left_set = iter.next().map(|c| btreeset! {c}).unwrap_or_default();
    let right_set = iter.next().map(|c| btreeset! {c}).unwrap_or_default();
    let left = Tree::new_leaf(LexerRuleElement::CharSet(left_set));
    let right = Tree::new_leaf(LexerRuleElement::CharSet(right_set));
    let mut root = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![left, right]);
    for c in iter {
        let right = Tree::new_leaf(LexerRuleElement::CharSet(btreeset! {c}));
        root = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![root, right]);
    }
    root
}

/// Returns the current lookahead. If None, pull a new one from the lexer. If EOF, return error.
fn get_lookahead<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
    expected: &str,
) -> Result<Token<'a>, ParseError> {
    if let Some(token) = lookahead_cache {
        Ok(*token)
    } else {
        match next_token(buffer, position) {
            Some(token) => {
                *lookahead_cache = Some(token);
                Ok(token)
            }
            None => Err(ParseError::SyntaxError {
                message: format!("Unexpected EOF, expecting {}", expected),
            }),
        }
    }
}

/// Sets the current lookahead to None
#[inline]
fn consume_lookahead(lookahead_cache: &mut Option<Token>) {
    *lookahead_cache = None;
}

/// Retrieves the next token from the lexer. Requires the entire input string (in ASCII) and the
/// position of the next character to be read (that will be updated accordingly).
///
/// Skips Whitespaces.
fn next_token<'a>(buffer: &'a [u8], position: &mut usize) -> Option<Token<'a>> {
    let mut state = DFA_START;
    let mut acc = None;
    let mut start_position = *position;
    let mut last_accepted_position = *position;
    loop {
        while state != DFA_SINK && *position < buffer.len() {
            // exits when in sink
            let byte = buffer[*position];
            *position += 1;
            let class = CHAR_CLASS[byte as usize];
            state = TRANSITION[state as usize][class as usize];
            let this_acc = ACCEPT[state as usize];
            if this_acc.is_some() {
                last_accepted_position = *position;
                acc = this_acc;
            }
        }
        if acc == Some(Acc::WS) {
            // skip whitespaces
            acc = None;
            *position = last_accepted_position;
            start_position = *position;
            state = DFA_START;
        } else {
            break;
        }
    }
    let token = acc.map(|acc| Token {
        tp: acc,
        val: std::str::from_utf8(&buffer[start_position..last_accepted_position])
            .expect("UTF-8 is not supported in the bootstrapping grammars"),
    });
    *position = last_accepted_position;
    token
}

#[cfg(test)]
mod tests {
    use super::{next_token, Acc, Token, CHAR_CLASS};
    use crate::grammar::bootstrap::bootstrap_grammar;
    use crate::grammar::Tree;
    use std::fmt::Write;

    /// encoded representation of a tree in form of string
    /// otherwise the formatted version takes a lot of space (macros too, given the tree generics)
    fn as_str<T: std::fmt::Display>(node: &Tree<T>) -> String {
        let mut string = String::new();
        let children = node.children().map(as_str).collect::<Vec<_>>();
        write!(&mut string, "{}", node.value()).unwrap();
        if !children.is_empty() {
            write!(&mut string, "({})", children.join(",")).unwrap();
        }
        string
    }

    #[test]
    fn char_class() {
        // spaces and newlines
        assert_eq!(CHAR_CLASS[u32::from(' ') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\t') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\r') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\n') as usize], 1);
        // uppercase letters
        for code in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 2)
        }
        // lowercase letters
        for code in "abcdefghijklmnopqrstuvwxyz".chars() {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 3)
        }
        //single symbols
        assert_eq!(CHAR_CLASS[u32::from(':') as usize], 4);
        assert_eq!(CHAR_CLASS[u32::from(';') as usize], 5);
        assert_eq!(CHAR_CLASS[u32::from('|') as usize], 6);
        assert_eq!(CHAR_CLASS[u32::from('[') as usize], 7);
        assert_eq!(CHAR_CLASS[u32::from(']') as usize], 8);
        assert_eq!(CHAR_CLASS[u32::from('(') as usize], 9);
        assert_eq!(CHAR_CLASS[u32::from(')') as usize], 10);
        assert_eq!(CHAR_CLASS[u32::from('*') as usize], 11);
        assert_eq!(CHAR_CLASS[u32::from('+') as usize], 12);
        assert_eq!(CHAR_CLASS[u32::from('?') as usize], 13);
        assert_eq!(CHAR_CLASS[u32::from('~') as usize], 14);
        assert_eq!(CHAR_CLASS[u32::from('.') as usize], 15);
        assert_eq!(CHAR_CLASS[u32::from('\'') as usize], 16);
        assert_eq!(CHAR_CLASS[u32::from('"') as usize], 17);
        assert_eq!(CHAR_CLASS[u32::from('_') as usize], 18);
        // digit
        for code in "0123456789".chars() {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 19)
        }
        // symbols
        let symbols = "!#$%&,/=<>^@\\{}-`";
        for code in symbols.chars() {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 20)
        }
    }

    #[test]
    fn transition() {
        let mut buf;
        let mut pos;
        let mut token;
        let test = [
            ("rUlE_0", Acc::NAME_NONTERM),
            ("RulE_0", Acc::NAME_TERM),
            (":", Acc::ASSIGN),
            (";", Acc::SEMI),
            ("|", Acc::BAR),
            ("(", Acc::LPAR),
            (")", Acc::RPAR),
            ("*", Acc::KLEENE),
            ("+", Acc::PL),
            ("?", Acc::QM),
            ("~", Acc::NOT),
            (".", Acc::ANY),
            ("..", Acc::TWODOTS),
            ("\"I'm a literal\"", Acc::LITERAL),
            ("'U+00D8'", Acc::LITERAL),
            ("'Double quote (\") allowed!'", Acc::LITERAL),
            ("[abc0123$_.~]", Acc::CHARSET),
        ];
        for (text, expected) in test {
            buf = text;
            pos = 0;
            token = Token {
                tp: expected,
                val: buf,
            };
            assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        }
        // skip spaces
        buf = " rule0";
        pos = 0;
        token = Token {
            tp: Acc::NAME_NONTERM,
            val: &buf[1..],
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
    }

    #[test]
    fn nonterm_ok() {
        let grammar = "rule0: rule1;";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_nonterm().next().unwrap().body;
        let expected = "NT[rule1]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn nonterm_missing_semi() {
        let grammar = "rule0: rule1";
        let parsed = bootstrap_grammar(grammar);
        assert!(parsed.is_err());
    }

    #[test]
    fn nonterm_call_term() {
        let grammar = "rule0: Rule1;";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_nonterm().next().unwrap().body;
        let expected = "T[Rule1]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_literal_ascii() {
        let grammar = "U: 'U';";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "[U]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_literal_unicode() {
        let grammar = "O_stroke: 'U+00d8';";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "[Ø]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_range_ascii() {
        let grammar = "AF: 'A'..'F';";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "[ABCDEF]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_range_unicode() {
        let grammar = "AF: 'U+20078'..'U+2007B';";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "[𠁸𠁹𠁺𠁻]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_literal_multiple() {
        let grammar = "Rule0: 'ab';";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "&([a],[b])";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_charset_ok() {
        let grammar = "Rule0: [abcde];";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_term().next().unwrap().body;
        let expected = "[abcde]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_charset_empty() {
        let grammar = "Epsilon: [];";
        let parsed = bootstrap_grammar(grammar);
        assert!(parsed.is_err());
    }

    #[test]
    fn term_missing_semi() {
        let grammar = "Rule0: 'literal'";
        let parsed = bootstrap_grammar(grammar);
        assert!(parsed.is_err());
    }

    #[test]
    fn nonterm_same_line() {
        let grammar = "rule0: Rule1; Rule1: \"ab\";";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_nonterm().next().unwrap().body;
        let expected = "T[Rule1]";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn nonterm_or() {
        let grammar = "rule0: rule1 | rule2 Rule3;";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_nonterm().next().unwrap().body;
        let expected = "|(NT[rule1],&(NT[rule2],T[Rule3]))";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn nonterm_empty() {
        let grammar = "rule0: Rule1
                            |
                            ;";
        let parsed = bootstrap_grammar(grammar).unwrap();
        let tree = &parsed.iter_nonterm().next().unwrap().body;
        let expected = "|(T[Rule1],ε)";
        assert_eq!(as_str(tree), expected);
    }

    #[test]
    fn term_correct_precedence_or_klenee() {
        let expr = "Rule: 'a'|'b'*'c';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "|([a],&(*([b]),[c]))";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_correct_precedence_klenee_par() {
        let expr = "Rule: 'a'*('b'|'c')*'d';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&(*([a]),*(|([b],[c]))),[d])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_correct_precedence_negation_par() {
        let expr = "Rule: ('a')~'b'('c')('d')'e';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&(&(&([a],~([b])),[c]),[d]),[e])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_correct_precedence_negation() {
        let expr = "Rule: 'a'~'b''c'('d');";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&(&([a],~([b])),[c]),[d])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_correct_precedence_qm_plus_klenee_par() {
        let expr = "Rule: 'a'?('b'*|'c'*)+'d';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&(?([a]),+(|(*([b]),*([c])))),[d])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_precedence_negation_klenee() {
        let expr = "Rule: ~'a'*;";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "*(~([a]))";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_precedence_negation_klenee_or() {
        let expr = "Rule: ~'a'*|'b';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "|(*(~([a])),[b])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_precedence_literal_negation_literal() {
        let expr = "Rule: 'a'~'a'*'a';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&([a],*(~([a]))),[a])";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn regression_disappearing_literal() {
        let expr = "Rule: 'a'*~'b'*;";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(*([a]),*(~([b])))";
        assert_eq!(as_str(prec_tree), expected);
    }

    #[test]
    fn term_precedence_nongreedy_negation() {
        let expr = "Rule: 'a'~'a'*?'a';";
        let grammar = bootstrap_grammar(expr).unwrap();
        let prec_tree = &grammar.iter_term().next().unwrap().body;
        let expected = "&(&([a],?(*(~([a])))),[a])";
        assert_eq!(as_str(prec_tree), expected);
    }
}
