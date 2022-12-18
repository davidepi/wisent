use crate::error::ParseError;
use crate::grammar::{
    Grammar, LexerOp, LexerProduction, LexerRuleElement, ParserProduction, ParserRuleElement, Tree,
};
use std::collections::{BTreeMap, BTreeSet};

/// Manually written DFA and recursive descent parser, so it
/// can be used to parse simple BNF grammars without depending on the crate itself.

/// Possible accepted production of the lexer.
#[allow(clippy::upper_case_acronyms, non_camel_case_types)]
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Acc {
    /// A literal "[^"]*" or '[^']*' (excluding non-representable symbols and non ASCII-chars).
    LITERAL,
    /// A whitespace [ \r\n\t]+
    WS,
    /// An identifier in the form [A-Za-z0-9_-]+ (yes they can start with numbers)
    RULE_NAME,
    /// `|`
    BAR,
    /// `=`
    ASSIGN,
    /// `;`
    SEMI,
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
// class 0: not in language
// 1: space \r \n \t
// 2: symbol (not ;=|)
// 3: letters, numbers, - and _
// 4: "
// 5: '
// 6: |
// 7: =
// 8: ;
const CHAR_CLASS: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 4, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 8, 2, 7, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 6, 2, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Transition table for the lexer. Starts from 0, sink in 9.
const TRANSITION: [[u8; 9]; 10] = [
    [9, 6, 9, 5, 7, 8, 1, 2, 3],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 5, 9, 9, 9, 9, 9],
    [9, 6, 9, 9, 9, 9, 9, 9, 9],
    [9, 7, 7, 7, 4, 7, 7, 7, 7],
    [9, 8, 8, 8, 8, 4, 8, 8, 8],
    [9, 9, 9, 9, 9, 9, 9, 9, 9],
];

/// Accepting states for the lexer.
const ACCEPT: [Option<Acc>; 10] = [
    None,
    Some(Acc::BAR),
    Some(Acc::ASSIGN),
    Some(Acc::SEMI),
    Some(Acc::LITERAL),
    Some(Acc::RULE_NAME),
    Some(Acc::WS),
    None,
    None,
    None,
];

/// Recursive descent implementation used to parse an EBNF grammar for bootstrapping other
/// grammars.
/// Grammar starts with [`parse_rulelist`]
pub(crate) fn bootstrap_ebnf(content: &str) -> Result<Grammar, ParseError> {
    let mut position = 0;
    let buf = content.as_bytes();
    let mut lookahead = None;
    let productions = parse_rulelist(buf, &mut position, &mut lookahead)?;
    let (terminals, non_terminals) = assign_terminal_names(productions);
    Ok(Grammar {
        terminals: vec![terminals],
        non_terminals,
        ..Default::default()
    })
}

/// Splits the productions in terminals and non-terminals.
fn assign_terminal_names(
    prods: Vec<ParserProduction>,
) -> (Vec<LexerProduction>, Vec<ParserProduction>) {
    let mut terminal_names = BTreeMap::default();
    let mut nonterms = Vec::with_capacity(prods.len());
    for mut prod in prods {
        replace_literal_rec(&mut prod.body, &mut terminal_names);
        nonterms.push(prod);
    }
    let mut terms = Vec::with_capacity(terminal_names.len());
    for (literal, term_id) in terminal_names {
        let letters = literal.chars().collect::<Vec<_>>();
        let tree = if letters.len() < 2 {
            let leaf = LexerRuleElement::CharSet(letters.into_iter().collect());
            Tree::new_leaf(leaf)
        } else {
            let mut letters = letters.into_iter();
            let left = Tree::new_leaf(LexerRuleElement::CharSet(
                [letters.next().unwrap()].into_iter().collect(),
            ));
            let right = Tree::new_leaf(LexerRuleElement::CharSet(
                [letters.next().unwrap()].into_iter().collect(),
            ));
            let mut root =
                Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![left, right]);
            for letter in letters {
                let right =
                    Tree::new_leaf(LexerRuleElement::CharSet([letter].into_iter().collect()));
                root = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![right]);
            }
            root
        };
        let head = format!("TERMINAL_{}", term_id);
        let term = LexerProduction {
            head,
            body: tree,
            actions: BTreeSet::new(),
        };
        terms.push(term);
    }
    (terms, nonterms)
}

/// Replaces literals 'this is a literal' with the name TERMINAL_idx where idx is a progressive
/// index. These will be used to generate the terminal productions.
fn replace_literal_rec(node: &mut Tree<ParserRuleElement>, names: &mut BTreeMap<String, u32>) {
    if let ParserRuleElement::Terminal(literal) = node.value_mut() {
        if let Some(val) = names.get(&*literal) {
            *literal = format!("TERMINAL_{}", *val);
        } else {
            let next_id = names.len() as u32;
            names.insert(literal.clone(), next_id);
            *literal = format!("TERMINAL_{}", next_id);
        }
    }
    node.children_mut()
        .for_each(|c| replace_literal_rec(c, names));
}

// rulelist: [rule]*
fn parse_rulelist<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead: &mut Option<Token<'a>>,
) -> Result<Vec<ParserProduction>, ParseError> {
    let mut rules = Vec::new();
    while get_lookahead(buffer, position, lookahead, "").is_ok() {
        // if there is a lookahead parse_rule MUST succeed, otherwise is a syntax error.
        // and not because for example there are no more lines to read.
        let rule = parse_rule(buffer, position, lookahead)?;
        rules.push(rule);
    }
    Ok(rules)
}

// rule: RULE_NAME ASSIGN expression SEMI
fn parse_rule<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<ParserProduction, ParseError> {
    let lookahead = get_lookahead(buffer, position, lookahead_cache, "RULE_NAME")?;
    if lookahead.tp == Acc::RULE_NAME {
        consume_lookahead(lookahead_cache);
        let assign = get_lookahead(buffer, position, lookahead_cache, "`=`")?;
        if assign.tp == Acc::ASSIGN {
            consume_lookahead(lookahead_cache);
            let expression = parse_expression(buffer, position, lookahead_cache)?;
            let semi = get_lookahead(buffer, position, lookahead_cache, "`;`")?;
            if semi.tp == Acc::SEMI {
                consume_lookahead(lookahead_cache);
                Ok(ParserProduction {
                    head: lookahead.val[1..lookahead.val.len() - 1].to_string(),
                    body: expression,
                })
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unexpected {:?}, expecting `;` ", semi.tp),
                })
            }
        } else {
            Err(ParseError::SyntaxError {
                message: format!("Unexpected {:?}, expecting `=` ", assign.tp),
            })
        }
    } else {
        Err(ParseError::SyntaxError {
            message: format!("Unexpected {:?}, expecting identifier", lookahead),
        })
    }
}

// expression: list [BAR list]+
fn parse_expression<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let mut lists = vec![parse_list(buffer, position, lookahead_cache)?];
    while let Ok(lookahead) = get_lookahead(buffer, position, lookahead_cache, "") {
        match lookahead.tp {
            Acc::BAR => {
                consume_lookahead(lookahead_cache);
                lists.push(parse_expression(buffer, position, lookahead_cache)?);
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

//list: term*
fn parse_list<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let mut terms = Vec::new();
    while let Ok(term) = parse_term(buffer, position, lookahead_cache) {
        terms.push(term);
    }
    let tree = if terms.is_empty() {
        Tree::new_leaf(ParserRuleElement::Empty)
    } else if terms.len() == 1 {
        assert!(terms.len() == 1);
        terms.pop().unwrap()
    } else {
        Tree::new_node(ParserRuleElement::Operation(LexerOp::And), terms)
    };
    Ok(tree)
}

// term: LITERAL | RULE_NAME
fn parse_term<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead_cache: &mut Option<Token<'a>>,
) -> Result<Tree<ParserRuleElement>, ParseError> {
    let lookahead = get_lookahead(buffer, position, lookahead_cache, "LITERAL or RULE_NAME")?;
    match lookahead.tp {
        Acc::LITERAL => {
            consume_lookahead(lookahead_cache);
            Ok(Tree::new_leaf(ParserRuleElement::Terminal(
                lookahead.val[1..lookahead.val.len() - 1].to_string(),
            )))
        }
        Acc::RULE_NAME => {
            consume_lookahead(lookahead_cache);
            Ok(Tree::new_leaf(ParserRuleElement::NonTerminal(
                lookahead.val[1..lookahead.val.len() - 1].to_string(),
            )))
        }
        _ => Err(ParseError::SyntaxError {
            message: format!(
                "Unexpected {:?}, expecting `'` identifier `'` or  `\"` identifier `\"` or `<` \
                identifier `>` ",
                lookahead
            ),
        }),
    }
}

/// Returns the current lookahead. If None, pull a new one from the lexer. If EOF, return error.
fn get_lookahead<'a>(
    buffer: &'a [u8],
    position: &mut usize,
    lookahead: &mut Option<Token<'a>>,
    expected: &str,
) -> Result<Token<'a>, ParseError> {
    if let Some(token) = lookahead {
        Ok(*token)
    } else {
        match next_token(buffer, position) {
            Some(token) => {
                *lookahead = Some(token);
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
fn consume_lookahead(lookahead: &mut Option<Token>) {
    *lookahead = None;
}

/// Retrieves the next token from the lexer. Requires the entire input string (in ASCII) and the
/// position of the next character to be read (that will be updated accordingly).
///
/// Skips Whitespaces.
fn next_token<'a>(buffer: &'a [u8], position: &mut usize) -> Option<Token<'a>> {
    let mut state = 0;
    let mut acc = None;
    let mut start_position = *position;
    let mut last_accepted_position = *position;
    loop {
        while state != 9 && *position < buffer.len() {
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
            state = 0;
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
    use super::{bootstrap_ebnf, next_token, Acc, Token, CHAR_CLASS};
    use crate::grammar::{ParserRuleElement, Tree};

    #[test]
    fn char_class() {
        // spaces and newlines
        assert_eq!(CHAR_CLASS[u32::from(' ') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\t') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\r') as usize], 1);
        assert_eq!(CHAR_CLASS[u32::from('\n') as usize], 1);
        // symbols
        let symbols = "!#$%&*+,./:<>?[]^{}~";
        for code in symbols.chars() {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 2)
        }
        // letters
        for code in ['-', '_']
            .into_iter()
            .chain('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
        {
            assert_eq!(CHAR_CLASS[u32::from(code) as usize], 3)
        }
        //single symbols
        assert_eq!(CHAR_CLASS[u32::from('"') as usize], 4);
        assert_eq!(CHAR_CLASS[u32::from('\'') as usize], 5);
        assert_eq!(CHAR_CLASS[u32::from('|') as usize], 6);
        assert_eq!(CHAR_CLASS[u32::from('=') as usize], 7);
        assert_eq!(CHAR_CLASS[u32::from(';') as usize], 8);
    }

    #[test]
    fn transition() {
        let mut pos = 0;
        let mut buf = "rule0";
        let mut token = Token {
            tp: Acc::RULE_NAME,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        buf = "|";
        pos = 0;
        token = Token {
            tp: Acc::BAR,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        buf = "=";
        pos = 0;
        token = Token {
            tp: Acc::ASSIGN,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        buf = ";";
        pos = 0;
        token = Token {
            tp: Acc::SEMI,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        buf = "\"I'm a <literal0>\"";
        pos = 0;
        token = Token {
            tp: Acc::LITERAL,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        buf = "'\"<literal0>\"'";
        pos = 0;
        token = Token {
            tp: Acc::LITERAL,
            val: buf,
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
        // skip spaces
        buf = " rule0";
        pos = 0;
        token = Token {
            tp: Acc::RULE_NAME,
            val: &buf[1..],
        };
        assert_eq!(next_token(buf.as_bytes(), &mut pos), Some(token));
    }

    #[test]
    fn ebnf_term_ok() {
        let ebnf = "rule0 = 'literal';";
        let parsed = bootstrap_ebnf(ebnf);
        assert!(parsed.is_ok());
    }

    #[test]
    fn ebnf_missing_semi() {
        let ebnf = "rule0 = 'literal'";
        let parsed = bootstrap_ebnf(ebnf);
        assert!(parsed.is_err());
    }

    #[test]
    fn ebnf_same_line() {
        let ebnf = "rule0 = rule1; rule1 = \"literal\";";
        let parsed = bootstrap_ebnf(ebnf).unwrap();
        assert_eq!(parsed.len_term(), 1);
        assert_eq!(parsed.len_nonterm(), 2);
    }

    #[test]
    fn ebnf_or() {
        let ebnf = "rule0 = rule1 | 'literal0';
            rule1 = rule2 | rule3 'literal1';
            rule2 = 'literal2';
            rule3 = \"literal2\";";
        let parsed = bootstrap_ebnf(ebnf).unwrap();
        assert_eq!(parsed.len_term(), 3);
        assert_eq!(parsed.len_nonterm(), 4);
    }

    #[test]
    fn ebnf_empty() {
        let ebnf = "rule0 = 'a'
                          |
                          ;
                    rule1 = 'b';";
        let parsed = bootstrap_ebnf(ebnf).unwrap();
        assert_eq!(
            *parsed.non_terminals[0].body.child(1).unwrap(),
            Tree::new_leaf(ParserRuleElement::Empty)
        );
        assert_eq!(parsed.len_term(), 2);
        assert_eq!(parsed.len_nonterm(), 2);
    }
}
