use crate::error::ParseError;
use crate::grammar::{Grammar, ParserRuleElement, Tree};
use crate::parser::ParserSymbol;
use std::collections::HashMap;

/// Terminal, Non-Terminal, And and Or only. No strings, allowed, only IDs.
#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub enum CanonicalParserRuleElement {
    Terminal(u32),
    NonTerminal(u32),
    Empty,
    And,
    Or,
}

/// Flattens the nonterminals of a grammar by converting them from a tree to a
/// 3D vector.
///
/// The first dimension represents each non terminal production
/// The second dimension represents the various alternative for a production
/// The third dimension represents the concatenation of the various symbols
pub(super) fn flatten(grammar: &Grammar) -> Result<Vec<Vec<Vec<ParserSymbol>>>, ParseError> {
    let mut names = HashMap::new();
    names.extend(
        grammar
            .iter_term()
            .enumerate()
            .map(|(id, t)| (t.head.as_str(), id as u32)),
    );
    names.extend(
        grammar
            .iter_nonterm()
            .enumerate()
            .map(|(id, t)| (t.head.as_str(), id as u32)),
    );
    let canonical = grammar
        .iter_nonterm()
        .map(|prod| canonicalise_rec(&prod.body, &names))
        .collect::<Result<Vec<Tree<CanonicalParserRuleElement>>, ParseError>>()?;
    Ok(canonical.into_iter().map(|prod| flatten1(&prod)).collect())
}

/// Removes nested productions and converts different grammars to a common
/// representation.
///
/// This common representation allows a single top-level Or, followed by
/// sub-productions containing only concatenations.
///
/// TODO: Missing nested productions and */+/?
fn canonicalise_rec(
    node: &Tree<ParserRuleElement>,
    names: &HashMap<&str, u32>,
) -> Result<Tree<CanonicalParserRuleElement>, ParseError> {
    match node.value() {
        ParserRuleElement::NonTerminal(name) => {
            if let Some(&id) = names.get(name.as_str()) {
                Ok(Tree::new_leaf(CanonicalParserRuleElement::NonTerminal(id)))
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unreferenced non-terminal {name}"),
                })
            }
        }
        ParserRuleElement::Terminal(name) => {
            if let Some(&id) = names.get(name.as_str()) {
                Ok(Tree::new_leaf(CanonicalParserRuleElement::Terminal(id)))
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unreferenced terminal {name}"),
                })
            }
        }
        ParserRuleElement::Empty => Ok(Tree::new_leaf(CanonicalParserRuleElement::Empty)),
        ParserRuleElement::Operation(op) => match op {
            crate::grammar::LexerOp::Kleene => todo!(),
            crate::grammar::LexerOp::Qm => todo!(),
            crate::grammar::LexerOp::Pl => todo!(),
            crate::grammar::LexerOp::Not => todo!(),
            crate::grammar::LexerOp::Or => {
                let children = node
                    .children()
                    .map(|c| canonicalise_rec(c, names))
                    .collect::<Result<Vec<_>, ParseError>>()?;
                Ok(Tree::new_node(CanonicalParserRuleElement::Or, children))
            }
            crate::grammar::LexerOp::And => {
                let children = node
                    .children()
                    .map(|c| canonicalise_rec(c, names))
                    .collect::<Result<Vec<_>, ParseError>>()?;
                Ok(Tree::new_node(CanonicalParserRuleElement::And, children))
            }
        },
    }
}

/// flatten the first fimension of a canonical parser tree (alternative)
fn flatten1(node: &Tree<CanonicalParserRuleElement>) -> Vec<Vec<ParserSymbol>> {
    match node.value() {
        CanonicalParserRuleElement::Or => node.children().map(flatten2).collect(),
        _ => vec![flatten2(node)],
    }
}

/// flatten the second fimension of a canonical parser tree (concatenation)
///
/// panics if the input tree contains OR
fn flatten2(node: &Tree<CanonicalParserRuleElement>) -> Vec<ParserSymbol> {
    match node.value() {
        CanonicalParserRuleElement::And => node.children().map(flatten3).collect(),
        CanonicalParserRuleElement::Or => panic!("Unexpected nested OR in canonical parser tree"),
        _ => vec![flatten3(node)],
    }
}

/// calculates the first fimension of a canonical parser tree
/// (terminal/nonterminal/epsilon)
///
/// panics if the input tree contains AND or OR
fn flatten3(node: &Tree<CanonicalParserRuleElement>) -> ParserSymbol {
    match node.value() {
        CanonicalParserRuleElement::Terminal(t) => ParserSymbol::Terminal(*t),
        CanonicalParserRuleElement::NonTerminal(nt) => ParserSymbol::NonTerminal(*nt),
        CanonicalParserRuleElement::Empty => ParserSymbol::Empty,
        CanonicalParserRuleElement::Or => panic!("Unexpected nested OR in canonical parser tree"),
        CanonicalParserRuleElement::And => panic!("Unexpected nested AND in canonical parser tree"),
    }
}

#[cfg(test)]
mod tests {
    use super::{flatten, ParserSymbol};
    use crate::error::ParseError;
    use crate::grammar::Grammar;
    use crate::parser::tests::grammar_428;

    // from https://stackoverflow.com/questions/53124930/how-do-you-test-for-a-specific-rust-error
    macro_rules! assert_err {
    ($expression:expr, $($pattern:tt)+) => {
        match $expression {
            $($pattern)+ => (),
            ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
        }
    }
}

    #[test]
    fn unreferenced_nonterminal() {
        let g = "e : t;";
        let grammar = Grammar::parse_bootstrap(g).unwrap();
        assert_err!(flatten(&grammar), Err(ParseError::SyntaxError { .. }))
    }

    #[test]
    fn unreferenced_terminal() {
        let g = "e: T;";
        let grammar = Grammar::parse_bootstrap(g).unwrap();
        assert_err!(flatten(&grammar), Err(ParseError::SyntaxError { .. }))
    }

    #[test]
    fn flatten_non_nested() {
        let g = grammar_428();
        let flatten = flatten(&g).unwrap();
        let expected = vec![
            vec![vec![ParserSymbol::NonTerminal(2), ParserSymbol::NonTerminal(1)]],
            vec![
                vec![
                    ParserSymbol::Terminal(0),
                    ParserSymbol::NonTerminal(2),
                    ParserSymbol::NonTerminal(1),
                ],
                vec![ParserSymbol::Empty],
            ],
            vec![vec![ParserSymbol::NonTerminal(4), ParserSymbol::NonTerminal(3)]],
            vec![
                vec![
                    ParserSymbol::Terminal(1),
                    ParserSymbol::NonTerminal(4),
                    ParserSymbol::NonTerminal(3),
                ],
                vec![ParserSymbol::Empty],
            ],
            vec![
                vec![
                    ParserSymbol::Terminal(2),
                    ParserSymbol::NonTerminal(0),
                    ParserSymbol::Terminal(3),
                ],
                vec![ParserSymbol::Terminal(4)],
            ],
        ];
        assert_eq!(flatten, expected);
    }
}
