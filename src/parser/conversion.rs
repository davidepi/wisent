use crate::error::ParseError;
use crate::grammar::{Grammar, ParserRuleElement, Tree};
use std::collections::HashMap;

/// Terminal, Non-Terminal, And and Or only. No strings, allowed, only IDs.
#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub enum CanonicalParserRuleElement {
    Terminal(u32),
    NonTerminal(u32),
    Empty,
    AND,
    OR,
}

pub fn canonicalise(
    grammar: &Grammar,
) -> Result<Vec<Tree<CanonicalParserRuleElement>>, ParseError> {
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
    grammar
        .iter_nonterm()
        .map(|prod| canonicalise_rec(&prod.body, &names))
        .collect()
}

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
                    message: format!("Unreferenced non-terminal {}", name),
                })
            }
        }
        ParserRuleElement::Terminal(name) => {
            if let Some(&id) = names.get(name.as_str()) {
                Ok(Tree::new_leaf(CanonicalParserRuleElement::Terminal(id)))
            } else {
                Err(ParseError::SyntaxError {
                    message: format!("Unreferenced terminal {}", name),
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
                Ok(Tree::new_node(CanonicalParserRuleElement::OR, children))
            }
            crate::grammar::LexerOp::And => {
                let children = node
                    .children()
                    .map(|c| canonicalise_rec(c, names))
                    .collect::<Result<Vec<_>, ParseError>>()?;
                Ok(Tree::new_node(CanonicalParserRuleElement::AND, children))
            }
        },
    }
}
