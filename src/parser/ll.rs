use super::EPSILON_STR;
use crate::grammar::{Grammar, LexerOp, ParserRuleElement, Tree};
use maplit::hashset;
use std::collections::{HashMap, HashSet};

fn first(grammar: &Grammar) -> Vec<HashSet<&str>> {
    let mut first: HashMap<&Tree<ParserRuleElement>, HashSet<&str>> = HashMap::new();
    let named_trees = grammar
        .iter_nonterm()
        .map(|p| (p.head.as_str(), &p.body))
        .collect::<HashMap<_, _>>();
    let mut saturated = false;
    while !saturated {
        let start_size: usize = first.values().map(|v| v.len()).sum();
        grammar
            .iter_nonterm()
            .for_each(|node| first_rec(&node.body, &mut first, &named_trees));
        let end_size: usize = first.values().map(|v| v.len()).sum();
        if start_size == end_size {
            saturated = true
        }
    }
    grammar
        .iter_nonterm()
        .map(|p| first.get(&p.body).cloned().unwrap_or_default())
        .collect()
}

fn first_rec<'a>(
    node: &'a Tree<ParserRuleElement>,
    first: &mut HashMap<&'a Tree<ParserRuleElement>, HashSet<&'a str>>,
    named_trees: &HashMap<&str, &'a Tree<ParserRuleElement>>,
) {
    match node.value() {
        ParserRuleElement::Terminal(name) => {
            first
                .entry(node)
                .and_modify(|set| {
                    set.insert(name.as_str());
                })
                .or_insert(hashset! {name.as_str()});
        }
        ParserRuleElement::NonTerminal(nt) => {
            let nt_tree = *named_trees
                .get(nt.as_str())
                .expect("Unreferenced non terminal");
            first_rec(nt_tree, first, named_trees);
            let nonterm_firsts = first.get(nt_tree).cloned().unwrap_or_default();
            first
                .entry(node)
                .and_modify(|set| set.extend(nonterm_firsts.iter()))
                .or_insert(nonterm_firsts);
        }
        ParserRuleElement::Empty => {
            first
                .entry(node)
                .and_modify(|set| {
                    set.insert(EPSILON_STR);
                })
                .or_insert(hashset! {EPSILON_STR});
        }
        ParserRuleElement::Operation(op) => match op {
            LexerOp::Kleene => todo!(),
            LexerOp::Qm => todo!(),
            LexerOp::Pl => todo!(),
            LexerOp::Not => todo!(),
            LexerOp::Or => {
                for child in node.children() {
                    first_rec(child, first, named_trees);
                    let child_first = first.get(child).cloned().unwrap_or_default();
                    first
                        .entry(node)
                        .and_modify(|set| {
                            set.extend(child_first.iter());
                        })
                        .or_insert(child_first);
                }
            }
            LexerOp::And => {
                for child in node.children() {
                    first_rec(child, first, named_trees);
                    let child_first = first.get(child).cloned().unwrap_or_default();
                    let has_epsilon = child_first.contains(EPSILON_STR);
                    first
                        .entry(node)
                        .and_modify(|set| {
                            set.extend(child_first.iter());
                        })
                        .or_insert(child_first);
                    if has_epsilon {
                        break;
                    }
                }
            }
        },
    }
}
