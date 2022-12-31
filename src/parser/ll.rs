use super::conversion::CanonicalParserRuleElement;
use super::{ENDLINE_VAL, EPSILON_VAL};
use crate::fxhashset;
use crate::grammar::Tree;
use rustc_hash::FxHashSet;

fn first(prods: &[Tree<CanonicalParserRuleElement>]) -> Vec<FxHashSet<u32>> {
    let mut old = vec![FxHashSet::default(); prods.len()];
    let mut saturated = false;
    while !saturated {
        let new = prods
            .iter()
            .map(|prod| first_rec(prod, prods, &old))
            .collect::<Vec<_>>();
        if old == new {
            saturated = true;
        } else {
            old = new;
        }
    }
    old
}

fn first_rec(
    node: &Tree<CanonicalParserRuleElement>,
    prods: &[Tree<CanonicalParserRuleElement>],
    first: &[FxHashSet<u32>],
) -> FxHashSet<u32> {
    match node.value() {
        CanonicalParserRuleElement::Terminal(t) => fxhashset!(*t),
        CanonicalParserRuleElement::NonTerminal(nt) => first[*nt as usize].clone(),
        CanonicalParserRuleElement::Empty => fxhashset!(EPSILON_VAL),
        CanonicalParserRuleElement::OR => node
            .children()
            .flat_map(|child| first_rec(child, prods, first))
            .collect(),
        CanonicalParserRuleElement::AND => {
            let mut set = FxHashSet::default();
            for child in node.children() {
                let child_first = first_rec(child, prods, first);
                let has_epsilon = child_first.contains(&EPSILON_VAL);
                set.extend(child_first);
                if !has_epsilon {
                    break;
                }
            }
            set
        }
    }
}

fn follow(
    prods: &[Tree<CanonicalParserRuleElement>],
    first: &[FxHashSet<u32>],
    start: usize,
) -> Vec<FxHashSet<u32>> {
    let mut old = vec![FxHashSet::default(); prods.len()];
    old[start].insert(ENDLINE_VAL);
    let mut saturated = false;
    while !saturated {
        let mut new = old.clone();
        for (prod_id, prod) in prods.iter().enumerate() {
            follow_rec(prod_id, prod, first, &mut new);
        }
        if old == new {
            saturated = true;
        } else {
            old = new;
        }
    }
    old
}

fn follow_rec(
    prod_id: usize,
    prod: &Tree<CanonicalParserRuleElement>,
    first: &[FxHashSet<u32>],
    follow: &mut [FxHashSet<u32>],
) {
    match prod.value() {
        CanonicalParserRuleElement::NonTerminal(nt) => {
            let cur_follow = follow[prod_id].iter().copied().collect::<Vec<_>>();
            follow[*nt as usize].extend(cur_follow);
        }
        CanonicalParserRuleElement::OR => {
            for child in prod.children() {
                follow_rec(prod_id, child, first, follow);
            }
        }
        CanonicalParserRuleElement::AND => {
            // If there is a production A ⇒ αΒ, or a production A ⇒ αΒβ where FIRST(β)
            // contains ε, then everything in FOLLOW(A) is in FOLLOW(B).
            for child in prod.children().rev() {
                match child.value() {
                    CanonicalParserRuleElement::NonTerminal(nt) => {
                        let cur_follow = follow[prod_id].iter().copied().collect::<Vec<_>>();
                        follow[*nt as usize].extend(cur_follow);
                        if !first[*nt as usize].contains(&EPSILON_VAL) {
                            break;
                        }
                    }
                    CanonicalParserRuleElement::Terminal(_) => break,
                    _ => panic!("Unexpected nested production in nonterminals"),
                }
            }
            // If there is a production A ⇒ αΒβ, then everything in FIRST(β), except for ε,
            // is placed in FOLLOW(B)
            let mut peekable = prod.children().peekable();
            while let Some(b) = peekable.next() {
                match b.value() {
                    CanonicalParserRuleElement::NonTerminal(nt) => {
                        if let Some(beta) = peekable.peek() {
                            match beta.value() {
                                CanonicalParserRuleElement::Terminal(t) => {
                                    follow[*nt as usize].insert(*t);
                                }
                                CanonicalParserRuleElement::NonTerminal(beta_nt) => {
                                    let beta_first = first[*beta_nt as usize]
                                        .iter()
                                        .filter(|&&f| f != EPSILON_VAL);
                                    follow[*nt as usize].extend(beta_first);
                                }
                                _ => panic!("Unexpected nested production in nonterminals"),
                            }
                        }
                    }
                    CanonicalParserRuleElement::Terminal(_) => (),
                    _ => panic!("Unexpected nested production in nonterminals"),
                }
            }
        }
        _ => (),
    }
}

#[cfg(test)]
mod tests {

    use super::first;
    use crate::fxhashset;
    use crate::grammar::Grammar;
    use crate::parser::conversion::canonicalise;
    use crate::parser::ll::follow;
    use crate::parser::{ENDLINE_VAL, EPSILON_VAL};

    /// Grammar 4.28 of the dragon book. Page 217 on the second edition.
    fn grammar_428() -> Grammar {
        let g = "e : t e1;
             e1: Plus t e1 | ;
             t: f t1;
             t1: Star f t1 | ;
             f: Lpar e Rpar | Id;
             Plus: '+';
             Star: '*';
             Lpar: '(';
             Rpar: ')';
             Id: [0123456789]+;";
        Grammar::parse_bootstrap(g).unwrap()
    }

    #[test]
    fn first_set() {
        let g = grammar_428();
        let canonical = canonicalise(&g).unwrap();
        let first = first(&canonical);
        assert_eq!(first[0], fxhashset! {2, 4});
        assert_eq!(first[1], fxhashset! {0, EPSILON_VAL});
        assert_eq!(first[2], fxhashset! {2, 4});
        assert_eq!(first[3], fxhashset! {1, EPSILON_VAL});
        assert_eq!(first[4], fxhashset! {2, 4});
    }

    #[test]
    fn follow_set() {
        let g = grammar_428();
        let canonical = canonicalise(&g).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        assert_eq!(follow[0], fxhashset! {3, ENDLINE_VAL});
        assert_eq!(follow[1], fxhashset! {3, ENDLINE_VAL});
        assert_eq!(follow[2], fxhashset! {0, 3, ENDLINE_VAL});
        assert_eq!(follow[3], fxhashset! {0, 3, ENDLINE_VAL});
        assert_eq!(follow[4], fxhashset! {0, 1, 3, ENDLINE_VAL});
    }
}
