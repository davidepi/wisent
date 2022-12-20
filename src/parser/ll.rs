use super::conversion::CanonicalParserRuleElement;
use super::EPSILON_VAL;
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

#[cfg(test)]
mod tests {

    use super::first;
    use crate::fxhashset;
    use crate::grammar::Grammar;
    use crate::parser::conversion::canonicalise;
    use crate::parser::EPSILON_VAL;

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
}
