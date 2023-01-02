use super::conversion::{canonicalise, CanonicalParserRuleElement};
use super::{ENDLINE_VAL, EPSILON_VAL};
use crate::error::ParseError;
use crate::fxhashset;
use crate::grammar::{Grammar, Tree};
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
    start: u32,
) -> Vec<FxHashSet<u32>> {
    let mut old = vec![FxHashSet::default(); prods.len()];
    old[start as usize].insert(ENDLINE_VAL);
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

// the last entry in the terminals is the $ sign (ENDLINE_VAL)
fn ll1_parsing_table(
    prods: &[Tree<CanonicalParserRuleElement>],
    terminals: u32,
    first: &[FxHashSet<u32>],
    follow: &[FxHashSet<u32>],
) -> Vec<Vec<Option<(u32, u32)>>> {
    let mut table = vec![vec![None; terminals as usize + 1]; prods.len()]; // +1 is the $ sign
    for (prod_id, prod) in prods.iter().enumerate() {
        if prod.value() == &CanonicalParserRuleElement::OR {
            for (child_id, child) in prod.children().enumerate() {
                ll1_rec(
                    prod_id as u32,
                    child_id as u32,
                    child,
                    terminals,
                    first,
                    follow,
                    &mut table,
                );
            }
        } else {
            ll1_rec(
                prod_id as u32,
                0,
                prod,
                terminals,
                first,
                follow,
                &mut table,
            );
        }
    }
    table
}

fn ll1_rec(
    id: u32,
    alternative: u32,
    prod: &Tree<CanonicalParserRuleElement>,
    terminals: u32,
    first: &[FxHashSet<u32>],
    follow: &[FxHashSet<u32>],
    table: &mut [Vec<Option<(u32, u32)>>],
) {
    let first = match prod.value() {
        CanonicalParserRuleElement::Terminal(t) => fxhashset!(*t),
        CanonicalParserRuleElement::NonTerminal(nt) => first[*nt as usize].clone(),
        CanonicalParserRuleElement::Empty => fxhashset!(EPSILON_VAL),
        CanonicalParserRuleElement::AND => match prod.children().next().unwrap().value() {
            CanonicalParserRuleElement::Terminal(t) => fxhashset!(*t),
            CanonicalParserRuleElement::NonTerminal(nt) => first[*nt as usize].clone(),
            _ => panic!("Unexpected nested production in nonterminals"),
        },
        CanonicalParserRuleElement::OR => panic!("Unexpected nested production in nonterminals"),
    };
    for &term in first.iter().filter(|&&e| e != EPSILON_VAL) {
        table[id as usize][term as usize] = Some((id, alternative));
    }
    if first.contains(&EPSILON_VAL) {
        for &term in follow[id as usize].iter().filter(|&&e| e != EPSILON_VAL) {
            if term != ENDLINE_VAL {
                table[id as usize][term as usize] = Some((id, alternative));
            } else {
                table[id as usize][terminals as usize] = Some((id, alternative));
            }
        }
    }
}

/// Returns the first sets and the follow sets of a given grammar.
///
/// These sets represent the set of terminals appearing respectively at the start and to the right
/// (following) a production. The terminals contained in each set are represented by their index in
/// the grammar. Two special symbols can appear in these sets: [`EPSILON_VAL`] and [`ENDLINE_VAL`].
///
/// The returned vectors are indexed by the nonterminal production. This means that `first[0]`
/// contains the first set of the first nonterminal production appearing in the grammar using
/// [`Grammar::iter_nonterm`].
/// The first set of each terminal production is trivial, being equal to the terminal itself.
/// For this reason, these are not contained in the returned vector.
///
/// This function requires also the index of the starting production.
/// # Example
/// Basic usage:
/// ```
/// # use wisent::grammar::Grammar;
/// # use wisent::parser::first_follow;
/// # use wisent::parser::ENDLINE_VAL;
/// let grammar_text = "LetterA: 'a';
///                     LetterB: 'b';
///                     LetterC: 'c';
///                     s: s LetterA | s LetterB | LetterC;";
/// let grammar = Grammar::parse_bootstrap(grammar_text).unwrap();
/// let (first, follow) = first_follow(&grammar, 0).unwrap();
///
/// assert!(first[0].contains(&2));
/// assert!(follow[0].contains(&0));
/// assert!(follow[0].contains(&1));
/// assert!(follow[0].contains(&ENDLINE_VAL));
/// ```
pub fn first_follow(
    grammar: &Grammar,
    start_index: u32,
) -> Result<(Vec<FxHashSet<u32>>, Vec<FxHashSet<u32>>), ParseError> {
    let canonical = canonicalise(grammar)?;
    let first = first(&canonical);
    let follow = follow(&canonical, &first, start_index);
    Ok((first, follow))
}

#[cfg(test)]
mod tests {
    use crate::fxhashset;
    use crate::grammar::Grammar;
    use crate::parser::conversion::canonicalise;
    use crate::parser::ll::{first, follow, ll1_parsing_table};
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

    #[test]
    fn parsing_table() {
        let g = grammar_428();
        let canonical = canonicalise(&g).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        let table = ll1_parsing_table(&canonical, g.len_term() as u32, &first, &follow);
        let expected = [
            [None, None, Some((0, 0)), None, Some((0, 0)), None],
            [Some((1, 0)), None, None, Some((1, 1)), None, Some((1, 1))],
            [None, None, Some((2, 0)), None, Some((2, 0)), None],
            [Some((3, 1)), Some((3, 0)), None, Some((3, 1)), None, Some((3, 1))],
            [None, None, Some((4, 0)), None, Some((4, 1)), None],
        ];
        assert_eq!(table, expected);
    }
}
