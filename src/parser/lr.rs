use super::ParserSymbol;
use rustc_hash::FxHashSet;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct LR0Item {
    rule: u32,
    production: u16,
    position: u16,
}

/// Expands a list of LR0 kernel items.
fn closure(
    items: &FxHashSet<LR0Item>,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
) -> FxHashSet<LR0Item> {
    let mut set = items.iter().cloned().collect::<FxHashSet<_>>();
    let mut old_len = 0;
    let mut new_len = set.len();
    while new_len != old_len {
        old_len = new_len;
        let mut to_add = FxHashSet::default();
        for item in &set {
            let production = &nonterminals[item.rule as usize][item.production as usize];
            if let Some(ParserSymbol::NonTerminal(nt)) = production.get(item.position as usize) {
                let new_items =
                    nonterminals[*nt as usize]
                        .iter()
                        .enumerate()
                        .map(|(prod_id, _)| LR0Item {
                            rule: *nt,
                            production: prod_id as u16,
                            position: 0,
                        });
                to_add.extend(new_items);
            } else {
                // do nothing for items like A -> · or terminals
            }
        }
        set.extend(to_add);
        new_len = set.len();
    }
    set
}

/// Advances a list of LR0 kernel items (considering also its closure) and returns only the kernel
/// items composing the new list.
fn goto(
    items: &FxHashSet<LR0Item>,
    symbol: ParserSymbol,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
) -> FxHashSet<LR0Item> {
    let mut advanced = FxHashSet::default();
    for item in closure(items, nonterminals) {
        if let Some(&sym) =
            nonterminals[item.rule as usize][item.production as usize].get(item.position as usize)
        {
            if sym == symbol {
                let mut advanced_symbol = item;
                advanced_symbol.position += 1;
                advanced.insert(advanced_symbol);
            }
        }
    }
    advanced
}

#[cfg(test)]
mod tests {
    use super::{closure, LR0Item};
    use crate::fxhashset;
    use crate::parser::conversion::flatten;
    use crate::parser::lr::goto;
    use crate::parser::tests::grammar_434;
    use crate::parser::ParserSymbol;

    #[test]
    fn closure_set() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(LR0Item {
            rule: 0,
            production: 0,
            position: 0
        });
        let closure = closure(&items, nonterminals.as_slice());
        let expected = fxhashset!(
            LR0Item {
                rule: 0,
                production: 0,
                position: 0,
            },
            LR0Item {
                rule: 1,
                production: 0,
                position: 0,
            },
            LR0Item {
                rule: 1,
                production: 1,
                position: 0,
            },
            LR0Item {
                rule: 2,
                production: 0,
                position: 0,
            },
            LR0Item {
                rule: 2,
                production: 1,
                position: 0,
            },
            LR0Item {
                rule: 3,
                production: 0,
                position: 0,
            },
            LR0Item {
                rule: 3,
                production: 1,
                position: 0,
            },
        );
        assert_eq!(closure, expected);
    }

    #[test]
    fn goto_function() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            LR0Item {
                rule: 0,
                production: 0,
                position: 1
            },
            LR0Item {
                rule: 1,
                production: 0,
                position: 1
            }
        );
        let advanced = goto(&items, ParserSymbol::Terminal(0), &nonterminals);
        let expected = fxhashset!(LR0Item {
            rule: 1,
            production: 0,
            position: 2,
        },);
        assert_eq!(advanced, expected);
    }

    #[test]
    fn goto_empty() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            LR0Item {
                rule: 0,
                production: 0,
                position: 1
            },
            LR0Item {
                rule: 1,
                production: 0,
                position: 1
            }
        );
        let advanced = goto(&items, ParserSymbol::Terminal(5), &nonterminals);
        assert!(advanced.is_empty());
    }
}
