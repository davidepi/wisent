use super::conversion::flatten;
use super::ll::{first, follow};
use super::ParserSymbol;
use crate::error::ParseError;
use crate::grammar::Grammar;
use rustc_hash::FxHashSet;

/// A grammar for LR Parsers
///
/// The struct `LRGrammar` can be used to construct LR-based parsers. This
/// struct can be built from an original [`Grammar`] with the method
/// [`LRGrammar::try_from`] only if the original grammar contains a rule for
/// each nonterminal referenced in each production.
///
/// By converting a [`Grammar`] to a [`LRGrammar`], a new production `S' -> S`,
/// where `S` is the starting production, is added.
///
/// This construction works even if the grammar is not LR, and conflicts should
/// be handled at table generation time.
#[derive(Clone)]
pub struct LRGrammar {
    token_names: Vec<String>,
    nonterminal_names: Vec<String>,
    nonterminals: Vec<Vec<Vec<ParserSymbol>>>,
    starting_rule: u32,
    follow: Vec<FxHashSet<u32>>,
}

impl TryFrom<&Grammar> for LRGrammar {
    type Error = ParseError;

    /// Converts a [`Grammar`] into a [`LRGrammar`].
    ///
    /// Returns [`ParseError::SyntaxError`] if any production references
    /// undeclared rules.
    fn try_from(value: &Grammar) -> Result<Self, Self::Error> {
        let mut nonterminals = flatten(value)?;
        let augmented_production = vec![vec![ParserSymbol::NonTerminal(value.starting_rule())]];
        nonterminals.push(augmented_production);
        let starting_rule = (nonterminals.len() - 1) as u32;
        let nonterminal_names = value
            .iter_nonterm()
            .map(|x| &x.head)
            .cloned()
            .chain(std::iter::once("augmented$".to_string()))
            .collect::<Vec<_>>();
        let token_names = value
            .iter_term()
            .map(|x| &x.head)
            .cloned()
            .collect::<Vec<_>>();
        let first = first(&nonterminals);
        let follow = follow(&nonterminals, &first, starting_rule);
        Ok(LRGrammar {
            token_names,
            nonterminal_names,
            nonterminals,
            starting_rule,
            follow,
        })
    }
}

/// Expands a list of LR0 kernel items.
fn closure(
    items: &FxHashSet<KernelLR0Item>,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
) -> FxHashSet<LR0Item> {
    let mut set = items
        .iter()
        .map(|x| LR0Item::from(*x))
        .collect::<FxHashSet<_>>();
    let mut old_len = 0;
    let mut new_len = set.len();
    while new_len != old_len {
        old_len = new_len;
        let mut to_add = FxHashSet::default();
        for item in &set {
            if let ParserSymbol::NonTerminal(nt) = item.peek(nonterminals) {
                let new_items = nonterminals[nt as usize]
                    .iter()
                    .enumerate()
                    .map(|(prod_id, _)| LR0Item::nonkernel(nt, prod_id as u16));
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

/// Advances a closure of LR0 kernel items and returns only the kernel items
/// composing the new list.
fn goto(
    items: &FxHashSet<LR0Item>,
    symbol: ParserSymbol,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
) -> FxHashSet<KernelLR0Item> {
    let mut advanced = FxHashSet::default();
    for item in items {
        if item.peek(nonterminals) == symbol {
            advanced.insert(item.next());
        }
    }
    advanced
        .into_iter()
        .filter_map(|x| match x {
            LR0Item::Kernel(k) => Some(k),
            LR0Item::NonKernel(_) => None,
        })
        .collect()
}

struct LRParsingTable {}

fn slr_parsing_table(
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    term_no: u32,
    start: u32,
    follow: &[FxHashSet<u32>],
) -> Result<LRParsingTable, ParserSymbol> {
    unimplemented!()
}

// Supporting structs to avoid unwanted mixing of kernel/nonkernel using goto on
// a non-closure and stuffs like that

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct KernelLR0Item {
    rule: u32,
    production: u16,
    position: u16,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct NonKernelLR0Item {
    rule: u32,
    production: u16,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum LR0Item {
    Kernel(KernelLR0Item),
    NonKernel(NonKernelLR0Item),
}

impl LR0Item {
    fn kernel(rule: u32, production: u16, position: u16) -> Self {
        Self::Kernel(KernelLR0Item { rule, production, position })
    }

    fn nonkernel(rule: u32, production: u16) -> Self {
        Self::NonKernel(NonKernelLR0Item { rule, production })
    }

    #[must_use]
    fn next(&self) -> Self {
        Self::Kernel(KernelLR0Item {
            rule: self.rule(),
            production: self.production(),
            position: self.position() + 1,
        })
    }

    // returns the next symbol. ParserSymbol::Empty in case of A -> ·
    fn peek(&self, nonterminals: &[Vec<Vec<ParserSymbol>>]) -> ParserSymbol {
        let (r, p, i) = match self {
            LR0Item::Kernel(k) => (k.rule as usize, k.production as usize, k.position as usize),
            LR0Item::NonKernel(nk) => (nk.rule as usize, nk.production as usize, 0),
        };
        nonterminals[r][p]
            .get(i)
            .copied()
            .unwrap_or(ParserSymbol::Empty)
    }

    fn rule(&self) -> u32 {
        match self {
            LR0Item::Kernel(k) => k.rule,
            LR0Item::NonKernel(nk) => nk.rule,
        }
    }

    fn production(&self) -> u16 {
        match self {
            LR0Item::Kernel(k) => k.production,
            LR0Item::NonKernel(nk) => nk.production,
        }
    }

    fn position(&self) -> u16 {
        match self {
            LR0Item::Kernel(k) => k.position,
            LR0Item::NonKernel(_) => 0,
        }
    }
}

impl From<KernelLR0Item> for LR0Item {
    fn from(value: KernelLR0Item) -> Self {
        LR0Item::Kernel(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{closure, LR0Item};
    use crate::fxhashset;
    use crate::parser::conversion::flatten;
    use crate::parser::lr::{goto, KernelLR0Item};
    use crate::parser::tests::grammar_434;
    use crate::parser::ParserSymbol;

    #[test]
    fn closure_set() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(KernelLR0Item { rule: 0, production: 0, position: 0 });
        let closure = closure(&items, nonterminals.as_slice());
        let expected = fxhashset!(
            LR0Item::kernel(0, 0, 0),
            LR0Item::nonkernel(1, 0),
            LR0Item::nonkernel(1, 1),
            LR0Item::nonkernel(2, 0),
            LR0Item::nonkernel(2, 1),
            LR0Item::nonkernel(3, 0),
            LR0Item::nonkernel(3, 1),
        );
        assert_eq!(closure, expected);
    }

    #[test]
    fn goto_function() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            KernelLR0Item { rule: 0, production: 0, position: 1 },
            KernelLR0Item { rule: 1, production: 0, position: 1 }
        );
        let ic = closure(&items, &nonterminals);
        let advanced = goto(&ic, ParserSymbol::Terminal(0), &nonterminals);
        let expected = fxhashset!(KernelLR0Item { rule: 1, production: 0, position: 2 },);
        assert_eq!(advanced, expected);
    }

    #[test]
    fn goto_empty() {
        let grammar = grammar_434();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            KernelLR0Item { rule: 0, production: 0, position: 1 },
            KernelLR0Item { rule: 1, production: 0, position: 1 }
        );
        let ic = closure(&items, &nonterminals);
        let advanced = goto(&ic, ParserSymbol::Terminal(5), &nonterminals);
        assert!(advanced.is_empty());
    }
}
