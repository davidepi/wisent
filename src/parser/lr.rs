use super::conversion::flatten;
use super::ll::{first, follow};
use super::ENDLINE_VAL;
use crate::error::ParseError;
use crate::fxhashset;
use crate::grammar::Grammar;
use crate::parser::ParserSymbol;
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

impl LRGrammar {
    /// Computes the SLR(1) parsing table for the given grammar.
    ///
    /// This table can be used in a [Table Driven Parser](super::LRParser).
    ///
    /// Expects the grammar and the index of the starting non-terminal as input.
    ///
    /// Returns [`ParseError::LRError`] if there are SHIFT/REDUCE or
    /// REDUCE/REDUCE conflicts.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::LRGrammar;
    /// let g = "sum: num PLUS num;
    ///          num: INT | REAL;
    ///          INT: [0-9]+;
    ///          REAL: [0-9]+ '.' [0-9]+;
    ///          PLUS: '+';";
    /// let grammar = Grammar::parse_bootstrap(g).unwrap();
    /// let grammar_lr = LRGrammar::try_from(&grammar).unwrap();
    /// let slr_table = grammar_lr.slr_parsing_table().unwrap();
    /// ```
    pub fn slr_parsing_table(&self) -> Result<LRParsingTable, ParseError> {
        slr_parsing_table(
            &self.nonterminals,
            self.token_names.len() as u32,
            self.starting_rule,
            &self.follow,
        )
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

/// Calculates the DFA representing the LR0 automaton.
/// Sometimes called canonical lr0 collection.
fn lr0_automaton(nonterminals: &[Vec<Vec<ParserSymbol>>], start: u32) -> LR0Automaton {
    let start_kernel = fxhashset!(KernelLR0Item {
        rule: start,
        production: 0,
        position: 0
    });
    let mut nodes = vec![start_kernel];
    let mut edges = vec![vec![]];
    let mut done = fxhashset!();
    let mut todo = vec![0];
    while let Some(set_id) = todo.pop() {
        done.insert(set_id);
        let kernel = &nodes[set_id];
        let clos = closure(kernel, nonterminals);
        for &item in &clos {
            let peek = item.peek(nonterminals);
            if peek != ParserSymbol::Empty {
                let next = goto(&clos, peek, nonterminals);
                let next_index = nodes.iter().position(|x| x == &next).unwrap_or_else(|| {
                    nodes.push(next);
                    edges.push(Vec::new());
                    todo.push(nodes.len() - 1);
                    nodes.len() - 1
                }) as u32;
                edges[set_id].push((peek, next_index));
            }
        }
        edges[set_id].sort_unstable();
        edges[set_id].dedup();
    }
    LR0Automaton { nodes, edges }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShiftReduceAction {
    Shift(u32),
    Reduce((u32, u16)),
    Error,
    Accept,
}

/// Stores the parsing table for a LR parser.
#[derive(Debug, Clone)]
pub struct LRParsingTable {
    pub(super) action: Vec<Vec<ShiftReduceAction>>,
    pub(super) goto: Vec<Vec<Option<u32>>>,
    /// Stores the flattened grammar.
    /// Used to know how many nonterminals or tokens to pop when reducing.
    pub(super) nonterminals: Vec<Vec<Vec<ParserSymbol>>>,
}

impl LRParsingTable {
    /// The value assigned to the EOF character (`$`) in the current table.
    pub(super) fn eof_val(&self) -> u32 {
        self.action
            .get(0)
            .and_then(|x| Some(x.len() - 1))
            .unwrap_or(0) as u32
    }
}

/// builds an SLR(1) parsing table
fn slr_parsing_table(
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    term_no: u32,
    start: u32,
    follow: &[FxHashSet<u32>],
) -> Result<LRParsingTable, ParseError> {
    // replace ENDLINE_VAL with the actual table index
    let mut follow_indexed = follow.to_vec();
    for set in follow_indexed.iter_mut() {
        if set.contains(&ENDLINE_VAL) {
            set.remove(&ENDLINE_VAL);
            set.insert(term_no);
        }
    }
    let automaton = lr0_automaton(nonterminals, start);
    let mut action = Vec::with_capacity(automaton.nodes.len());
    let mut goto = Vec::with_capacity(automaton.nodes.len());
    for (node, edge) in automaton.nodes.iter().zip(automaton.edges.iter()) {
        let mut action_entry = vec![ShiftReduceAction::Error; (term_no + 1) as usize];
        let mut goto_entry = vec![None; nonterminals.len()];
        // fill shift
        for (symbol, target) in edge {
            match symbol {
                ParserSymbol::Terminal(t) => {
                    action_entry[*t as usize] = ShiftReduceAction::Shift(*target)
                }
                ParserSymbol::NonTerminal(nt) => goto_entry[*nt as usize] = Some(*target),
                ParserSymbol::Empty => panic!(),
            }
        }
        // fill reduce
        for item in node {
            if nonterminals[item.rule as usize][item.production as usize]
                .get(item.position as usize)
                .is_none()
            {
                if item.rule != start {
                    let red = ShiftReduceAction::Reduce((item.rule, item.production));
                    for &follow_term in &follow_indexed[item.rule as usize] {
                        let current_action = &mut action_entry[follow_term as usize];
                        match current_action {
                            ShiftReduceAction::Shift(_) => {
                                return Err(ParseError::LRError {
                                    message: "SHIFT/REDUCE conflict in SLR(1) generation"
                                        .to_string(),
                                })
                            }
                            ShiftReduceAction::Reduce(_) => {
                                return Err(ParseError::LRError {
                                    message: "REDUCE/REDUCE conflict in SLR(1) generation"
                                        .to_string(),
                                })
                            }
                            ShiftReduceAction::Error => *current_action = red,
                            ShiftReduceAction::Accept => panic!(),
                        }
                    }
                } else {
                    action_entry[term_no as usize] = ShiftReduceAction::Accept;
                }
            }
        }
        action.push(action_entry);
        goto.push(goto_entry);
    }
    Ok(LRParsingTable {
        action,
        goto,
        nonterminals: nonterminals.to_vec(),
    })
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

type LR0Automaton = Graph<FxHashSet<KernelLR0Item>, ParserSymbol>;
struct Graph<T, U> {
    nodes: Vec<T>,
    edges: Vec<Vec<(U, u32)>>,
}

#[cfg(test)]
mod tests {
    use super::ShiftReduceAction::{Accept, Error, Reduce, Shift};
    use super::{closure, lr0_automaton, slr_parsing_table, LR0Item};
    use crate::fxhashset;
    use crate::parser::conversion::flatten;
    use crate::parser::ll::{first, follow};
    use crate::parser::lr::{goto, KernelLR0Item};
    use crate::parser::tests::grammar_440;
    use crate::parser::ParserSymbol;

    #[test]
    fn closure_set() {
        let grammar = grammar_440();
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
        let grammar = grammar_440();
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
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            KernelLR0Item { rule: 0, production: 0, position: 1 },
            KernelLR0Item { rule: 1, production: 0, position: 1 }
        );
        let ic = closure(&items, &nonterminals);
        let advanced = goto(&ic, ParserSymbol::Terminal(5), &nonterminals);
        assert!(advanced.is_empty());
    }

    #[test]
    fn canonical_lr0_automaton() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let collection = lr0_automaton(&nonterminals, 0);
        // IXX number refers to the Ids in the dragon book example.
        let expected_nodes = vec![
            fxhashset!(KernelLR0Item { rule: 0, production: 0, position: 0 }), // I0
            fxhashset!(
                KernelLR0Item { rule: 0, production: 0, position: 1 },
                KernelLR0Item { rule: 1, production: 0, position: 1 }
            ), // I1
            fxhashset!(
                KernelLR0Item { rule: 2, production: 0, position: 1 },
                KernelLR0Item { rule: 1, production: 1, position: 1 }
            ), // I2
            fxhashset!(KernelLR0Item { rule: 3, production: 0, position: 1 }), // I4
            fxhashset!(KernelLR0Item { rule: 2, production: 1, position: 1 }), // I3
            fxhashset!(KernelLR0Item { rule: 3, production: 1, position: 1 }), // I5
            fxhashset!(
                KernelLR0Item { rule: 1, production: 0, position: 1 },
                KernelLR0Item { rule: 3, production: 0, position: 2 }
            ), // I8
            fxhashset!(KernelLR0Item { rule: 1, production: 0, position: 2 }), // I6
            fxhashset!(KernelLR0Item { rule: 3, production: 0, position: 3 }), // I11
            fxhashset!(
                KernelLR0Item { rule: 2, production: 0, position: 1 },
                KernelLR0Item { rule: 1, production: 0, position: 3 }
            ), // I9
            fxhashset!(KernelLR0Item { rule: 2, production: 0, position: 2 }), // I7
            fxhashset!(KernelLR0Item { rule: 2, production: 0, position: 3 }), // I10
        ];
        assert_eq!(expected_nodes, collection.nodes);
        let expected_edges = vec![
            vec![
                (ParserSymbol::Terminal(2), 3),
                (ParserSymbol::Terminal(4), 5),
                (ParserSymbol::NonTerminal(1), 1),
                (ParserSymbol::NonTerminal(2), 2),
                (ParserSymbol::NonTerminal(3), 4),
            ],
            vec![(ParserSymbol::Terminal(0), 7)],
            vec![(ParserSymbol::Terminal(1), 10)],
            vec![
                (ParserSymbol::Terminal(2), 3),
                (ParserSymbol::Terminal(4), 5),
                (ParserSymbol::NonTerminal(1), 6),
                (ParserSymbol::NonTerminal(2), 2),
                (ParserSymbol::NonTerminal(3), 4),
            ],
            vec![],
            vec![],
            vec![(ParserSymbol::Terminal(0), 7), (ParserSymbol::Terminal(3), 8)],
            vec![
                (ParserSymbol::Terminal(2), 3),
                (ParserSymbol::Terminal(4), 5),
                (ParserSymbol::NonTerminal(2), 9),
                (ParserSymbol::NonTerminal(3), 4),
            ],
            vec![],
            vec![(ParserSymbol::Terminal(1), 10)],
            vec![
                (ParserSymbol::Terminal(2), 3),
                (ParserSymbol::Terminal(4), 5),
                (ParserSymbol::NonTerminal(3), 11),
            ],
            vec![],
        ];
        assert_eq!(expected_edges, collection.edges);
    }

    #[test]
    fn slr1() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let first = first(&nonterminals);
        let follow = follow(&nonterminals, &first, 0);
        let table = slr_parsing_table(&nonterminals, 5, 0, &follow).unwrap();
        let expected_action = vec![
            vec![Error, Error, Shift(3), Error, Shift(5), Error],
            vec![Shift(7), Error, Error, Error, Error, Accept],
            vec![Reduce((1, 1)), Shift(10), Error, Reduce((1, 1)), Error, Reduce((1, 1))],
            vec![Error, Error, Shift(3), Error, Shift(5), Error],
            vec![Reduce((2, 1)), Reduce((2, 1)), Error, Reduce((2, 1)), Error, Reduce((2, 1))],
            vec![Reduce((3, 1)), Reduce((3, 1)), Error, Reduce((3, 1)), Error, Reduce((3, 1))],
            vec![Shift(7), Error, Error, Shift(8), Error, Error],
            vec![Error, Error, Shift(3), Error, Shift(5), Error],
            vec![Reduce((3, 0)), Reduce((3, 0)), Error, Reduce((3, 0)), Error, Reduce((3, 0))],
            vec![Reduce((1, 0)), Shift(10), Error, Reduce((1, 0)), Error, Reduce((1, 0))],
            vec![Error, Error, Shift(3), Error, Shift(5), Error],
            vec![Reduce((2, 0)), Reduce((2, 0)), Error, Reduce((2, 0)), Error, Reduce((2, 0))],
        ];
        let expected_goto = vec![
            vec![None, Some(1), Some(2), Some(4)],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, Some(6), Some(2), Some(4)],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, Some(9), Some(4)],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, Some(11)],
            vec![None, None, None, None],
        ];
        assert_eq!(table.action, expected_action);
        assert_eq!(table.goto, expected_goto);
    }
}
