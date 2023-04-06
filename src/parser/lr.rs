use super::conversion::flatten;
use super::ll::{first, follow};
use super::{ENDLINE_VAL, EPSILON_VAL};
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
    first: Vec<FxHashSet<u32>>,
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
            first,
            follow,
        })
    }
}

impl LRGrammar {
    /// Computes the LR(0) parsing table for the given grammar.
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
    /// let lr0_table = grammar_lr.lr0_parsing_table().unwrap();
    /// ```
    pub fn lr0_parsing_table(&self) -> Result<LRParsingTable, ParseError> {
        lr_parsing_table::<0>(
            &self.nonterminals,
            self.token_names.len() as u32,
            self.starting_rule,
            &self.first,
        )
    }

    /// Computes the LR(1) parsing table for the given grammar.
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
    /// let lr1_table = grammar_lr.lr1_parsing_table().unwrap();
    /// ```
    pub fn lr1_parsing_table(&self) -> Result<LRParsingTable, ParseError> {
        lr_parsing_table::<1>(
            &self.nonterminals,
            self.token_names.len() as u32,
            self.starting_rule,
            &self.first,
        )
    }

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

/// Expands a list of LR<k> kernel items.
/// `first` can be empty if K = 0
fn closure<const K: usize>(
    items: &FxHashSet<KernelLRItem<K>>,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    first: &[FxHashSet<u32>],
) -> FxHashSet<LRItem<K>> {
    assert!(K < 2);
    let mut set = items
        .iter()
        .map(|x| LRItem::from(*x))
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
                    .map(|(prod_id, _)| LRItem::nonkernel(nt, prod_id as u16, [0; K]))
                    .collect::<Vec<_>>();
                if K == 1 {
                    let lookaheads = match item.next().peek(nonterminals) {
                        ParserSymbol::Terminal(t) => fxhashset!(t),
                        ParserSymbol::NonTerminal(nt) => first[nt as usize].clone(),
                        ParserSymbol::Empty => fxhashset!(EPSILON_VAL),
                    }
                    .into_iter()
                    .map(|x| {
                        if x == EPSILON_VAL {
                            item.lookahead()[0]
                        } else {
                            x
                        }
                    });
                    for lookahead in lookaheads {
                        let new_items_with_lookaheads = new_items.iter().copied().map(|mut n| {
                            n.lookahead_mut()[0] = lookahead;
                            n
                        });
                        to_add.extend(new_items_with_lookaheads);
                    }
                } else {
                    to_add.extend(new_items.into_iter());
                }
            } else {
                // do nothing for items like A -> · or terminals
            }
        }
        set.extend(to_add);
        new_len = set.len();
    }
    set
}

/// Advances a closure of LR<k> kernel items and returns only the kernel items
/// composing the new list.
fn goto<const K: usize>(
    items: &FxHashSet<LRItem<K>>,
    symbol: ParserSymbol,
    nonterminals: &[Vec<Vec<ParserSymbol>>],
) -> FxHashSet<KernelLRItem<K>> {
    let mut advanced = FxHashSet::default();
    for item in items {
        if item.peek(nonterminals) == symbol {
            advanced.insert(item.next());
        }
    }
    advanced
        .into_iter()
        .filter_map(|x| match x {
            LRItem::Kernel(k) => Some(k),
            LRItem::NonKernel(_) => None,
        })
        .collect()
}

/// Calculates the DFA representing the LR0 automaton.
/// Sometimes called canonical lr0 collection.
/// `first` can be empty if K = 0
fn lr_automaton<const K: usize>(
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    first: &[FxHashSet<u32>],
    start: u32,
) -> LRAutomaton<K> {
    let start_kernel = fxhashset!(KernelLRItem::<K> {
        rule: start,
        production: 0,
        position: 0,
        lookahead: [ENDLINE_VAL; K],
    });
    let mut nodes = vec![start_kernel];
    let mut edges = vec![vec![]];
    let mut done = fxhashset!();
    let mut todo = vec![0];
    while let Some(set_id) = todo.pop() {
        done.insert(set_id);
        let kernel = &nodes[set_id];
        let clos = closure(kernel, nonterminals, first);
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
    LRAutomaton { nodes, edges }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
        self.action.get(0).map(|x| x.len() - 1).unwrap_or(0) as u32
    }
}

/// builds an LR(K) parsing table
fn lr_parsing_table<const K: usize>(
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    term_no: u32,
    start: u32,
    first: &[FxHashSet<u32>],
) -> Result<LRParsingTable, ParseError> {
    let automaton = lr_automaton::<K>(nonterminals, first, start);
    let mut action = Vec::with_capacity(automaton.nodes.len());
    let mut goto = Vec::with_capacity(automaton.nodes.len());
    for (node, edge) in automaton.nodes.iter().zip(automaton.edges.iter()) {
        let mut action_entry = vec![ShiftReduceAction::Error; (term_no + 1) as usize];
        let mut goto_entry = vec![None; nonterminals.len() - 1];
        // fill shift
        for (symbol, target) in edge {
            match symbol {
                ParserSymbol::Terminal(t) => {
                    action_entry[*t as usize] = ShiftReduceAction::Shift(*target)
                }
                ParserSymbol::NonTerminal(nt) => {
                    // reindex goto to avoid the start symbol
                    let index = if *nt > start { *nt - 1 } else { *nt };
                    goto_entry[index as usize] = Some(*target)
                }
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
                    let reduce_condition = if K == 1 {
                        // convert lookahead from ENDLINE_VAL to the table index for $
                        item.lookahead
                            .into_iter()
                            .map(|x| if x == ENDLINE_VAL { term_no } else { x })
                            .collect::<Vec<_>>()
                    } else {
                        (0..=term_no).collect()
                    };
                    for term in reduce_condition {
                        // term_no is ENDLINE_VAL
                        let current_action = &mut action_entry[term as usize];
                        match current_action {
                            ShiftReduceAction::Shift(_) => {
                                return Err(ParseError::LRError {
                                    message: format!(
                                        "SHIFT/REDUCE conflict in LR({}) generation",
                                        K
                                    ),
                                })
                            }
                            ShiftReduceAction::Reduce(_) => {
                                return Err(ParseError::LRError {
                                    message: format!(
                                        "SHIFT/REDUCE conflict in LR({}) generation",
                                        K
                                    ),
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
    let automaton = lr_automaton::<0>(nonterminals, &[], start);
    let mut action = Vec::with_capacity(automaton.nodes.len());
    let mut goto = Vec::with_capacity(automaton.nodes.len());
    for (node, edge) in automaton.nodes.iter().zip(automaton.edges.iter()) {
        let mut action_entry = vec![ShiftReduceAction::Error; (term_no + 1) as usize];
        // -1 because augmented production is not in the goto table.
        let mut goto_entry = vec![None; nonterminals.len() - 1];
        // fill shift
        for (symbol, target) in edge {
            match symbol {
                ParserSymbol::Terminal(t) => {
                    action_entry[*t as usize] = ShiftReduceAction::Shift(*target)
                }
                ParserSymbol::NonTerminal(nt) => {
                    // reindex goto to avoid the start symbol
                    let index = if *nt > start { *nt - 1 } else { *nt };
                    goto_entry[index as usize] = Some(*target)
                }
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
struct KernelLRItem<const K: usize> {
    rule: u32,
    production: u16,
    position: u16,
    lookahead: [u32; K],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct NonKernelLRItem<const K: usize> {
    rule: u32,
    production: u16,
    lookahead: [u32; K],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum LRItem<const K: usize> {
    Kernel(KernelLRItem<K>),
    NonKernel(NonKernelLRItem<K>),
}

impl<const K: usize> LRItem<K> {
    fn kernel(rule: u32, production: u16, position: u16, lookahead: [u32; K]) -> Self {
        Self::Kernel(KernelLRItem {
            rule,
            production,
            position,
            lookahead,
        })
    }

    fn nonkernel(rule: u32, production: u16, lookahead: [u32; K]) -> Self {
        Self::NonKernel(NonKernelLRItem { rule, production, lookahead })
    }

    #[must_use]
    fn next(&self) -> Self {
        Self::Kernel(KernelLRItem {
            rule: self.rule(),
            production: self.production(),
            position: self.position() + 1,
            lookahead: self.lookahead(),
        })
    }

    // returns the next symbol. ParserSymbol::Empty in case of A -> ·
    fn peek(&self, nonterminals: &[Vec<Vec<ParserSymbol>>]) -> ParserSymbol {
        let (r, p, i) = match self {
            LRItem::Kernel(k) => (k.rule as usize, k.production as usize, k.position as usize),
            LRItem::NonKernel(nk) => (nk.rule as usize, nk.production as usize, 0),
        };
        nonterminals[r][p]
            .get(i)
            .copied()
            .unwrap_or(ParserSymbol::Empty)
    }

    fn rule(&self) -> u32 {
        match self {
            LRItem::Kernel(k) => k.rule,
            LRItem::NonKernel(nk) => nk.rule,
        }
    }

    fn production(&self) -> u16 {
        match self {
            LRItem::Kernel(k) => k.production,
            LRItem::NonKernel(nk) => nk.production,
        }
    }

    fn position(&self) -> u16 {
        match self {
            LRItem::Kernel(k) => k.position,
            LRItem::NonKernel(_) => 0,
        }
    }

    fn lookahead(&self) -> [u32; K] {
        match self {
            LRItem::Kernel(k) => k.lookahead,
            LRItem::NonKernel(nk) => nk.lookahead,
        }
    }

    fn lookahead_mut(&mut self) -> &mut [u32; K] {
        match self {
            LRItem::Kernel(k) => &mut k.lookahead,
            LRItem::NonKernel(nk) => &mut nk.lookahead,
        }
    }
}

impl<const K: usize> From<KernelLRItem<K>> for LRItem<K> {
    fn from(value: KernelLRItem<K>) -> Self {
        LRItem::Kernel(value)
    }
}

type LRAutomaton<const K: usize> = Graph<FxHashSet<KernelLRItem<K>>, ParserSymbol>;
struct Graph<T, U> {
    nodes: Vec<T>,
    edges: Vec<Vec<(U, u32)>>,
}

#[cfg(test)]
mod tests {
    use super::ShiftReduceAction::{Accept, Error, Reduce, Shift};
    use super::{
        closure, lr_parsing_table, slr_parsing_table, LRItem, LRParsingTable, ENDLINE_VAL,
    };
    use crate::fxhashset;
    use crate::parser::conversion::flatten;
    use crate::parser::ll::{first, follow};
    use crate::parser::lr::{goto, KernelLRItem};
    use crate::parser::tests::{grammar_440, grammar_454, grammar_458};
    use crate::parser::ParserSymbol;

    #[test]
    fn closure_set_lookahead0() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(KernelLRItem::<0> {
            rule: 0,
            production: 0,
            position: 0,
            lookahead: []
        });
        let closure = closure(&items, nonterminals.as_slice(), &[]);
        let expected = fxhashset!(
            LRItem::<0>::kernel(0, 0, 0, []),
            LRItem::<0>::nonkernel(1, 0, []),
            LRItem::<0>::nonkernel(1, 1, []),
            LRItem::<0>::nonkernel(2, 0, []),
            LRItem::<0>::nonkernel(2, 1, []),
            LRItem::<0>::nonkernel(3, 0, []),
            LRItem::<0>::nonkernel(3, 1, []),
        );
        assert_eq!(closure, expected);
    }

    #[test]
    fn closure_set_lookahead1() {
        let grammar = grammar_454();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(KernelLRItem::<1> {
            rule: 0,
            production: 0,
            position: 0,
            lookahead: [ENDLINE_VAL]
        });
        let first = first(&nonterminals);
        let closure = closure(&items, nonterminals.as_slice(), &first);
        let expected = fxhashset!(
            LRItem::<1>::kernel(0, 0, 0, [ENDLINE_VAL]),
            LRItem::<1>::nonkernel(1, 0, [ENDLINE_VAL]),
            LRItem::<1>::nonkernel(2, 0, [0]),
            LRItem::<1>::nonkernel(2, 0, [1]),
            LRItem::<1>::nonkernel(2, 1, [0]),
            LRItem::<1>::nonkernel(2, 1, [1]),
        );
        assert_eq!(closure, expected);
    }

    #[test]
    fn goto_function_lookahead0() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            KernelLRItem::<0> {
                rule: 0,
                production: 0,
                position: 1,
                lookahead: []
            },
            KernelLRItem::<0> {
                rule: 1,
                production: 0,
                position: 1,
                lookahead: []
            }
        );
        let ic = closure(&items, &nonterminals, &[]);
        let advanced = goto(&ic, ParserSymbol::Terminal(0), &nonterminals);
        let expected = fxhashset!(KernelLRItem::<0> {
            rule: 1,
            production: 0,
            position: 2,
            lookahead: []
        },);
        assert_eq!(advanced, expected);
    }

    #[test]
    fn goto_empty_lookahead0() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let items = fxhashset!(
            KernelLRItem::<0> {
                rule: 0,
                production: 0,
                position: 1,
                lookahead: []
            },
            KernelLRItem::<0> {
                rule: 1,
                production: 0,
                position: 1,
                lookahead: []
            }
        );
        let ic = closure(&items, &nonterminals, &[]);
        let advanced = goto(&ic, ParserSymbol::Terminal(5), &nonterminals);
        assert!(advanced.is_empty());
    }

    // parsing table state ID is non deterministic, so to keep a decently
    // maintainable test the parsing tables are compared in the type of entries
    // only (i.e. state number is dropped and it is checked that n row should
    // have k number of shifts, etc.)
    fn compare_tables(mut result: LRParsingTable, mut expected: LRParsingTable) {
        assert_eq!(result.action.len(), expected.action.len());
        assert_eq!(result.goto.len(), expected.goto.len());
        for row in expected.action.iter_mut() {
            for action in row.iter_mut() {
                match action {
                    Shift(_) => *action = Shift(0),
                    Reduce(_) => *action = Reduce((0, 0)),
                    Error => (),
                    Accept => (),
                }
            }
        }
        for row in expected.goto.iter_mut() {
            for action in row.iter_mut().flatten() {
                *action = 0;
            }
        }
        expected.goto.sort();
        expected.action.sort();
        for row in result.action.iter_mut() {
            for action in row.iter_mut() {
                match action {
                    Shift(_) => *action = Shift(0),
                    Reduce(_) => *action = Reduce((0, 0)),
                    Error => (),
                    Accept => (),
                }
            }
        }
        for row in result.goto.iter_mut() {
            for action in row.iter_mut().flatten() {
                *action = 0;
            }
        }
        result.goto.sort();
        result.action.sort();
        assert_eq!(result.action, expected.action);
        assert_eq!(result.goto, expected.goto);
    }

    #[test]
    fn lr0() {
        let grammar = grammar_454();
        let nonterminals = flatten(&grammar).unwrap();
        let table = lr_parsing_table::<0>(&nonterminals, 2, 0, &[]).unwrap();
        let expected = LRParsingTable {
            action: vec![
                vec![Shift(1), Shift(2), Error],
                vec![Shift(1), Shift(2), Error],
                vec![Reduce((2, 1)), Reduce((2, 1)), Reduce((2, 1))],
                vec![Error, Error, Accept],
                vec![Shift(1), Shift(2), Error],
                vec![Reduce((1, 0)), Reduce((1, 0)), Reduce((1, 0))],
                vec![Reduce((2, 0)), Reduce((2, 0)), Reduce((2, 0))],
            ],
            goto: vec![
                vec![Some(3), Some(4)],
                vec![None, Some(6)],
                vec![None, None],
                vec![None, None],
                vec![None, Some(5)],
                vec![None, None],
                vec![None, None],
            ],
            nonterminals,
        };
        compare_tables(table, expected);
    }

    #[test]
    fn slr1() {
        let grammar = grammar_440();
        let nonterminals = flatten(&grammar).unwrap();
        let first = first(&nonterminals);
        let follow = follow(&nonterminals, &first, 0);
        let table = slr_parsing_table(&nonterminals, 5, 0, &follow).unwrap();
        let expected = LRParsingTable {
            action: vec![
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
            ],
            goto: vec![
                vec![Some(1), Some(2), Some(4)],
                vec![None, None, None],
                vec![None, None, None],
                vec![Some(6), Some(2), Some(4)],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, Some(9), Some(4)],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, Some(11)],
                vec![None, None, None],
            ],
            nonterminals,
        };
        compare_tables(table, expected);
    }

    #[test]
    fn lr1() {
        let grammar = grammar_458();
        let nonterminals = flatten(&grammar).unwrap();
        let first = first(&nonterminals);
        let table = lr_parsing_table::<1>(&nonterminals, 5, 0, &first).unwrap();
        let expected = LRParsingTable {
            action: vec![
                vec![Shift(2), Shift(3), Error, Error, Error, Error],
                vec![Error, Error, Error, Error, Error, Accept],
                vec![Error, Error, Shift(6), Error, Error, Error],
                vec![Error, Error, Shift(9), Error, Error, Error],
                vec![Error, Error, Error, Shift(10), Error, Error],
                vec![Error, Error, Error, Error, Shift(11), Error],
                vec![Error, Error, Error, Reduce((2, 0)), Reduce((3, 0)), Error],
                vec![Error, Error, Error, Shift(12), Error, Error],
                vec![Error, Error, Error, Error, Shift(13), Error],
                vec![Error, Error, Error, Reduce((3, 0)), Reduce((2, 0)), Error],
                vec![Error, Error, Error, Error, Error, Reduce((1, 0))],
                vec![Error, Error, Error, Error, Error, Reduce((1, 1))],
                vec![Error, Error, Error, Error, Error, Reduce((1, 2))],
                vec![Error, Error, Error, Error, Error, Reduce((1, 3))],
            ],
            goto: vec![
                vec![Some(1), None, None],
                vec![None, None, None],
                vec![None, Some(4), Some(5)],
                vec![None, Some(8), Some(9)],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
                vec![None, None, None],
            ],
            nonterminals,
        };
        compare_tables(table, expected);
    }
}
