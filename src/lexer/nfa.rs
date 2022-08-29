use super::dfa::Dfa;
use super::grammar_conversion::{canonical_trees, CanonicalTree, Literal};
use super::{Automaton, SymbolTable};
use crate::grammar::Grammar;
use fnv::{FnvHashMap, FnvHashSet};
use std::fmt::Write;

/// A Non-deterministic Finite Automaton for lexical analysis.
///
/// A NFA is an automaton that may contain ϵ productions (moves on an empty symbol) or different
/// moves for the same input symbol.
///
/// An example of NFA recognizing the language `a|b*` is the following:
///
/// ![NFA Example](../../../../doc/images/nfa.svg)
///
/// Simulating this automaton is inefficient and using a DFA is highly suggested.
#[derive(Clone)]
pub struct Nfa {
    /// Number of states.
    pub(super) states_no: u32,
    /// Transition map. (node index, symbol) -> Set(node index).
    pub(super) transition: FnvHashMap<(u32, u32), FnvHashSet<u32>>,
    /// All the symbols recognized by the NFA, except EPSILON and ANY_VALUE.
    pub(super) alphabet: SymbolTable,
    /// Starting node of the NFA.
    pub(super) start: u32,
    /// Accepting states. (node index) -> (production index)
    pub(super) accept: FnvHashMap<u32, u32>,
}

impl Nfa {
    /// Builds an NFA using the
    /// [*McNaughton-Yamada-Thompson* algorithm](https://en.wikipedia.org/wiki/Thompson%27s_construction).
    ///
    /// Note that the resulting NFA will have a lot of epsilon moves.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Nfa;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    /// ```
    pub fn new(grammar: &Grammar) -> Nfa {
        let (canonical_trees, symtable) = canonical_trees(grammar);
        if canonical_trees.is_empty() {
            // no production found, return a single state, no transaction NFA.
            Nfa {
                states_no: 1,
                transition: FnvHashMap::default(),
                alphabet: SymbolTable::default(),
                start: 0,
                accept: FnvHashMap::default(),
            }
        } else {
            let mut index = 0; //used to keep unique node indices thompson construction
            let epsilon_id = symtable.epsilon_id();
            let mut thompson_nfas = canonical_trees
                .iter()
                .enumerate()
                .map(|x| {
                    let nfa = thompson_construction(x.1, index, x.0 as u32, epsilon_id);
                    index += nfa.nodes();
                    nfa
                })
                .collect::<Vec<_>>();
            //merge productions into a single NFA by adding a new start node with epsilon moves
            // to the old start nodes
            if thompson_nfas.len() > 1 {
                let start_transition = thompson_nfas
                    .iter()
                    .map(|x| x.start)
                    .collect::<FnvHashSet<_>>();
                let accept = thompson_nfas
                    .iter()
                    .flat_map(|x| x.accept.clone())
                    .collect::<FnvHashMap<_, _>>();
                let mut transition_table = thompson_nfas
                    .into_iter()
                    .flat_map(|x| x.transition)
                    .collect::<FnvHashMap<_, _>>();
                transition_table.insert((index, epsilon_id), start_transition);
                index += 1;
                Nfa {
                    states_no: index,
                    transition: transition_table,
                    alphabet: symtable,
                    start: index - 1,
                    accept,
                }
            } else {
                let mut nfa = thompson_nfas.pop().unwrap();
                nfa.alphabet = symtable;
                nfa
            }
        }
    }

    /// Converts the NFA to a DFA.
    ///
    /// The generated DFA is always the DFA with the minimum number of states capable of recognizing
    /// the requested language.
    ///
    /// **NOTE**: This conversion uses the *Subset Construction* algorithm, which has a **very**
    /// high time complexity, `O(2^n)`. Although the average case can be handled withouth any
    /// problems, consider constructing directly a DFA for very large grammars.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Nfa;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    /// nfa.to_dfa();
    /// ```
    pub fn to_dfa(self) -> Dfa {
        Dfa::from(self)
    }
}

impl std::fmt::Display for Nfa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NFA({},{})", self.nodes(), self.edges())
    }
}

impl Automaton for Nfa {
    fn is_empty(&self) -> bool {
        self.transition.is_empty()
    }

    fn nodes(&self) -> u32 {
        self.states_no
    }

    fn edges(&self) -> u32 {
        self.transition
            .iter()
            .fold(0, |acc, x| x.1.len() as u32 + acc)
    }

    fn to_dot(&self) -> String {
        let mut f = String::new();
        write!(&mut f, "digraph{{start[shape=point];").unwrap();
        for state in &self.accept {
            write!(
                &mut f,
                "{}[shape=doublecircle;xlabel=\"ACC({})\"];",
                state.0, state.1
            )
            .unwrap();
        }
        write!(&mut f, "start->{};", &self.start).unwrap();
        for trans in &self.transition {
            for target in trans.1 {
                let source = (trans.0).0;
                let symbol_id = (trans.0).1;
                let symbol_label = self.alphabet.label(symbol_id).replace('"', "\\\"");
                write!(
                    &mut f,
                    "{}->{}[label=\"{}\"];",
                    source, target, symbol_label
                )
                .unwrap();
            }
        }
        write!(&mut f, "}}").unwrap();
        f
    }
}

/// Performs the thompson construction on a regex canonical parse tree to obtain an NFA.
///
/// - `start_index`: Starts assigning indices to NFA nodes from this number.
/// - `production`: This is the announced production index for the current parse tree.
fn thompson_construction(
    prod: &CanonicalTree,
    start_index: u32,
    production: u32,
    epsilon_id: u32,
) -> Nfa {
    let mut index = start_index;
    let mut visit = vec![prod];
    let mut todo = Vec::new();
    let mut done = Vec::<Nfa>::new();
    //first transform the parse tree into a stack, this will be the processing order
    while let Some(node) = visit.pop() {
        if let Some(l) = &node.left {
            visit.push(l);
        }
        if let Some(r) = &node.right {
            visit.push(r);
        }
        todo.push(node);
    }
    //now process every node in order, depending on its type
    while let Some(node) = todo.pop() {
        let pushme;
        match node.value {
            Literal::Symbol(val) => {
                let mut target_set = FnvHashSet::default();
                target_set.insert(index + 1);
                let mut accept = FnvHashMap::default();
                accept.insert(index + 1, production);
                let mut transition = FnvHashMap::default();
                transition.insert((index, val), target_set);
                pushme = Nfa {
                    states_no: 2,
                    transition,
                    alphabet: SymbolTable::default(),
                    start: index,
                    accept,
                };
                index += 2;
            }
            Literal::KLEENE => {
                let new_start = index;
                let new_end = index + 1;
                index += 2;
                let mut first = done.pop().unwrap();
                let mut target_set = FnvHashSet::default();
                target_set.insert(first.start);
                target_set.insert(new_end);
                for acc in first.accept {
                    first
                        .transition
                        .insert((acc.0, epsilon_id), target_set.clone());
                }
                first
                    .transition
                    .insert((new_start, epsilon_id), target_set.clone());
                first.start = new_start;
                let mut accept = FnvHashMap::default();
                accept.insert(new_end, production);
                first.accept = accept;
                first.states_no += 2;
                pushme = first;
            }
            Literal::AND => {
                let second = done.pop().unwrap();
                let mut first = done.pop().unwrap();
                first.transition.extend(second.transition);
                let mut target_set = FnvHashSet::default();
                target_set.insert(second.start);
                for acc in first.accept {
                    first
                        .transition
                        .insert((acc.0, epsilon_id), target_set.clone());
                }
                first.accept = second.accept;
                first.states_no += second.states_no;
                pushme = first;
            }
            Literal::OR => {
                let new_start = index;
                let new_end = index + 1;
                index += 2;
                let second = done.pop().unwrap();
                let mut first = done.pop().unwrap();
                let mut target_set = FnvHashSet::default();
                target_set.insert(first.start);
                target_set.insert(second.start);
                let mut accept = FnvHashMap::default();
                accept.insert(new_end, production);
                let mut target_set = FnvHashSet::default();
                target_set.insert(first.start);
                target_set.insert(second.start);
                first.transition.extend(second.transition);
                first.transition.insert((new_start, epsilon_id), target_set);
                target_set = FnvHashSet::default();
                target_set.insert(new_end);
                for acc in first.accept.into_iter().chain(second.accept.into_iter()) {
                    first
                        .transition
                        .insert((acc.0, epsilon_id), target_set.clone());
                }
                first.start = new_start;
                first.accept = accept;
                first.states_no += second.states_no + 2;
                pushme = first;
            }
            Literal::Acc(_) => panic!("Accept state not allowed in thompson construction!"),
        }
        done.push(pushme);
    }
    done.pop().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;
    use crate::lexer::{Automaton, Nfa};

    #[test]
    fn nfa_set_productions() {
        let grammar = Grammar::new(&["[a-c]([b-d]?[e-g])*", "[fg]+"], &[], &["LONG1", "LONG2"]);
        let nfa = Nfa::new(&grammar);
        assert!(!nfa.is_empty());
        assert_eq!(nfa.nodes(), 31);
        assert_eq!(nfa.edges(), 38);
    }

    #[test]
    fn nfa_manually_checked_edges() {
        let grammar = Grammar::new(&["'a''b'*"], &[], &["A BSTAR"]);
        let nfa = Nfa::new(&grammar);
        let a = nfa.alphabet.symbol_id('a');
        let b = nfa.alphabet.symbol_id('b');
        let e = nfa.alphabet.epsilon_id();
        assert_eq!(
            *nfa.transition.get(&(0, a)).unwrap().iter().next().unwrap(),
            1
        );
        assert_eq!(
            *nfa.transition.get(&(1, e)).unwrap().iter().next().unwrap(),
            4
        );
        assert_eq!(nfa.transition.get(&(4, e)).unwrap().len(), 2);
        assert_eq!(
            *nfa.transition.get(&(2, b)).unwrap().iter().next().unwrap(),
            3
        );
        assert_eq!(nfa.transition.get(&(3, e)).unwrap().len(), 2);
        assert_eq!(*nfa.accept.get(&5).unwrap(), 0);
        assert_eq!(nfa.nodes(), 6);
        assert_eq!(nfa.edges(), 7);
    }

    #[test]
    fn nfa_start_accepting() {
        let grammar = Grammar::new(&["'ab'*"], &[], &["ABSTAR"]);
        let nfa = Nfa::new(&grammar);
        assert!(!nfa.is_empty());
        assert_eq!(nfa.nodes(), 6);
        assert_eq!(nfa.edges(), 7);
    }

    #[test]
    fn nfa_construction_single_production() {
        let terminal = "(('a'*'b')|'c')?'c'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let nfa = Nfa::new(&grammar);
        assert!(!nfa.is_empty());
        assert_eq!(nfa.nodes(), 16);
        assert_eq!(nfa.edges(), 19);
    }

    #[test]
    fn nfa_construction_multi_production() {
        let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
        let nfa = Nfa::new(&grammar);
        assert!(!nfa.is_empty());
        assert_eq!(nfa.nodes(), 7);
        assert_eq!(nfa.edges(), 8);
    }

    #[test]
    fn nfa_empty() {
        let grammar = Grammar::new(&[], &[], &[]);
        let nfa = Nfa::new(&grammar);
        assert!(nfa.is_empty());
    }
}
