use super::grammar_conversion::{canonical_trees, CanonicalTree, Literal};
use super::nfa::Nfa;
use super::{Automaton, BSTree, SymbolTable};
use crate::grammar::Grammar;
use fnv::{FnvHashMap, FnvHashSet};
use maplit::btreeset;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

/// A Deterministic Finite Automaton for lexical analysis.
///
/// A DFA is an automaton where each state has a single transaction for a given input symbol, and
/// no transactions on empty symbols (ϵ-moves).
///
/// An example of DFA recognizing the language `a|b*` is the following:
///
/// ![DFA Example](../../../../doc/images/dfa.svg)
#[derive(Clone)]
pub struct Dfa {
    /// Number of states.
    states_no: u32,
    /// Transition function: (node index, symbol) -> (node).
    transition: FnvHashMap<(u32, u32), u32>,
    /// Set of symbols in the language.
    alphabet: SymbolTable,
    /// Starting node.
    start: u32,
    /// Accepting states: (node index) -> (accepted production).
    accept: FnvHashMap<u32, u32>,
}

impl std::fmt::Display for Dfa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DFA({},{})", self.nodes(), self.edges())
    }
}

impl Dfa {
    /// Constructs a DFA given an input grammar.
    ///
    /// The DFA is constructed directly from the regex parse tree without using an intermediate NFA.
    ///
    /// The generated DFA has the minimum number of states required to recognized the requested
    /// language.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Dfa;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let dfa = Dfa::new(&grammar);
    /// ```
    pub fn new(grammar: &Grammar) -> Dfa {
        let (canonical_trees, symtable) = canonical_trees(grammar);
        if canonical_trees.is_empty() {
            // no production found, return a single state, no transaction DFA.
            Dfa {
                states_no: 1,
                transition: FnvHashMap::default(),
                alphabet: SymbolTable::default(),
                start: 0,
                accept: FnvHashMap::default(),
            }
        } else {
            // merge all trees of every production into a single one
            let merged_tree = merge_regex_trees(canonical_trees);
            // build the dfa
            let big_dfa = direct_construction(merged_tree, symtable);
            // minimize the dfa
            min_dfa(big_dfa)
        }
    }
}

impl Automaton for Dfa {
    fn is_empty(&self) -> bool {
        self.transition.is_empty()
    }

    fn nodes(&self) -> u32 {
        self.states_no
    }

    fn edges(&self) -> u32 {
        self.transition.len() as u32
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
            let source = (trans.0).0;
            let symbol_id = (trans.0).1;
            let symbol_label = self.alphabet.label(symbol_id).replace('"', "\\\"");
            write!(
                &mut f,
                "{}->{}[label=\"{}\"];",
                source, trans.1, symbol_label
            )
            .unwrap();
        }
        write!(&mut f, "}}").unwrap();
        f
    }
}

impl From<Nfa> for Dfa {
    fn from(nfa: Nfa) -> Self {
        let big_dfa = subset_construction(&nfa);
        min_dfa(big_dfa)
    }
}

/// Transforms a NFA to a DFA using subset construction algorithm.
///
/// Guaranteed to have a move on every symbol for every node.
fn subset_construction(nfa: &Nfa) -> Dfa {
    let mut ds_marked = BTreeSet::new();
    let mut ds_unmarked = Vec::new();
    let mut indices = HashMap::new();
    let mut index = 0;
    let mut transition = FnvHashMap::default();
    let mut accept = FnvHashMap::default();

    let mut start_set = FnvHashSet::default();
    start_set.insert(nfa.start);
    let s0 = sc_epsilon_closure(start_set, &nfa.transition, nfa.alphabet.epsilon_id());
    indices.insert(s0.clone(), index);
    // possible acceptance check is done at creation time
    if let Some(accepted_production) = sc_accepting(&s0, &nfa.accept) {
        accept.insert(index, accepted_production);
    }
    ds_unmarked.push(s0);
    index += 1;
    while let Some(t) = ds_unmarked.pop() {
        let t_idx = *indices.get(&t).unwrap();
        ds_marked.insert(t.clone());
        for symbol in nfa.alphabet.iter() {
            let sym = *symbol.1;
            let mov = sc_move(&t, sym, &nfa.transition);
            let u = sc_epsilon_closure(mov, &nfa.transition, nfa.alphabet.epsilon_id());
            let u_idx;
            if !u.is_empty() {
                //check if node has already been created
                if !indices.contains_key(&u) {
                    u_idx = index;
                    indices.insert(u.clone(), u_idx);
                    index += 1;
                    // check if state is accepting
                    if let Some(accepted_production) = sc_accepting(&u, &nfa.accept) {
                        accept.insert(u_idx, accepted_production);
                    }
                    ds_unmarked.push(u);
                } else {
                    u_idx = *indices.get(&u).unwrap();
                }
                transition.insert((t_idx, sym), u_idx);
            }
        }
    }
    //add sink (it's not guaranteed to have a sink and REQUIRED by the min_dfa function)
    //accidentally adding a second sink is no problem: it will be removed by min_dfa function
    let sink = index;
    index += 1;
    for node in 0..index {
        for symbol in nfa.alphabet.iter() {
            transition.entry((node, *symbol.1)).or_insert(sink);
        }
    }
    Dfa {
        states_no: index,
        alphabet: nfa.alphabet.clone(),
        transition,
        accept,
        start: 0,
    }
}

/// Part of the subset construction algorithm:
///
/// Returns the set of nodes reachable with epsilon moves from the current set.
///
/// - `set`: the input set where to start the epsilon move
/// - `transition`: transition table of the NFA
fn sc_epsilon_closure(
    set: FnvHashSet<u32>,
    transition_table: &FnvHashMap<(u32, u32), FnvHashSet<u32>>,
    epsilon_id: u32,
) -> BTreeSet<u32> {
    let mut stack = set.iter().copied().collect::<Vec<_>>();
    let mut closure = set;
    while let Some(t) = stack.pop() {
        if let Some(eset) = transition_table.get(&(t, epsilon_id)) {
            closure = closure.union(eset).cloned().collect();
            stack.extend(eset.iter());
        }
    }
    closure.into_iter().collect::<BTreeSet<_>>()
}

/// Part of the subset construction algorithm:
///
/// Returns Some(production index) if the current set is accepting, None otherwise. Will return
/// the lowest production index possible (production appearing first).
fn sc_accepting(set: &BTreeSet<u32>, accepting: &FnvHashMap<u32, u32>) -> Option<u32> {
    let mut productions = BTreeSet::new(); //so I can easily get the min()
    for node in accepting {
        if set.contains(node.0) {
            productions.insert(node.1);
        }
    }
    if !productions.is_empty() {
        Some(**productions.iter().next().unwrap()) //get smallest value (production appearing first)
    } else {
        None
    }
}

/// Part of the subset construction algorithm:
///
/// Returns the set of nodes reachable with a move on a given symbol from the current set.
///
/// - `set`: The input set
/// - `symbol`: The given symbol for the move
/// - `transition`: The transition table
fn sc_move(
    set: &BTreeSet<u32>,
    symbol: u32,
    transition: &FnvHashMap<(u32, u32), FnvHashSet<u32>>,
) -> FnvHashSet<u32> {
    let mut ret = FnvHashSet::default();
    for node in set {
        if let Some(t) = transition.get(&(*node, symbol)) {
            ret = ret.union(t).cloned().collect::<FnvHashSet<_>>();
        }
    }
    ret
}

/// Merges different canonical trees into a single canonical tree with multiple accepting nodes.
/// Accepting states are labeled with a new node in the canonical tree.
fn merge_regex_trees(nodes: Vec<CanonicalTree>) -> CanonicalTree {
    let mut roots = Vec::new();
    for (node_no, node) in nodes.into_iter().enumerate() {
        let right = BSTree {
            value: Literal::Acc(node_no as u32),
            left: None,
            right: None,
        };
        let root = BSTree {
            value: Literal::AND,
            left: Some(Box::new(node)),
            right: Some(Box::new(right)),
        };
        roots.push(root);
    }
    while roots.len() > 1 {
        let right = roots.pop().unwrap();
        let left = roots.pop().unwrap();
        let new_root = BSTree {
            value: Literal::OR,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        };
        roots.push(new_root);
    }
    roots.pop().unwrap()
}

/// Helper for the direct construction of the DFA.
struct DCHelper {
    /// Literal type (need to know if kleenee or AND in the followpos computation)
    /// and the followpos is deferred from the firstpos, nullable, lastpos.
    ttype: Literal,
    ///
    index: u32,
    /// true if the current node is nullable.
    nullable: bool,
    /// firstpos for the current set (has an average size of 22 on the C grammar so BTreeSet is
    /// faster than HashSet).
    firstpos: BTreeSet<u32>,
    /// lastpos for the current set (has an average size of 22 on the C grammar so BTreeSet is
    /// faster than HashSet).
    lastpos: BTreeSet<u32>,
}

/// Performs a direct DFA construction from the canonical tree without using an intermediate NFA.
/// Refers to "Compilers, principle techniques and tools" of A.Aho et al. (p.179 on 2nd edition).
///
/// Guaranteed to have a move on every symbol for every node.
fn direct_construction(node: CanonicalTree, symtable: SymbolTable) -> Dfa {
    let helper = build_dc_helper(&node, 0, symtable.epsilon_id());
    let mut indices = vec![symtable.epsilon_id(); (helper.value.index + 1) as usize];
    let mut followpos = vec![BTreeSet::new(); (helper.value.index + 1) as usize];
    //retrieve accepting nodes (they are embedded in the helper tree, we don't have NFA here)
    let mut accepting_nodes = FnvHashMap::default();
    dc_assign_index_to_literal(&helper, &mut indices, &mut accepting_nodes);
    dc_compute_followpos(&helper, &mut followpos);
    let mut accept = FnvHashMap::default();
    let mut done = HashMap::new();
    done.insert(helper.value.firstpos.clone(), 0);
    // check the first node if it can be accepting, this is done in the loop at creation time.
    // pick the production with the lowest index in the same group.
    if let Some(acc_prod) = helper
        .value
        .firstpos
        .iter()
        .flat_map(|x| accepting_nodes.get(x))
        .min()
    {
        accept.insert(0, *acc_prod);
    }
    let mut index = 1;
    let mut unmarked = vec![helper.value.firstpos];
    let mut tran = FnvHashMap::default();
    let alphabet = indices
        .iter()
        .filter(|x| **x != symtable.epsilon_id())
        .copied()
        .collect::<FnvHashSet<u32>>();
    // loop, conceptually similar to subset construction, but uses followpos instead of NFA
    // (followpos is essentially an NFA without epsilon moves)
    while let Some(node_set) = unmarked.pop() {
        for symbol in &alphabet {
            let u = node_set
                .iter()
                .filter(|x| indices[**x as usize] == *symbol)
                .flat_map(|x| &followpos[*x as usize])
                .cloned()
                .collect::<BTreeSet<_>>();
            let u_idx;
            if let Some(got) = done.get(&u) {
                u_idx = *got;
            } else {
                u_idx = index;
                index += 1;
                if let Some(acc_prod) = u.iter().flat_map(|x| accepting_nodes.get(x)).min() {
                    accept.insert(u_idx, *acc_prod);
                }
                unmarked.push(u.clone());
                done.insert(u, u_idx);
            }
            let set_idx = *done.get(&node_set).unwrap();
            tran.insert((set_idx, *symbol), u_idx);
        }
    }
    Dfa {
        states_no: index,
        start: 0,
        transition: tran,
        alphabet: symtable,
        accept,
    }
}

/// Part of the direct DFA construction:
///
/// Computes `firstpos`, `lastpos` and `nullpos` for the parse tree.
///
/// - `node`: the root of the tree (it's a recursive function)
/// - `start_index`: starting index of the output DFA (0 for the first invocation)
fn build_dc_helper(node: &CanonicalTree, start_index: u32, epsilon_id: u32) -> BSTree<DCHelper> {
    //postorder because I need to build it bottom up
    let mut index = start_index;
    let mut children = [&node.left, &node.right]
        .iter()
        .map(|x| match x {
            Some(c) => {
                let helper = build_dc_helper(c, index, epsilon_id);
                index = helper.value.index + 1;
                Some(Box::new(helper))
            }
            None => None,
        })
        .collect::<Vec<_>>();
    let right = children.pop().unwrap();
    let left = children.pop().unwrap();
    let nullable;
    let firstpos;
    let lastpos;
    match &node.value {
        Literal::Symbol(val) => {
            if *val == epsilon_id {
                nullable = true;
                firstpos = BTreeSet::new();
                lastpos = BTreeSet::new();
            } else {
                nullable = false;
                firstpos = btreeset! {index};
                lastpos = btreeset! {index};
            }
        }
        Literal::KLEENE => {
            let c1 = &left.as_ref().unwrap().value;
            nullable = true;
            firstpos = c1.firstpos.clone();
            lastpos = c1.lastpos.clone();
        }
        Literal::AND => {
            let c1 = &left.as_ref().unwrap().value;
            let c2 = &right.as_ref().unwrap().value;
            nullable = c1.nullable && c2.nullable;
            firstpos = if c1.nullable {
                c1.firstpos.union(&c2.firstpos).cloned().collect()
            } else {
                c1.firstpos.clone()
            };
            lastpos = if c2.nullable {
                c1.lastpos.union(&c2.lastpos).cloned().collect()
            } else {
                c2.lastpos.clone()
            };
        }
        Literal::OR => {
            let c1 = &left.as_ref().unwrap().value;
            let c2 = &right.as_ref().unwrap().value;
            nullable = c1.nullable || c2.nullable;
            firstpos = c1.firstpos.union(&c2.firstpos).cloned().collect();
            lastpos = c1.lastpos.union(&c2.lastpos).cloned().collect();
        }
        Literal::Acc(_) => {
            nullable = false;
            firstpos = btreeset! {index};
            lastpos = btreeset! {index};
        }
    }
    BSTree {
        value: DCHelper {
            ttype: node.value,
            index,
            nullable,
            firstpos,
            lastpos,
        },
        left,
        right,
    }
}

/// Part of the direct DFA construction:
///
/// Computes `followpos` set. Each index of the output vector is the index of the DFA node, and the
/// content of that cell is the followpos set.
fn dc_compute_followpos(node: &BSTree<DCHelper>, graph: &mut Vec<BTreeSet<u32>>) {
    if let Some(l) = &node.left {
        dc_compute_followpos(l, graph);
    }
    if let Some(r) = &node.right {
        dc_compute_followpos(r, graph);
    }
    match &node.value.ttype {
        Literal::Symbol(_) => {}
        Literal::Acc(_) => {}
        Literal::OR => {}
        Literal::AND => {
            let c1 = &**node.left.as_ref().unwrap();
            let c2 = &**node.right.as_ref().unwrap();
            for i in &c1.value.lastpos {
                graph[*i as usize] = graph[*i as usize]
                    .union(&c2.value.firstpos)
                    .cloned()
                    .collect();
            }
        }
        Literal::KLEENE => {
            for i in &node.value.lastpos {
                graph[*i as usize] = graph[*i as usize]
                    .union(&node.value.firstpos)
                    .cloned()
                    .collect();
            }
        }
    }
}
/// Part of the direct DFA construction:
///
/// Assigns an unique index to each node of the parse tree (required by the algorithm) and records
/// the production number for each accepting node.
fn dc_assign_index_to_literal(
    node: &BSTree<DCHelper>,
    indices: &mut Vec<u32>,
    acc: &mut FnvHashMap<u32, u32>,
) {
    if let Some(l) = &node.left {
        dc_assign_index_to_literal(l, indices, acc);
    }
    if let Some(r) = &node.right {
        dc_assign_index_to_literal(r, indices, acc);
    }
    match &node.value.ttype {
        Literal::Symbol(val) => indices[node.value.index as usize] = *val,
        Literal::Acc(prod) => {
            acc.insert(node.value.index, *prod);
        }
        _ => {}
    }
}

/// Given a DFA returns the DFA with the minimum number of nodes.
///
/// **REQUIRED** to have a move on every symbol for every node.
///
/// Again the source of this algorithm is "Compilers, principle techniques and tools" of
/// A.Aho et al. (p.180 on 2nd edition).
fn min_dfa(dfa: Dfa) -> Dfa {
    let mut partitions = init_partitions(&dfa);
    let mut positions = FnvHashMap::default();
    for (partition_index, partition) in partitions.iter().enumerate() {
        for node in partition {
            positions.insert(*node, partition_index as u32);
        }
    }
    while partitions.len() < dfa.states_no as usize {
        let mut old_partitions = Vec::new();
        let mut new_partitions = Vec::new();
        for partition in partitions {
            let split = split_partition(partition, &positions, &dfa);
            old_partitions.push(split.0);
            if !split.1.is_empty() {
                new_partitions.push(split.1);
            }
        }
        if new_partitions.is_empty() {
            partitions = old_partitions;
            break;
        } else {
            //reindex positions
            for (new_idx, partition) in new_partitions.iter().enumerate() {
                for node in partition {
                    positions.remove(node);
                    positions.insert(*node, (old_partitions.len() + new_idx) as u32);
                }
            }
            old_partitions.append(&mut new_partitions);
            partitions = old_partitions;
        }
    }
    remap(partitions, positions, dfa)
}

/// Part of the min DFA algorithm:
///
/// Creates the initial partitions: non accepting nodes, and a partition for each group of accepting
/// nodes announcing the same rule.
fn init_partitions(dfa: &Dfa) -> Vec<FnvHashSet<u32>> {
    let mut announced_max = std::u32::MIN;
    let mut acc = FnvHashSet::default();
    for (announcing_state, announced_rule) in &dfa.accept {
        acc.insert(*announcing_state);
        announced_max = announced_max.max(*announced_rule);
    }
    let accepting_no = announced_max + 1;
    let nacc = (0..)
        .take(dfa.states_no as usize)
        .collect::<FnvHashSet<_>>()
        .difference(&acc)
        .cloned()
        .collect::<FnvHashSet<_>>();
    // this is a DFA for lexical analysis so I need to further split acc by announced rule
    let mut ret = vec![FnvHashSet::default(); accepting_no as usize];
    for (announcing_state, announced_rule) in &dfa.accept {
        ret[(*announced_rule) as usize].insert(*announcing_state);
    }
    ret.push(nacc); //add the non_accepting partition to the end
    ret
}

/// Part of the min DFA algorithm:
///
/// Splits a partition if two nodes goes to different partitions on the same symbol.
fn split_partition(
    partition: FnvHashSet<u32>,
    position: &FnvHashMap<u32, u32>,
    dfa: &Dfa,
) -> (FnvHashSet<u32>, FnvHashSet<u32>) {
    let mut split = FnvHashSet::default();
    if partition.len() > 1 {
        for symbol in dfa.alphabet.iter() {
            let mut iter = partition.iter();
            let first = *iter.next().unwrap();
            let expected_target = *position
                .get(dfa.transition.get(&(first, *symbol.1)).unwrap())
                .unwrap();
            for node in iter {
                let target = *position
                    .get(dfa.transition.get(&(*node, *symbol.1)).unwrap())
                    .unwrap();
                if target != expected_target {
                    split.insert(*node);
                }
            }
            if !split.is_empty() {
                break;
            }
        }
        (partition.difference(&split).cloned().collect(), split)
    } else {
        (partition, split)
    }
}

/// Given the final set of partitions rewrites the transition table in order to get the efficient
/// one.
///
/// Also, removes the sink, if any and not accepting.
fn remap(partitions: Vec<FnvHashSet<u32>>, positions: FnvHashMap<u32, u32>, dfa: Dfa) -> Dfa {
    //first record in which partitions is every node
    let mut new_trans = FnvHashMap::default();
    let mut accept = FnvHashMap::default();
    let mut in_degree = vec![0; partitions.len()];
    let mut out_degree = vec![0; partitions.len()];
    //remap accepting nodes
    for acc_node in dfa.accept {
        accept.insert(*positions.get(&acc_node.0).unwrap(), acc_node.1);
    }
    //remap transitions
    for transition in dfa.transition {
        let old_source = (transition.0).0;
        let new_source = *positions.get(&old_source).unwrap();
        let letter = (transition.0).1;
        let old_target = transition.1;
        let new_target = *positions.get(&old_target).unwrap();
        if new_source != new_target {
            out_degree[new_source as usize] += 1;
            in_degree[new_target as usize] += 1;
        }
        new_trans.insert((new_source, letter), new_target);
    }
    // start = partition of previous start
    let start = *positions.get(&0).unwrap();
    // remove unreachable states (and non-accepting sinks)
    //broken: no in-edges and no start state OR no out-edges and not accepting, excluding self-loops
    let broken_states = (0_u32..)
        .take(partitions.len())
        .filter(|x| {
            (in_degree[*x as usize] == 0 && *x != start)
                || (out_degree[*x as usize] == 0 && !accept.contains_key(x))
        })
        .collect::<BTreeSet<_>>();
    new_trans = new_trans
        .into_iter()
        .filter(|x| !broken_states.contains(&(x.0).0) && !broken_states.contains(&x.1))
        .collect();
    Dfa {
        states_no: (partitions.len() - broken_states.len()) as u32,
        transition: new_trans,
        alphabet: dfa.alphabet,
        accept,
        start,
    }
}

#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;
    use crate::lexer::{Automaton, Dfa, Nfa};

    #[test]
    fn dfa_conflicts_resolution() {
        //they should be different: the second accept abb as a*b+ (appearing first in the productions)
        let grammar1 = Grammar::new(
            &["'a'", "'abb'", "'a'*'b'+"],
            &[],
            &["A", "ABB", "ASTARBPLUS"],
        );
        let dfa1 = Dfa::new(&grammar1);
        let grammar2 = Grammar::new(
            &["'a'*'b'+", "'abb'", "'a'"],
            &[],
            &["ASTARBPLUS", "ABB", "A"],
        );
        let dfa2 = Dfa::new(&grammar2);
        assert!(!dfa1.is_empty());
        assert_eq!(dfa1.nodes(), 6);
        assert_eq!(dfa1.edges(), 9);
        assert!(!dfa2.is_empty());
        assert_eq!(dfa2.nodes(), 4);
        assert_eq!(dfa2.edges(), 7);
    }

    #[test]
    fn dfa_direct_construction_no_sink() {
        let terminal = "('a'|'b')*'abb'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 4);
        assert_eq!(dfa.edges(), 8);
    }

    #[test]
    fn dfa_direct_construction_sink_accepting() {
        let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 3);
        assert_eq!(dfa.edges(), 3);
    }

    #[test]
    fn dfa_direct_construction_set_productions() {
        let grammar = Grammar::new(&["[a-c]([b-d]?[e-g])*", "[fg]+"], &[], &["LONG1", "LONG2"]);
        let dfa = Dfa::new(&grammar);
        assert_eq!(dfa.nodes(), 4);
        assert_eq!(dfa.edges(), 10);
    }

    #[test]
    fn dfa_direct_construction_start_accepting() {
        let grammar = Grammar::new(&["'ab'*"], &[], &["ABSTAR"]);
        let dfa_direct = Dfa::new(&grammar);
        assert!(!dfa_direct.is_empty());
        assert_eq!(dfa_direct.nodes(), 2);
        assert_eq!(dfa_direct.edges(), 2);
    }

    #[test]
    fn dfa_direct_construction_single_acc() {
        let terminal = "(('a'*'b')|'c')?'c'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 5);
        assert_eq!(dfa.edges(), 7);
    }

    #[test]
    fn dfa_direct_construction_multi_production() {
        let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 3);
        assert_eq!(dfa.edges(), 3);
    }

    #[test]
    fn dfa_direct_construction_empty() {
        let grammar = Grammar::new(&[], &[], &[]);
        let dfa = Dfa::new(&grammar);
        assert!(dfa.is_empty());
    }

    #[test]
    fn dfa_subset_construction_no_sink() {
        let terminal = "('a'|'b')*'abb'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert_eq!(dfa.nodes(), 4);
        assert_eq!(dfa.edges(), 8);
    }

    #[test]
    fn dfa_subset_construction_sink_accepting() {
        let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert_eq!(dfa.nodes(), 3);
        assert_eq!(dfa.edges(), 3);
    }

    #[test]
    fn dfa_subset_construction_set_productions() {
        let grammar = Grammar::new(&["[a-c]([b-d]?[e-g])*", "[fg]+"], &[], &["LONG1", "LONG2"]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert_eq!(dfa.nodes(), 4);
        assert_eq!(dfa.edges(), 10);
    }

    #[test]
    fn dfa_subset_construction_multi_production() {
        let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert_eq!(dfa.nodes(), 3);
        assert_eq!(dfa.edges(), 3);
    }

    #[test]
    fn dfa_subset_construction_start_accepting() {
        let grammar = Grammar::new(&["'ab'*"], &[], &["ABSTAR"]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 2);
        assert_eq!(dfa.edges(), 2);
    }

    #[test]
    fn dfa_subset_construction_single_production() {
        let terminal = "(('a'*'b')|'c')?'c'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.to_dfa();
        assert_eq!(dfa.nodes(), 5);
        assert_eq!(dfa.edges(), 7);
    }

    #[test]
    fn dfa_subset_construction_empty() {
        let grammar = Grammar::new(&[], &[], &[]);
        let nfa = Nfa::new(&grammar);
        let dfa = nfa.clone().to_dfa();
        assert!(nfa.is_empty());
        assert!(dfa.is_empty());
    }
}
