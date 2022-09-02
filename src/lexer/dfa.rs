use super::grammar_conversion::{canonical_trees, CanonicalTree, Literal};
use super::{BSTree, SymbolTable};
use crate::error::ParseError;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    /// Number of states.
    pub(super) states_no: u32,
    /// Transition function: [node][symbol] -> node.
    pub(super) transition: Vec<Vec<u32>>,
    /// Set of symbols in the language.
    pub(super) alphabet: SymbolTable,
    /// Starting node.
    pub(super) start: u32,
    /// Sink node.
    pub(super) sink: u32,
    /// Accepted production for each node. u32::MAX if the node is not accepting.
    pub(super) accept: Vec<u32>,
}

impl std::fmt::Display for Dfa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DFA({})", self.nodes())
    }
}

impl Default for Dfa {
    fn default() -> Self {
        Self {
            states_no: 1,
            start: 0,
            sink: 0,
            transition: vec![vec![0]],
            accept: vec![u32::MAX],
            alphabet: SymbolTable::default(),
        }
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
            // no production found, return default DFA
            Dfa::default()
        } else {
            // merge all trees of every production into a single one
            let merged_tree = merge_regex_trees(canonical_trees);
            // build the dfa
            let big_dfa = direct_construction(merged_tree, symtable);
            // minimize the dfa
            min_dfa(big_dfa)
        }
    }

    /// Serialize the current DFA into a vector of bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut retval = Vec::new();
        let symtable = self.alphabet.as_bytes();
        retval.extend(u32::to_le_bytes(self.states_no));
        retval.extend(u32::to_le_bytes(self.start));
        retval.extend(u32::to_le_bytes(self.sink));
        retval.extend(u32::to_le_bytes(symtable.len() as u32));
        retval.extend(symtable);
        // single loop for both transition table and accept table.
        for node in 0..self.states_no {
            for symbol in 0..self.alphabet.ids() {
                retval.extend(u32::to_le_bytes(
                    self.transition[node as usize][symbol as usize],
                ));
            }
            retval.extend(u32::to_le_bytes(self.accept[node as usize]));
        }
        retval
    }

    /// Deserialize a DFA from a slice of bytes.
    pub fn from_bytes(v: &[u8]) -> Result<Self, ParseError> {
        let malformed_err = "malformed dfa";
        let parse_u32 = |i: &mut usize| -> Result<u32, ParseError> {
            let bytes: [u8; 4] = v
                .get(*i..*i + 4)
                .ok_or_else(|| ParseError::DeserializeError {
                    message: malformed_err.to_string(),
                })?
                .try_into()
                .map_err(|_| ParseError::DeserializeError {
                    message: malformed_err.to_string(),
                })?;
            *i += 4;
            Ok(u32::from_le_bytes(bytes))
        };
        let mut i = 0;
        let states_no = parse_u32(&mut i)?;
        let start = parse_u32(&mut i)?;
        let sink = parse_u32(&mut i)?;
        let symtable_len = parse_u32(&mut i)? as usize;
        let symtable = SymbolTable::from_bytes(&v[i..i + symtable_len])?;
        i += symtable_len;
        let mut transition = Vec::with_capacity(states_no as usize);
        let mut accept = Vec::with_capacity(states_no as usize);
        for _ in 0..states_no {
            let mut moves = Vec::with_capacity(symtable.ids() as usize);
            for _ in 0..symtable.ids() {
                moves.push(parse_u32(&mut i)?);
            }
            transition.push(moves);
            accept.push(parse_u32(&mut i)?);
        }
        Ok(Dfa {
            states_no,
            transition,
            alphabet: symtable,
            start,
            sink,
            accept,
        })
    }

    /// Returns true if the DFA is empty.
    ///
    /// A DFA is empty if there are no transitions, and, as such, it halts in the starting
    /// state.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Dfa;
    ///
    /// let grammar = Grammar::new(&[], &[], &[]);
    /// let dfa = Dfa::new(&grammar);
    ///
    /// assert!(dfa.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.transition.len() <= 1
    }

    /// Returns the number of nodes in the DFA, excluding the eventual sink node.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Dfa;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let dfa = Dfa::new(&grammar);
    ///
    /// assert_eq!(dfa.nodes(), 3);
    /// ```
    pub fn nodes(&self) -> u32 {
        self.states_no - 1
    }

    /// Returns a graphviz dot representation of the automaton as string.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::Dfa;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let dfa = Dfa::new(&grammar);
    /// dfa.to_dot();
    /// ```
    pub fn to_dot(&self) -> String {
        let mut f = String::new();
        write!(&mut f, "digraph{{start[shape=point];").unwrap();
        for (state, accepted_rule) in self.accept.iter().enumerate() {
            if *accepted_rule != u32::MAX {
                write!(
                    &mut f,
                    "{}[shape=doublecircle;xlabel=\"ACC({})\"];",
                    state, accepted_rule
                )
                .unwrap();
            }
        }
        write!(&mut f, "start->{};", &self.start).unwrap();
        for state in 0..self.states_no {
            for symbol in 0..self.alphabet.ids() {
                let dst = self.transition[state as usize][symbol as usize];
                if dst != self.sink {
                    let symbol_label = self.alphabet.label(symbol).replace('"', "\\\"");
                    write!(&mut f, "{}->{}[label=\"{}\"];", state, dst, symbol_label).unwrap();
                }
            }
        }
        write!(&mut f, "}}").unwrap();
        f
    }
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
    let mut accept_map = FnvHashMap::default();
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
        accept_map.insert(0, *acc_prod);
    }
    let mut index = 1;
    let mut unmarked = vec![helper.value.firstpos];
    let mut tran = FnvHashMap::default();
    // loop, conceptually similar to subset construction, but uses followpos instead of NFA
    // (followpos is essentially an NFA without epsilon moves)
    while let Some(node_set) = unmarked.pop() {
        for symbol in 0..symtable.ids() {
            let u = node_set
                .iter()
                .filter(|x| indices[**x as usize] == symbol)
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
                    accept_map.insert(u_idx, *acc_prod);
                }
                unmarked.push(u.clone());
                done.insert(u, u_idx);
            }
            let set_idx = *done.get(&node_set).unwrap();
            tran.insert((set_idx, symbol), u_idx);
        }
    }
    // converts the transition map to a transition table and marks the sink
    // the transition is not complete as
    let mut transition = Vec::with_capacity(index as usize);
    let mut sink = 0;
    for node in 0..index {
        let ids_no = symtable.ids() as usize;
        let mut next_nodes = Vec::with_capacity(ids_no);
        let mut all_same = FnvHashSet::with_capacity_and_hasher(ids_no, Default::default());
        for symbol in 0..symtable.ids() as u32 {
            let next = *tran.get(&(node, symbol)).unwrap_or(&sink);
            next_nodes.push(next);
            all_same.insert(next);
        }
        if all_same.len() == 1 && *all_same.iter().next().unwrap() == node {
            sink = node;
        }
        transition.push(next_nodes);
    }
    // converts the accepting states map into array
    let mut accept = Vec::with_capacity(index as usize);
    for i in 0..index {
        match accept_map.entry(i) {
            std::collections::hash_map::Entry::Occupied(val) => accept.push(*val.get()),
            std::collections::hash_map::Entry::Vacant(_) => accept.push(u32::MAX),
        }
    }
    Dfa {
        states_no: index,
        start: 0,
        transition,
        alphabet: symtable,
        accept,
        sink,
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
    // find how many announcing_rules exists
    let announced_max = dfa
        .accept
        .iter()
        .copied()
        .filter(|&acc| acc != u32::MAX)
        .max()
        .unwrap_or(0) as usize;
    // creates the various sets (+2 because last element is the non-accepting nodes)
    let mut ret = vec![FnvHashSet::default(); announced_max + 2];
    for (announcing_state, announced_rule) in dfa.accept.iter().enumerate() {
        if *announced_rule != u32::MAX {
            ret[(*announced_rule) as usize].insert(announcing_state as u32);
        } else {
            ret[announced_max + 1].insert(announcing_state as u32);
        }
    }
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
        for symbol_id in 0..dfa.alphabet.ids() {
            let mut iter = partition.iter();
            let first = *iter.next().unwrap();
            let expected_target = *position
                .get(&dfa.transition[first as usize][symbol_id as usize])
                .unwrap();
            for node in iter {
                let target = *position
                    .get(&dfa.transition[*node as usize][symbol_id as usize])
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
fn remap(partitions: Vec<FnvHashSet<u32>>, positions: FnvHashMap<u32, u32>, dfa: Dfa) -> Dfa {
    //first record in which partitions is every node
    let mut new_trans = FnvHashMap::default();
    let mut accept_map = FnvHashMap::default();
    let mut in_degree = vec![0; partitions.len()];
    let mut out_degree = vec![0; partitions.len()];
    //remap accepting nodes
    for (acc_node, acc_rule) in dfa.accept.iter().enumerate() {
        if *acc_rule != u32::MAX {
            accept_map.insert(*positions.get(&(acc_node as u32)).unwrap(), *acc_rule);
        }
    }
    //remap transitions
    for node in 0..dfa.states_no {
        for symbol in 0..dfa.alphabet.ids() {
            let new_src = *positions.get(&node).unwrap();
            let old_dst = dfa.transition[node as usize][symbol as usize];
            let new_dst = *positions.get(&old_dst).unwrap();
            if new_src != new_dst {
                out_degree[new_src as usize] += 1;
                in_degree[new_dst as usize] += 1;
            }
            new_trans.insert((new_src, symbol), new_dst);
        }
    }
    let start = *positions.get(&dfa.start).unwrap();
    let sink = *positions.get(&dfa.sink).unwrap();
    // remove unreachable states (and non-accepting sinks)
    //broken: no in-edges and no start state OR no out-edges and not accepting, excluding self-loops
    //        and sink
    let broken_states = (0_u32..)
        .take(partitions.len())
        .filter(|x| {
            (in_degree[*x as usize] == 0 && *x != start)
                || (out_degree[*x as usize] == 0 && !accept_map.contains_key(x) && *x != sink)
        })
        .collect::<BTreeSet<_>>();
    let mut remapped_indices = vec![0; dfa.states_no as usize];
    let mut new_index = 0_u32;
    for index in 0..dfa.states_no {
        if !broken_states.contains(&index) {
            remapped_indices[index as usize] = new_index;
            new_index += 1;
        }
    }
    // now that the number of states is fixed it's time to change map to table.
    // Unlike the direct consturction, this time we need to reindex nodes.
    let states_no = (partitions.len() - broken_states.len()) as u32;
    let mut transition = Vec::with_capacity(states_no as usize);
    for node in 0..dfa.states_no {
        if !broken_states.contains(&node) {
            let ids_no = dfa.alphabet.ids() as usize;
            let mut next_nodes = Vec::with_capacity(ids_no);
            for symbol in 0..dfa.alphabet.ids() as u32 {
                let next = *new_trans.get(&(node, symbol)).unwrap_or(&sink);
                if !broken_states.contains(&next) {
                    let next_remapped = remapped_indices[next as usize];
                    next_nodes.push(next_remapped);
                }
            }
            transition.push(next_nodes);
        }
    }
    // converts the accepting states map into array
    let mut accept = Vec::with_capacity(states_no as usize);
    for i in 0..states_no {
        match accept_map.entry(i) {
            std::collections::hash_map::Entry::Occupied(val) => accept.push(*val.get()),
            std::collections::hash_map::Entry::Vacant(_) => accept.push(u32::MAX),
        }
    }
    Dfa {
        states_no,
        transition,
        alphabet: dfa.alphabet,
        accept,
        start,
        sink,
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ParseError;
    use crate::grammar::Grammar;
    use crate::lexer::dfa::min_dfa;
    use crate::lexer::grammar_conversion::canonical_trees;
    use crate::lexer::Dfa;

    use super::{direct_construction, merge_regex_trees};

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
        assert!(!dfa2.is_empty());
        assert_eq!(dfa2.nodes(), 4);
    }

    #[test]
    fn dfa_direct_construction_no_sink() {
        let terminal = "('a'|'b')*'abb'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 4);
    }

    #[test]
    fn dfa_direct_construction_sink_accepting() {
        let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digit", "more_digits"]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 3);
    }

    #[test]
    fn dfa_direct_construction_set_productions() {
        let grammar = Grammar::new(&["[a-c]([b-d]?[e-g])*", "[fg]+"], &[], &["LONG1", "LONG2"]);
        let dfa = Dfa::new(&grammar);
        assert_eq!(dfa.nodes(), 4);
    }

    #[test]
    fn dfa_direct_construction_start_accepting() {
        let grammar = Grammar::new(&["'ab'*"], &[], &["ABSTAR"]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 2);
    }

    #[test]
    fn dfa_direct_construction_single_acc() {
        let terminal = "(('a'*'b')|'c')?'c'";
        let names = "PROD1";
        let grammar = Grammar::new(&[terminal], &[], &[names]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 5);
    }

    #[test]
    fn dfa_direct_construction_multi_production() {
        let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
        let dfa = Dfa::new(&grammar);
        assert!(!dfa.is_empty());
        assert_eq!(dfa.nodes(), 3);
    }

    #[test]
    fn dfa_direct_construction_empty() {
        let grammar = Grammar::new(&[], &[], &[]);
        let dfa = Dfa::new(&grammar);
        assert!(dfa.is_empty());
    }

    #[test]
    fn dfa_minimization() {
        let grammar = Grammar::new(
            &["('00'|'11')*(('01'|'10')('00'|'11')*('01'|'10')('00'|'11')*)*"],
            &[],
            &["SEQUENCE"],
        );
        let (canonical_trees, symtable) = canonical_trees(&grammar);
        let merged_tree = merge_regex_trees(canonical_trees);
        let big_dfa = direct_construction(merged_tree, symtable);
        let min_dfa = min_dfa(big_dfa.clone());
        assert_ne!(big_dfa.nodes(), min_dfa.nodes());
        assert_eq!(min_dfa.nodes(), 4);
    }

    #[test]
    fn dfa_serialization() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &["[a-c].?([b-d]?[e-g])*", "[fg]+"],
            &[],
            &["LONG1", "LONG2"],
        );
        let dfa = Dfa::new(&grammar);
        let serialized = dfa.as_bytes();
        let deserialized = Dfa::from_bytes(&serialized)?;
        assert_eq!(dfa, deserialized);
        Ok(())
    }
}
