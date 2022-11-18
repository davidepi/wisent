use super::grammar_conversion::{canonicalise, parse_trees, CanonicalTree, Literal};
use super::{GraphvizDot, SymbolTable, Tree};
use crate::grammar::Grammar;
use maplit::btreeset;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;
use std::ops::Index;

/// Bit indicating a non-greedy production
const NG_FLAG: u32 = 0x80000000;

///
/// # Modes
/// This struct support a multiple modes (context), by wrapping together multiple transtion tables.
/// For this reason, every move in the DFA requires specifying the current mode. The default mode
/// has always index 0.
///
/// If the mode does not exist in the current DFA, the function will panic. This is expected
/// behaviour by design: the DFA is supposed to be used by a simulator, and invocations with wrong
/// modes should *NOT* happen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiDfa {
    /// Contains the various transition tables, one for each mode.
    /// At least one with index 0 is guaranteed to exist.
    tts: Vec<Dfa>,
    /// Associates an unique ID to each read lexeme. Every ID is then associated with a transition.
    symtable: SymbolTable,
}

impl Default for MultiDfa {
    fn default() -> Self {
        Self {
            tts: vec![Dfa::default()],
            symtable: Default::default(),
        }
    }
}

impl Index<usize> for MultiDfa {
    type Output = Dfa;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tts[index]
    }
}

impl MultiDfa {
    /// Constructs the DFAs for a given Grammar.
    ///
    /// The DFAs are constructed directly from the regex parse tree without using an
    /// intermediate NFA.
    ///
    /// The generated DFAs have the minimum number of states required to recognized the requested
    /// language.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::MultiDfa;
    /// let grammar = Grammar::new(
    ///     &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'*").into()],
    ///     &[],
    /// );
    /// let dfa = MultiDfa::new(&grammar);
    /// ```
    pub fn new(grammar: &Grammar) -> Self {
        let modes = grammar.len_modes();
        let mut trees = Vec::with_capacity(modes);
        let mut symtables = Vec::with_capacity(modes);
        let mut tts = Vec::with_capacity(modes);
        for mode in grammar.iter_modes() {
            let terminals = grammar.iter_term_in_mode(mode);
            let (parse_tree, symtable, nongreedy) = parse_trees(terminals);
            trees.push((parse_tree, nongreedy));
            symtables.push(symtable);
        }
        let joined_symtable = SymbolTable::join(&symtables);
        for (parse_trees, nongreedy) in trees {
            let canonical_trees = parse_trees
                .into_iter()
                .map(|pt| canonicalise(pt, &joined_symtable))
                .collect::<Vec<_>>();
            let tt = Dfa::new(canonical_trees, &joined_symtable, nongreedy);
            tts.push(tt);
        }
        Self {
            tts,
            symtable: joined_symtable,
        }
    }

    /// Returns the amount of nodes in this DFA.
    pub fn modes(&self) -> usize {
        self.tts.len()
    }

    /// Returns the Dfa for the given mode, None if the mode does not exist.
    pub fn dfa(&self, mode: usize) -> Option<&Dfa> {
        self.tts.get(mode)
    }

    /// Returns the symbol table associated with the DFAs in this struct.
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symtable
    }
}

/// A Deterministic Finite Automaton for lexical analysis.
///
/// A DFA is an automaton where each state has a single transaction for a given input symbol, and
/// no transactions on empty symbols (ϵ-moves).
///
/// In this crate, a DFA can not be constructed directly, but must be interfaced by a [`MultiDfa`]
/// to properly support multi-modes lexers (e.g. ANTLR modes or flex contextes).
///
/// An example of DFA recognizing the language `a|b+` is the following:
///
/// ![DFA Example](../../../../doc/images/dfa.svg)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dfa {
    /// Number of states.
    states_no: u32,
    /// Transition function: (node,symbol) -> node.
    transition: Vec<Vec<u32>>,
    /// Starting node.
    start: u32,
    /// Sink node.
    sink: u32,
    /// Accepted production for each node. u32::MAX if the node is not accepting.
    accept: Vec<u32>,
}

impl Default for Dfa {
    fn default() -> Self {
        Self {
            states_no: 1,
            start: 0,
            sink: 0,
            transition: vec![vec![0]],
            accept: vec![u32::MAX],
        }
    }
}

impl Dfa {
    /// Creates a new transition table from the given canonical trees (one per production) and
    /// symbol table. The vector index correspond to the accepted production index.
    /// The nongreedy array specifies if each production is nongreedy.
    fn new(
        canonical_trees: Vec<CanonicalTree>,
        symtable: &SymbolTable,
        nongreedy: Vec<bool>,
    ) -> Dfa {
        if canonical_trees.is_empty() {
            // no production found, return default DFA
            Dfa::default()
        } else {
            // merge all trees of every production into a single one
            let merged_tree = merge_regex_trees(canonical_trees);
            // build the dfa
            let big_dfa = direct_construction(merged_tree, symtable);
            // minimize the dfa
            let mut dfa = min_dfa(symtable, big_dfa);
            // mark nongreedy productions
            for state in 0..dfa.states_no {
                if let Some(prod) = dfa.accepting(state) {
                    if nongreedy[prod as usize] {
                        dfa.accept[state as usize] |= NG_FLAG;
                    }
                }
            }
            dfa
        }
    }

    /// Returns true if the DFA is empty.
    ///
    /// A DFA is empty if there are no transitions, and, as such, it halts in the starting
    /// state.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::MultiDfa;
    /// const DEFAULT_MODE: usize = 0;
    /// let grammar = Grammar::empty();
    /// let mdfa = MultiDfa::new(&grammar);
    /// let dfa = mdfa.dfa(DEFAULT_MODE).unwrap();
    ///
    /// assert!(dfa.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.transition.len() <= 1
    }

    /// Returns the number of states in the DFA, excluding the eventual sink state.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::MultiDfa;
    /// const DEFAULT_MODE: usize = 0;
    /// let grammar = Grammar::new(
    ///     &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'*").into()],
    ///     &[],
    /// );
    /// let mdfa = MultiDfa::new(&grammar);
    /// let dfa = mdfa.dfa(DEFAULT_MODE).unwrap();
    ///
    /// assert_eq!(dfa.states(), 3);
    /// ```
    pub fn states(&self) -> u32 {
        self.states_no - 1
    }

    /// Return the initial state for this DFA.
    pub fn start(&self) -> u32 {
        self.start
    }

    /// Perform a move in the transition table of this DFA and returns the next state.
    ///
    /// Returns None if such a move is not possible
    ///
    /// # Panics
    /// Panics if the symbol is not contained in [`Dfa::symbol_table`] or the given state does not
    /// exist in the DFA.
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::MultiDfa;
    /// const DEFAULT_MODE: usize = 0;
    /// let grammar = Grammar::new(
    ///     &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'").into()],
    ///     &[],
    /// );
    /// let mdfa = MultiDfa::new(&grammar);
    /// let dfa = mdfa.dfa(DEFAULT_MODE).unwrap();
    /// let a_id = mdfa.symbol_table().symbol_id('a');
    /// let next = dfa.moove(dfa.start(), a_id);
    /// assert!(next.is_some());
    /// ```
    pub fn moove(&self, state: u32, symbol: u32) -> Option<u32> {
        let next = self.transition[state as usize][symbol as usize];
        if next != self.sink {
            Some(next)
        } else {
            None
        }
    }

    /// Returns the accepted production if the given state is an accepting one.
    ///
    /// Returns None otherwise.
    /// # Panics
    /// Panics if the given state does not exist in the DFA.
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::MultiDfa;
    /// const DEFAULT_MODE: usize = 0;
    /// let grammar = Grammar::new(
    ///     &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'").into()],
    ///     &[],
    /// );
    /// let mdfa = MultiDfa::new(&grammar);
    /// let dfa = mdfa.dfa(DEFAULT_MODE).unwrap();
    /// assert!(dfa.accepting(dfa.start()).is_none());
    /// let a_id = mdfa.symbol_table().symbol_id('a');
    /// let next = dfa.moove(dfa.start(), a_id).unwrap();
    /// assert!(dfa.accepting(next).is_some());
    /// ```
    pub fn accepting(&self, state: u32) -> Option<u32> {
        let prod = self.accept[state as usize];
        if prod == u32::MAX {
            None
        } else {
            Some(prod & !NG_FLAG)
        }
    }

    /// Returns true if the current state is a non-greedy production or not-accepting.
    ///
    /// Returns false if the current state is a greedy production.
    ///
    /// # Panics
    /// Panics if the given state does not exist in the DFA.
    pub fn non_greedy(&self, state: u32) -> bool {
        let prod = self.accept[state as usize];
        (prod & NG_FLAG) != 0
    }
}

impl GraphvizDot for MultiDfa {
    fn to_dot(&self) -> String {
        let mut f = String::new();
        for (i, dfa) in self.tts.iter().enumerate() {
            writeln!(f, "digraph DFA{} {{\n    start[shape=point];", i).unwrap();
            for state in 0..dfa.states_no {
                if let Some(accepted_rule) = dfa.accepting(state) {
                    let acc_label = if dfa.non_greedy(state) {
                        "ACC_NG"
                    } else {
                        "ACC"
                    };
                    writeln!(
                        f,
                        "    {}[shape=doublecircle;xlabel=\"{}({})\"];",
                        state, acc_label, accepted_rule
                    )
                    .unwrap();
                }
            }
            writeln!(f, "    start->{};", &dfa.start).unwrap();
            for state in 0..dfa.states_no {
                // group labels together
                let mut transitions = vec![String::new(); dfa.states_no as usize];
                for symbol in 0..self.symtable.ids() {
                    let dst = dfa.transition[state as usize][symbol as usize];
                    if dst != dfa.sink {
                        let symbol_label = self
                            .symtable
                            .label(symbol)
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\t', "<TAB>")
                            .replace('\n', "<LF>")
                            .replace('\r', "<CR>")
                            .replace(' ', "<SPACE>");
                        transitions[dst as usize].push_str(&symbol_label);
                    }
                }
                // print grouped labels
                for dst in 0..dfa.states_no {
                    let label = &transitions[dst as usize];
                    if !label.is_empty() {
                        // !empty = not sink
                        writeln!(f, "    {}->{}[label=\"{}\"];", state, dst, label).unwrap();
                    }
                }
            }
            f.push('}');
            f.push('\n');
        }
        f
    }
}

/// Merges different canonical trees into a single canonical tree with multiple accepting nodes.
/// Accepting states are labeled with a new node in the canonical tree.
fn merge_regex_trees(nodes: Vec<CanonicalTree>) -> CanonicalTree {
    // for each regex assign an acceptance node with ID
    let mut roots = nodes
        .into_iter()
        .enumerate()
        .map(|(node_no, root)| {
            Tree::new_node(
                Literal::AND,
                vec![root, Tree::new_leaf(Literal::Acc(node_no as u32))],
            )
        })
        .collect::<Vec<_>>();
    if roots.len() > 1 {
        Tree::new_node(Literal::OR, roots)
    } else {
        roots.pop().unwrap()
    }
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
fn direct_construction(node: CanonicalTree, symtable: &SymbolTable) -> Dfa {
    let helper = build_dc_helper(&node, 0, symtable.epsilon_id());
    let mut indices = vec![symtable.epsilon_id(); (helper.value.index + 1) as usize];
    let mut followpos = vec![BTreeSet::new(); (helper.value.index + 1) as usize];
    //retrieve accepting nodes (they are embedded in the helper tree, we don't have NFA here)
    let mut accepting_nodes = FxHashMap::default();
    dc_assign_index_to_literal(&helper, &mut indices, &mut accepting_nodes);
    dc_compute_followpos(&helper, &mut followpos);
    let mut accept_map = FxHashMap::default();
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
    let mut tran = FxHashMap::default();
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
        let mut all_same = FxHashSet::with_capacity_and_hasher(ids_no, Default::default());
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
fn build_dc_helper(node: &CanonicalTree, start_index: u32, epsilon_id: u32) -> Tree<DCHelper> {
    //postorder because I need to build it bottom up
    let mut index = start_index;
    let children = node
        .children()
        .map(|c| {
            let helper = build_dc_helper(c, index, epsilon_id);
            index = helper.value.index + 1;
            helper
        })
        .collect::<Vec<_>>();
    let nullable;
    let mut firstpos;
    let mut lastpos;
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
            let c1 = &children[0].value;
            nullable = true;
            firstpos = c1.firstpos.clone();
            lastpos = c1.lastpos.clone();
        }
        Literal::AND => {
            nullable = children.iter().all(|c| c.value().nullable);
            firstpos = Default::default();
            lastpos = Default::default();
            // the firstpos computation is valid only for children == 2
            // otherwise followpos needs to recalculate them (for children c2, c3..) and it is ugly
            debug_assert!(children.len() == 2, "AND node can have up to 2 children");
            for child in &children {
                firstpos.extend(child.value().firstpos.iter().cloned());
                if !child.value().nullable {
                    break;
                }
            }
            for child in children.iter().rev() {
                lastpos.extend(child.value().lastpos.iter().cloned());
                if !child.value().nullable {
                    break;
                }
            }
        }
        Literal::OR => {
            nullable = children.iter().any(|c| c.value().nullable);
            firstpos = Default::default();
            lastpos = Default::default();
            for child in &children {
                firstpos.extend(child.value().firstpos.iter().cloned());
                lastpos.extend(child.value().lastpos.iter().cloned());
            }
        }
        Literal::Acc(_) => {
            nullable = false;
            firstpos = btreeset! {index};
            lastpos = btreeset! {index};
        }
    }
    let new_value = DCHelper {
        ttype: node.value,
        index,
        nullable,
        firstpos,
        lastpos,
    };
    Tree::new_node(new_value, children)
}

/// Part of the direct DFA construction:
///
/// Computes `followpos` set. Each index of the `graph` vector is the index of the DFA node, and the
/// content of that cell is the followpos set.
fn dc_compute_followpos(node: &Tree<DCHelper>, graph: &mut Vec<BTreeSet<u32>>) {
    node.children().for_each(|c| dc_compute_followpos(c, graph));
    match &node.value.ttype {
        Literal::Symbol(_) => {}
        Literal::Acc(_) => {}
        Literal::OR => {}
        Literal::AND => {
            debug_assert!(
                node.children().count() == 2,
                "AND node can have up to 2 children"
            );
            for i in &node.children[0].value.lastpos {
                graph[*i as usize].extend(node.children[1].value.firstpos.iter().copied());
            }
        }
        Literal::KLEENE => {
            for i in &node.value.lastpos {
                graph[*i as usize].extend(node.value.firstpos.iter().copied());
            }
        }
    }
}
/// Part of the direct DFA construction:
///
/// Assigns an unique index to each node of the parse tree (required by the algorithm) and records
/// the production number for each accepting node.
fn dc_assign_index_to_literal(
    node: &Tree<DCHelper>,
    indices: &mut Vec<u32>,
    acc: &mut FxHashMap<u32, u32>,
) {
    node.children()
        .for_each(|c| dc_assign_index_to_literal(c, indices, acc));
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
fn min_dfa(symtable: &SymbolTable, dfa: Dfa) -> Dfa {
    let mut partitions = init_partitions(&dfa);
    let mut positions = FxHashMap::default();
    for (partition_index, partition) in partitions.iter().enumerate() {
        for node in partition {
            positions.insert(*node, partition_index as u32);
        }
    }
    while partitions.len() < dfa.states_no as usize {
        let mut old_partitions = Vec::new();
        let mut new_partitions = Vec::new();
        for partition in partitions {
            let split = split_partition(partition, &positions, symtable, &dfa);
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
    remap(partitions, positions, symtable, dfa)
}

/// Part of the min DFA algorithm:
///
/// Creates the initial partitions: non accepting nodes, and a partition for each group of accepting
/// nodes announcing the same rule.
fn init_partitions(dfa: &Dfa) -> Vec<FxHashSet<u32>> {
    // find how many announcing_rules exists
    let announced_max = dfa
        .accept
        .iter()
        .copied()
        .filter(|&acc| acc != u32::MAX)
        .max()
        .unwrap_or(0) as usize;
    // creates the various sets (+2 because last element is the non-accepting nodes)
    let mut ret = vec![FxHashSet::default(); announced_max + 2];
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
    partition: FxHashSet<u32>,
    position: &FxHashMap<u32, u32>,
    symtable: &SymbolTable,
    dfa: &Dfa,
) -> (FxHashSet<u32>, FxHashSet<u32>) {
    let mut split = FxHashSet::default();
    if partition.len() > 1 {
        for symbol_id in 0..symtable.ids() {
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
fn remap(
    partitions: Vec<FxHashSet<u32>>,
    positions: FxHashMap<u32, u32>,
    symtable: &SymbolTable,
    dfa: Dfa,
) -> Dfa {
    //first record in which partitions is every node
    let mut new_trans = FxHashMap::default();
    let mut accept_map = FxHashMap::default();
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
        for symbol in 0..symtable.ids() {
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
    let mut start = *positions.get(&dfa.start).unwrap();
    // remove unreachable states (and non-accepting sinks)
    // broken: no in-edges and no start OR no out-edges and not accepting, excluding self-loops
    // this removes also the sink but that will be readded later
    let broken_partitions = (0_u32..)
        .take(partitions.len())
        .filter(|x| {
            (in_degree[*x as usize] == 0 && *x != start)
                || (out_degree[*x as usize] == 0 && !accept_map.contains_key(x))
        })
        .collect::<BTreeSet<_>>();
    // index remapping required because some states will be removed
    let mut remapped_indices = vec![0; partitions.len()];
    let mut new_index = 0_u32;
    for index in 0..partitions.len() as u32 {
        if !broken_partitions.contains(&index) {
            remapped_indices[index as usize] = new_index;
            new_index += 1;
        }
    }
    // now that the number of states is fixed it's time to change map to table.
    start = remapped_indices[start as usize];
    let sink = (partitions.len() - broken_partitions.len()) as u32;
    let states_no = sink + 1;
    let mut transition = vec![vec![sink; symtable.ids() as usize]; states_no as usize];
    for ((state, symbol), next) in new_trans {
        if !broken_partitions.contains(&state) && !broken_partitions.contains(&next) {
            let remapped_state = remapped_indices[state as usize];
            let remapped_next = remapped_indices[next as usize];
            transition[remapped_state as usize][symbol as usize] = remapped_next;
        }
    }
    // converts the accepting states map into array
    let mut accept = vec![u32::MAX; states_no as usize];
    for i in 0..partitions.len() as u32 {
        if let Some(production) = accept_map.get(&i) {
            if !broken_partitions.contains(&i) {
                let remapped = remapped_indices[i as usize];
                accept[remapped as usize] = *production;
            }
        }
    }
    Dfa {
        states_no,
        transition,
        accept,
        start,
        sink,
    }
}

#[cfg(test)]
mod tests {
    use super::{direct_construction, merge_regex_trees};
    use crate::grammar::{Grammar, Production};
    use crate::lexer::dfa::min_dfa;
    use crate::lexer::grammar_conversion::{canonicalise, parse_trees};
    use crate::lexer::MultiDfa;

    #[test]
    fn dfa_conflicts_resolution() {
        //they should be different: the second accept abb as a*b+ (appearing first in the productions)
        let grammar1 = Grammar::new(
            &[
                ("A", "'a'").into(),
                ("ABB", "'abb'").into(),
                ("ASTARBPLUS", "'a'*'b'+").into(),
            ],
            &[],
        );
        let mdfa1 = MultiDfa::new(&grammar1);
        let dfa1 = mdfa1.dfa(0).unwrap();
        let grammar2 = Grammar::new(
            &[
                ("ASTARBPLUS", "'a'*'b'+").into(),
                ("ABB", "'abb'").into(),
                ("A", "'a'").into(),
            ],
            &[],
        );
        let mdfa2 = MultiDfa::new(&grammar2);
        let dfa2 = mdfa2.dfa(0).unwrap();
        assert!(!dfa1.is_empty());
        assert_eq!(dfa1.states(), 6);
        assert!(!dfa2.is_empty());
        assert_eq!(dfa2.states(), 4);
    }

    #[test]
    fn dfa_direct_construction_no_sink() {
        let terminal = Production::from(("PROD1", "('a'|'b')*'abb'"));
        let grammar = Grammar::new(&[terminal], &[]);
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.states(), 4);
    }

    #[test]
    fn dfa_direct_construction_sink_accepting() {
        let grammar = Grammar::new(
            &[("DIGIT", "[0-9]").into(), ("NUMBER", "[0-9]+").into()],
            &[],
        );
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.states(), 3);
    }

    #[test]
    fn dfa_direct_construction_set_productions() {
        let grammar = Grammar::new(
            &[
                ("LONG1", "[a-c]([b-d]?[e-g])*").into(),
                ("LONG2", "[fg]+").into(),
            ],
            &[],
        );
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert_eq!(dfa.states(), 4);
    }

    #[test]
    fn dfa_direct_construction_start_accepting() {
        let grammar = Grammar::new(&[("ABSTAR", "'ab'*").into()], &[]);
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.states(), 2);
    }

    #[test]
    fn dfa_direct_construction_single_acc() {
        let terminal = Production::from(("PROD1", "(('a'*'b')|'c')?'c'"));
        let grammar = Grammar::new(&[terminal], &[]);
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.states(), 5);
    }

    #[test]
    fn dfa_direct_construction_multi_production() {
        let grammar = Grammar::new(
            &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'*").into()],
            &[],
        );
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(!dfa.is_empty());
        assert_eq!(dfa.states(), 3);
    }

    #[test]
    fn dfa_direct_construction_empty() {
        let grammar = Grammar::empty();
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(dfa.is_empty());
    }

    #[test]
    fn dfa_minimization() {
        let grammar = Grammar::new(
            &[(
                "SEQ",
                "('00'|'11')*(('01'|'10')('00'|'11')*('01'|'10')('00'|'11')*)*",
            )
                .into()],
            &[],
        );
        let (parse_trees, symtable, _) = parse_trees(grammar.iter_term());
        let canonical_trees = parse_trees
            .into_iter()
            .map(|pt| canonicalise(pt, &symtable))
            .collect::<Vec<_>>();
        let merged_tree = merge_regex_trees(canonical_trees);
        let big_dfa = direct_construction(merged_tree, &symtable);
        let min_dfa = min_dfa(&symtable, big_dfa.clone());
        assert_ne!(big_dfa.states(), min_dfa.states());
        assert_eq!(min_dfa.states(), 4);
    }

    #[test]
    fn dfa_transition_correct_size() {
        let grammar = Grammar::new(
            &[(
                "SEQ",
                "('00'|'11')*(('01'|'10')('00'|'11')*('01'|'10')('00'|'11')*)*",
            )
                .into()],
            &[],
        );
        let (parse_trees, symtable, _) = parse_trees(grammar.iter_term());
        let canonical_trees = parse_trees
            .into_iter()
            .map(|pt| canonicalise(pt, &symtable))
            .collect::<Vec<_>>();
        let merged_tree = merge_regex_trees(canonical_trees);
        let big_dfa = direct_construction(merged_tree, &symtable);
        assert_eq!(big_dfa.states_no as usize, big_dfa.transition.len());
        assert_eq!(big_dfa.states_no as usize, big_dfa.accept.len());
        let min_dfa = min_dfa(&symtable, big_dfa);
        assert_eq!(min_dfa.states_no as usize, min_dfa.transition.len());
        assert_eq!(min_dfa.states_no as usize, min_dfa.accept.len());
    }

    #[test]
    fn dfa_moves() {
        let terminal = Production::from(("PROD1", "'c'*'ab'"));
        let grammar = Grammar::new(&[terminal], &[]);
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        let a = mdfa.symbol_table().symbol_id('a');
        let b = mdfa.symbol_table().symbol_id('b');
        let c = mdfa.symbol_table().symbol_id('c');
        assert_eq!(dfa.moove(dfa.start(), c).unwrap(), dfa.start());
        assert!(dfa.moove(dfa.start(), b).is_none());
        assert_ne!(dfa.moove(dfa.start(), a).unwrap(), dfa.start());
    }

    #[test]
    fn dfa_accepting_single() {
        let terminal = Production::from(("PROD1", "'a'"));
        let grammar = Grammar::new(&[terminal], &[]);
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(dfa.accepting(dfa.start()).is_none());
        let next = dfa
            .moove(dfa.start(), mdfa.symbol_table().symbol_id('a'))
            .unwrap();
        assert_eq!(dfa.accepting(next).unwrap(), 0);
    }

    #[test]
    fn nongreedy_rule() {
        let grammar = Grammar::new(
            &[("GREEDY", "'a'+").into(), ("NON_GREEDY", "'b'+?").into()],
            &[],
        );
        let mdfa = MultiDfa::new(&grammar);
        let dfa = mdfa.dfa(0).unwrap();
        assert!(dfa.accepting(dfa.start()).is_none());
        assert!(dfa.non_greedy(dfa.start()));
        let greedy_state = dfa
            .moove(dfa.start(), mdfa.symbol_table().symbol_id('a'))
            .unwrap();
        let ng_state = dfa
            .moove(dfa.start(), mdfa.symbol_table().symbol_id('b'))
            .unwrap();
        assert!(dfa.non_greedy(ng_state));
        assert!(!dfa.non_greedy(greedy_state));
    }
}
