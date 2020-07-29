use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use std::iter::{Enumerate, Peekable};
use std::str::Chars;

use crate::grammar::Grammar;

/// The epsilon value for NFAs.
const EPSILON_VALUE: char = '\u{107FE1}';
/// Placeholder for any character not in the alphabet.
const ANY_VALUE: char = '\u{10A261}';

/// A Binary Search Tree.
#[derive(Clone)]
struct BSTree<T> {
    /// The value contained in the tree.
    value: T,
    /// The left child of the tree.
    left: Option<Box<BSTree<T>>>,
    /// The right child of the tree.
    right: Option<Box<BSTree<T>>>,
}

impl<T: std::fmt::Display> std::fmt::Display for BSTree<T> {
    /// Prints the JSON representation of the tree.
    ///
    /// Each node is formatted as `{val:XXX,left:{...},right:{...}}`.
    /// The `left:{...}` or `right:{...}` parts are omitted if missing.
    /// This method requires the type of the tree to implement the Display trait.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\"val\":\"{}\"", &self.value)?;
        if let Some(left) = &self.left {
            write!(f, ",\"left\":{}", *left)?;
        }
        if let Some(right) = &self.right {
            write!(f, ",\"right\":{}", *right)?;
        }
        write!(f, "}}")
    }
}

/// An Interface for a lexing Finite State Machine.
pub trait Automaton {
    /// Returns true if the automaton is empty.
    ///
    /// An automaton is empty if there are no transitions, and, as such, it halts in the starting
    /// state.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, NFA};
    ///
    /// let grammar = Grammar::new(&[], &[], &[]);
    /// let nfa = NFA::new(&grammar);
    ///
    /// assert!(nfa.is_empty());
    /// ```
    fn is_empty(&self) -> bool;

    /// Returns the number of nodes in the automaton.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, NFA};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    ///
    /// assert_eq!(nfa.nodes(), 7);
    /// ```
    fn nodes(&self) -> usize;

    /// Returns the number of edges in the automaton.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, NFA};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    ///
    /// assert_eq!(nfa.edges(), 8)
    /// ```
    fn edges(&self) -> usize;

    /// Returns a graphviz dot representation of the automaton as string.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, NFA};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    /// nfa.to_dot();
    /// ```
    fn to_dot(&self) -> String;

    /// Saves the graphviz dot representation of the automaton to the given file.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, NFA};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    /// nfa.save_dot("/home/user/nfa.dot");
    /// ```
    fn save_dot(&self, path: &str) {
        std::fs::write(path, self.to_dot())
            .unwrap_or_else(|_| panic!("Unable to write file {}", path));
    }
}

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
pub struct NFA {
    /// Number of states.
    states_no: usize,
    /// Transition map. (node index, symbol) -> Set(node index).
    transition: HashMap<(usize, char), HashSet<usize>>,
    /// All the symbols recognized by the NFA, except EPSILON and ANY_VALUE.
    alphabet: HashSet<char>,
    /// Starting node of the NFA.
    start: usize,
    /// Accepting states. (node index) -> (production index)
    accept: HashMap<usize, usize>,
}

impl NFA {
    /// Builds an NFA using the
    /// [*McNaughton-Yamada-Thompson* algorithm](https://en.wikipedia.org/wiki/Thompson%27s_construction).
    ///
    /// Note that the resulting NFA will have a lot of epsilon moves.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::NFA;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    /// ```
    pub fn new(grammar: &Grammar) -> NFA {
        // Convert a grammar into a series of parse trees, then expand the sets
        let parse_trees = grammar
            .iter_term()
            .map(|x| gen_precedence_tree(x))
            .map(expand_literals)
            .collect::<Vec<_>>();
        if parse_trees.is_empty() {
            // no production found, return a single state, no transaction NFA.
            NFA {
                states_no: 1,
                transition: HashMap::new(),
                alphabet: HashSet::new(),
                start: 0,
                accept: HashMap::new(),
            }
        } else {
            // collect the alphabet for DFA
            let alphabet = parse_trees
                .iter()
                .flat_map(get_alphabet)
                .collect::<HashSet<char>>();
            // convert the parse tree into a canonical one (not ? or +, only *)
            let canonical_tree = parse_trees
                .into_iter()
                .map(|x| canonicalise(x, &alphabet))
                .collect::<Vec<_>>();
            let mut index = 0 as usize; //used to keep unique node indices
                                        // thompson construction
            let mut thompson_nfas = canonical_tree
                .iter()
                .enumerate()
                .map(|x| {
                    let nfa = thompson_construction(x.1, index, x.0);
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
                    .collect::<HashSet<_>>();
                //FIXME: this clone is not particularly efficient (even though I expect nodes in the order of hundredth)
                let accept = thompson_nfas
                    .iter()
                    .flat_map(|x| x.accept.clone())
                    .collect::<HashMap<_, _>>();
                let mut transition_table = thompson_nfas
                    .into_iter()
                    .flat_map(|x| x.transition)
                    .collect::<HashMap<_, _>>();
                transition_table.insert((index, EPSILON_VALUE), start_transition);
                index += 1;
                NFA {
                    states_no: index,
                    transition: transition_table,
                    alphabet,
                    start: index - 1,
                    accept,
                }
            } else {
                thompson_nfas.pop().unwrap()
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
    /// use wisent::lexer::NFA;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = NFA::new(&grammar);
    /// nfa.to_dfa();
    /// ```
    pub fn to_dfa(&self) -> DFA {
        let big_dfa = subset_construction(&self);
        min_dfa(big_dfa)
    }
}

impl std::fmt::Display for NFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NFA({},{})", self.nodes(), self.edges())
    }
}

impl Automaton for NFA {
    fn is_empty(&self) -> bool {
        self.transition.is_empty()
    }

    fn nodes(&self) -> usize {
        self.states_no
    }

    fn edges(&self) -> usize {
        self.transition.iter().fold(0, |acc, x| x.1.len() + acc)
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
                let mut symbol = (trans.0).1;
                if symbol == EPSILON_VALUE {
                    symbol = '\u{03F5}';
                } else if (symbol as usize) < 32 {
                    symbol = '\u{FFFF}';
                } else if symbol == '"' {
                    symbol = '\u{2033}';
                }
                write!(&mut f, "{}->{}[label=\"{}\"];", source, target, symbol).unwrap();
            }
        }
        write!(&mut f, "}}").unwrap();
        f
    }
}

/// A Deterministic Finite Automaton.
///
/// A DFA is an automaton where each state has a single transaction for a given input symbol, and
/// no transactions on empty symbols (ϵ-moves).
///
/// An example of DFA recognizing the language `a|b*` is the following:
///
/// ![DFA Example](../../../../doc/images/dfa.svg)
pub struct DFA {
    /// Number of states.
    states_no: usize,
    /// Transition function: (node index, symbol) -> (node).
    transition: HashMap<(usize, char), usize>,
    /// Set of symbols in the language.
    alphabet: HashSet<char>,
    /// Starting node.
    start: usize,
    /// Accepting states: (node index) -> (accepted production).
    accept: HashMap<usize, usize>,
}

impl std::fmt::Display for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DFA({},{})", self.nodes(), self.edges())
    }
}

impl DFA {
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
    /// use wisent::lexer::DFA;
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = DFA::new(&grammar);
    /// ```
    pub fn new(grammar: &Grammar) -> Self {
        // Convert a grammar into a series of parse trees, then expand the sets
        let parse_trees = grammar
            .iter_term()
            .map(|x| gen_precedence_tree(x))
            .map(expand_literals)
            .collect::<Vec<_>>();
        if parse_trees.is_empty() {
            // no production found, return a single state, no transaction DFA.
            DFA {
                states_no: 1,
                transition: HashMap::new(),
                alphabet: HashSet::new(),
                start: 0,
                accept: HashMap::new(),
            }
        } else {
            // collect the alphabet for DFA
            let alphabet = parse_trees
                .iter()
                .flat_map(get_alphabet)
                .collect::<HashSet<char>>();
            // convert the parse tree into a canonical one (not ? or +, only *)
            let canonical_tree = parse_trees
                .into_iter()
                .map(|x| canonicalise(x, &alphabet))
                .collect::<Vec<_>>();
            // merge all trees of every production into a single one
            let merged_tree = merge_regex_trees(canonical_tree);
            // build the dfa
            let dfa = direct_construction(merged_tree);
            // minimize the dfa
            min_dfa(dfa)
        }
    }
}

impl Automaton for DFA {
    fn is_empty(&self) -> bool {
        self.transition.is_empty()
    }

    fn nodes(&self) -> usize {
        self.states_no
    }

    fn edges(&self) -> usize {
        self.transition.len()
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
            let mut symbol = (trans.0).1;
            if (symbol as usize) < 32 || (symbol as usize) > 126 {
                symbol = '\u{FFFF}';
            } else if symbol == '"' {
                symbol = '\u{2033}';
            } else if symbol == '\\' {
                symbol = '\u{2216}';
            }
            write!(&mut f, "{}->{}[label=\"{}\"];", source, trans.1, symbol).unwrap();
        }
        write!(&mut f, "}}").unwrap();
        f
    }
}

#[derive(Copy, Clone)]
/// Operands/Operators that can be found in a Regex along with their value and priority.
/// This struct is used to build the regex parse tree with the correct priority.
struct RegexOp<'a> {
    /// Type of operand/operator for the regex (for example concatenation, ?, *, (, id...).
    r#type: OpType,
    /// Value of the operand as string slice. Used mostly for the various ID.
    value: &'a str,
    /// priority of the operator.
    priority: u8,
}

///Operators for a regex (and an operand, ID).
#[derive(PartialEq, Debug, Copy, Clone)]
enum OpType {
    /// Kleenee star `*`.
    KLEENE,
    /// Question mark `?`.
    QM,
    /// Plus sign `+`.
    PL,
    /// Left parenthesis `(`.
    LP,
    /// Right parenthesis `)`.
    RP,
    /// Not `~`.
    NOT,
    /// Alternation of two operands.
    OR,
    /// Concatenation of two operands.
    AND,
    /// An Operand.
    ID,
}

impl std::fmt::Display for RegexOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::KLEENE => write!(f, "*"),
            OpType::QM => write!(f, "?"),
            OpType::PL => write!(f, "+"),
            OpType::LP => write!(f, "("),
            OpType::RP => write!(f, ")"),
            OpType::NOT => write!(f, "~"),
            OpType::OR => write!(f, "|"),
            OpType::AND => write!(f, "&"),
            OpType::ID => write!(f, "ID"),
        }
    }
}

/// Extended Literal: exacltly like RegexOp but does not depend on the string slice
/// (because every set has been expanded to a single letter).
#[derive(PartialEq, Debug, Copy, Clone)]
enum ExLiteral {
    Value(char),
    AnyValue,
    Operation(OpType),
}

impl std::fmt::Display for ExLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExLiteral::Value(i) => write!(f, "VALUE({})", i),
            ExLiteral::AnyValue => write!(f, "ANY"),
            ExLiteral::Operation(tp) => write!(f, "OP({})", tp),
        }
    }
}

/// Two operands (Symbol and Accepting state) and a limited set of operators (*, AND, OR).
/// Used to build the canonical parse tree.
#[derive(PartialEq, Debug, Copy, Clone)]
enum Literal {
    /// The input symbol (a single letter).
    Symbol(char),
    /// The accepting state (production number).
    ///
    /// Used only in the direct construction after merging all parse trees into a single one.
    /// This value will record the accepted production at some point in the tree.
    Acc(usize),
    /// Kleenee star unary operator `*`.
    KLEENE,
    /// Concatenation operator.
    AND,
    /// Alternation operator.
    OR,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Symbol(i) => write!(f, "{}", i),
            Literal::Acc(i) => write!(f, "ACC({})", i),
            Literal::KLEENE => write!(f, "*"),
            Literal::AND => write!(f, "&"),
            Literal::OR => write!(f, "|"),
        }
    }
}

/// Helper for the direct construction of the DFA.
struct DCHelper {
    /// Literal type (need to know if kleenee or AND in the followpos computation)
    /// and the followpos is deferred from the firstpos, nullable, lastpos.
    ttype: Literal,
    ///
    index: usize,
    /// true if the current node is nullable.
    nullable: bool,
    /// firstpos for the current set (has an average size of 22 on the C grammar so BTreeSet is
    /// faster than HashSet).
    firstpos: BTreeSet<usize>,
    /// lastpos for the current set (has an average size of 22 on the C grammar so BTreeSet is
    /// faster than HashSet).
    lastpos: BTreeSet<usize>,
}

/// Parse tree for the regex operands, accounting for precedence.
type PrecedenceTree<'a> = BSTree<RegexOp<'a>>;
/// Parse tree for the regex without sets (only single letters).
type ExpandedPrecedenceTree = BSTree<ExLiteral>;
/// Parse tree for the regex with only *, AND, OR.
type CanonicalTree = BSTree<Literal>;

/// Creates a parse tree with correct precedence given the input regex.
///
/// This works similarly to the conversion to Reverse-Polish Notation: two stacks where to push
/// or pop (operators and operands) based on the encountered operators.
fn gen_precedence_tree(regex: &str) -> PrecedenceTree {
    let mut operands = Vec::new();
    let mut operators: Vec<RegexOp> = Vec::new();
    // first get a sequence of operands and operators
    let tokens = regex_to_operands(&regex);
    // for each in the sequence do the following actions
    for operator in tokens {
        match operator.r#type {
            //operators after operand with highest priority -> solve immediately
            OpType::KLEENE | OpType::QM | OpType::PL => {
                operators.push(operator);
                combine_nodes(&mut operands, &mut operators);
            }
            //operators before operand solve if precedent has higher priority and not (
            //then push current
            OpType::NOT | OpType::OR | OpType::AND => {
                if !operators.is_empty() {
                    let top = operators.last().unwrap();
                    if top.priority > operator.priority && top.r#type != OpType::LP {
                        combine_nodes(&mut operands, &mut operators);
                    }
                }
                operators.push(operator);
            }
            // left parenthesis: push. Will be resolved by a right parenthesis.
            OpType::LP => operators.push(operator),
            // right parenthesis: combine all the nodes until left parenthesis is found.
            OpType::RP => {
                while !operators.is_empty() && operators.last().unwrap().r#type != OpType::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
            // id: push to stack
            OpType::ID => {
                let leaf = BSTree {
                    value: operator,
                    left: None,
                    right: None,
                };
                operands.push(leaf);
            }
        }
    }
    //solve all remaining operators
    while !operators.is_empty() {
        combine_nodes(&mut operands, &mut operators);
    }
    operands.pop().unwrap()
}

/// Consumes the input (accounting for escaped chars) until the `until` character is found.
fn consume_counting_until(it: &mut Enumerate<Peekable<Chars>>, until: char) -> usize {
    let mut escapes = 0;
    let mut skipped = 0;
    for skip in it {
        if skip.1 == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if skip.1 == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
        skipped += skip.1.len_utf8();
    }
    skipped
}

//No clippy, this is not more readable.
#[allow(clippy::useless_let_if_seq)]
/// Returns the slice of the regexp representing the token as 'a' or 'a'..'b' or '[a-z]'.
fn read_token<'a>(input: &'a str, first: char, it: &mut Enumerate<Peekable<Chars>>) -> &'a str {
    match first {
        '.' => &input[..1],
        '[' => {
            let counted = consume_counting_until(it, ']');
            &input[..counted + 2] //2 is to match [], plus all the bytes counted inside
        }
        '\'' => {
            let mut counted = consume_counting_until(it, '\'');
            let mut id = &input[..counted + 2]; //this is a valid literal. check for range ''..''
            if input.len() > counted + 5 && &input[(counted + 2)..(counted + 5)] == "..\'" {
                //this is actually a range ''..'' so first advance the iterator by 3 pos
                it.next();
                it.next();
                it.next();
                //then update the counter for the literal length by accounting also the new lit.
                counted += consume_counting_until(it, '\'');
                id = &input[..counted + 6];
            }
            id
        }
        _ => panic!("Unsupported literal {}", input),
    }
}

/// Complementary to the function gen_precedence_tree, combines two nodes with the last operator
/// in the stack.
fn combine_nodes<'a>(operands: &mut Vec<PrecedenceTree<'a>>, operators: &mut Vec<RegexOp<'a>>) {
    let operator = operators.pop().unwrap();
    let left;
    let right;
    if operator.r#type == OpType::OR || operator.r#type == OpType::AND {
        //binary operator
        right = Some(Box::new(operands.pop().unwrap()));
        left = Some(Box::new(operands.pop().unwrap()));
    } else {
        // unary operator
        left = Some(Box::new(operands.pop().unwrap()));
        right = None;
    };
    let ret = BSTree {
        value: operator,
        left,
        right,
    };
    operands.push(ret);
}

/// Transforms a regexp in a sequence of operands or operators.
fn regex_to_operands(regex: &str) -> Vec<RegexOp> {
    let mut tokenz = Vec::<RegexOp>::new();
    let mut iter = regex.chars().peekable().enumerate();
    while let Some((index, char)) = iter.next() {
        let tp;
        let val;
        let priority;
        match char {
            '*' => {
                tp = OpType::KLEENE;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '|' => {
                tp = OpType::OR;
                val = &regex[index..index + 1];
                priority = 1;
            }
            '?' => {
                tp = OpType::QM;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '+' => {
                tp = OpType::PL;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '~' => {
                tp = OpType::NOT;
                val = &regex[index..index + 1];
                priority = 3;
            }
            '(' => {
                tp = OpType::LP;
                val = &regex[index..index + 1];
                priority = 5;
            }
            ')' => {
                tp = OpType::RP;
                val = &regex[index..index + 1];
                priority = 5;
            }
            _ => {
                tp = OpType::ID;
                priority = 0;
                val = read_token(&regex[index..], char, &mut iter);
            }
        };
        if !tokenz.is_empty() && implicit_concatenation(&tokenz.last().unwrap().r#type, &tp) {
            tokenz.push(RegexOp {
                r#type: OpType::AND,
                value: "&",
                priority: 2,
            })
        }
        tokenz.push(RegexOp {
            r#type: tp,
            value: val,
            priority,
        });
    }
    tokenz
}

/// Given two tokens, `a` and `b` returns true if a concatenation is implied between the two.
/// For example if the tokens are `'a'` and `'b'` the result will be true, because the word `ab`
/// is the concatenation of the two letters.
/// If the tokens are `'a'` and `*` the result will be false as there is no implicit operator.
///
/// Refer to the following table for each rule:
/// - `/` = Not allowed.
/// - `✗` = Concatenation not required.
/// - `✓` = Concatenation required.
/// - Rows: parameter `last`
/// - Columns: parameter `current`
/// Operator `*` is valid also for `?` and `+`.
///
/// |last|`|`|`~`|`*`|`(`|`)`| ID|
///  |---|---|---|---|---|---|---|
///  |`|`|`/`|`/`|`/`|`/`|`/`|`✗`|
///  |`~`|`/`|`✗`|`/`|`✗`|`/`|`✗`|
///  |`*`|`/`|`/`|`/`|`✓`|`✗`|`✓`|
///  |`(`|`/`|`✗`|`/`|`✗`|`✗`|`✗`|
///  |`)`|`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
///  |ID |`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
fn implicit_concatenation(last: &OpType, current: &OpType) -> bool {
    let last_is_kleene_family =
        *last == OpType::KLEENE || *last == OpType::PL || *last == OpType::QM;
    let cur_is_lp_or_id = *current == OpType::LP || *current == OpType::ID;
    (last_is_kleene_family && cur_is_lp_or_id)
        || (*last == OpType::RP && (*current == OpType::NOT || cur_is_lp_or_id))
        || (*last == OpType::ID && (*current == OpType::NOT || cur_is_lp_or_id))
}

/// Transforms a precedence parse tree in a precedence parse tree where the groups like `[a-z]`
/// are expanded in `a | b | c ... | y | z`
fn expand_literals(node: PrecedenceTree) -> ExpandedPrecedenceTree {
    match node.value.r#type {
        OpType::ID => expand_literal_node(node.value.value),
        n => {
            let left = match node.left {
                Some(l) => Some(Box::new(expand_literals(*l))),
                None => None,
            };
            let right = match node.right {
                Some(r) => Some(Box::new(expand_literals(*r))),
                None => None,
            };
            BSTree {
                value: ExLiteral::Operation(n),
                left,
                right,
            }
        }
    }
}

/// Expand a single node containing sets like `[a-z]` in the single symbols concatenated.
/// Replace also the . symbol with the special placeholder to represent any value.
fn expand_literal_node(literal: &str) -> ExpandedPrecedenceTree {
    if literal == "." {
        return BSTree {
            value: ExLiteral::AnyValue,
            left: None,
            right: None,
        };
    }
    let mut charz = Vec::new();
    let mut iter = literal.chars();
    let start = iter.next().unwrap();
    let end;
    let mut last = '\x00';
    let mut set_op = if start == '[' {
        end = ']';
        OpType::OR
    } else {
        end = '\'';
        OpType::AND
    };
    while let Some(char) = iter.next() {
        let mut pushme = char;
        if char == '\\' {
            //escaped char
            pushme = unescape_character(iter.next().unwrap(), &mut iter);
            last = pushme;
            charz.push(BSTree {
                value: ExLiteral::Value(pushme),
                left: None,
                right: None,
            });
        } else if set_op == OpType::OR && char == '-' {
            //set in form a-z, A-Z, 0-9, etc..
            let from = last as u32 + 1;
            let until = iter.next().unwrap() as u32 + 1; //included
            for i in from..until {
                pushme = std::char::from_u32(i).unwrap();
                charz.push(BSTree {
                    value: ExLiteral::Value(pushme),
                    left: None,
                    right: None,
                });
            }
        } else if char == end {
            //end of sequence
            break;
        } else {
            //normal char
            last = pushme;
            charz.push(BSTree {
                value: ExLiteral::Value(pushme as char),
                left: None,
                right: None,
            });
        }
    }
    //check possible range in form 'a'..'z', at this point I ASSUME this can be a literal only
    //and the syntax has already been checked.
    if let Some(_c @ '.') = iter.next() {
        set_op = OpType::OR;
        iter.next();
        iter.next();
        let mut until_char = iter.next().unwrap();
        if until_char == '\\' {
            until_char = unescape_character(iter.next().unwrap(), &mut iter);
        }
        let from = last as u32 + 1;
        let until = until_char as u32 + 1;
        for i in from..until {
            let pushme = std::char::from_u32(i).unwrap();
            charz.push(BSTree {
                value: ExLiteral::Value(pushme),
                left: None,
                right: None,
            });
        }
    }
    while charz.len() >= 2 {
        let right = charz.pop().unwrap();
        let left = charz.pop().unwrap();
        let new = BSTree {
            value: ExLiteral::Operation(set_op),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        };
        charz.push(new);
    }
    charz.pop().unwrap()
}

/// Transforms escaped strings in the form "\\n" to the single character they represent '\n'.
/// Works also for unicode in the form "\\UXXXX".
///
/// **NOTE**: it does NOT work for every unicode character (as they can be up to \uXXXXXX) because
/// ANTLR grammars support only up to \u{FFFF}.
fn unescape_character<T: Iterator<Item = char>>(letter: char, iter: &mut T) -> char {
    match letter {
        'n' => '\n',
        'r' => '\r',
        'b' => '\x08',
        't' => '\t',
        'f' => '\x0C',
        'u' | 'U' => {
            //NOTE: ANTLR grammars support only the BMP(0th) plane. Max is \uXXXX
            //      So no escaped emojis :'(
            let digit3 = iter.next().unwrap().to_digit(16).unwrap();
            let digit2 = iter.next().unwrap().to_digit(16).unwrap();
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit3 << 12 | digit2 << 8 | digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        'x' | 'X' => {
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        _ => letter,
    }
}

/// Returns a set containing all the characters used in the regexp extended tree.
fn get_alphabet(node: &ExpandedPrecedenceTree) -> HashSet<char> {
    let mut ret = HashSet::new();
    let mut todo_nodes = vec![node];
    while let Some(node) = todo_nodes.pop() {
        match node.value {
            ExLiteral::Value(i) => {
                ret.insert(i);
            }
            ExLiteral::AnyValue => {}
            ExLiteral::Operation(_) => {
                //nothing to do with the "operation" itself, but this is the only non-leaf type node
                if let Some(left) = &node.left {
                    todo_nodes.push(&*left);
                }
                if let Some(right) = &node.right {
                    todo_nodes.push(&*right);
                }
            }
        }
    }
    ret
}

/// Transform a regex extended parse tree to a canonical parse tree (i.e. a tree with only symbols,
/// the *any symbol* placeholder, concatenation, alternation, kleene star).
fn canonicalise(node: ExpandedPrecedenceTree, alphabet: &HashSet<char>) -> CanonicalTree {
    match node.value {
        ExLiteral::Value(i) => BSTree {
            value: Literal::Symbol(i),
            left: None,
            right: None,
        },
        ExLiteral::AnyValue => {
            let mut chars = alphabet
                .iter()
                .chain(&[ANY_VALUE])
                .map(|c| BSTree {
                    value: Literal::Symbol(*c),
                    left: None,
                    right: None,
                })
                .collect::<Vec<_>>();
            while chars.len() >= 2 {
                let left = Some(Box::new(chars.pop().unwrap()));
                let right = Some(Box::new(chars.pop().unwrap()));
                let tree = BSTree {
                    value: Literal::OR,
                    left,
                    right,
                };
                chars.push(tree);
            }
            chars.pop().unwrap()
        }
        ExLiteral::Operation(op) => {
            match op {
                OpType::NOT => {
                    //get the entire used alphabet for both nodes
                    let mut subnode_alphabet = HashSet::new();
                    if let Some(l) = node.left {
                        subnode_alphabet.extend(get_alphabet(&*l));
                    }
                    if let Some(r) = node.right {
                        subnode_alphabet.extend(get_alphabet(&*r));
                    }
                    // this creates problems in ~. but this statement is stupid so I don't care
                    let mut diff = alphabet
                        .difference(&subnode_alphabet)
                        .chain(&[ANY_VALUE])
                        .map(|c| BSTree {
                            value: Literal::Symbol(*c),
                            left: None,
                            right: None,
                        })
                        .collect::<Vec<_>>();
                    while diff.len() >= 2 {
                        let right = diff.pop().unwrap();
                        let left = diff.pop().unwrap();
                        let new_node = BSTree {
                            value: Literal::OR,
                            left: Some(Box::new(left)),
                            right: Some(Box::new(right)),
                        };
                        diff.push(new_node);
                    }
                    diff.pop().unwrap()
                }
                OpType::OR => {
                    let left = match node.left {
                        Some(l) => Some(Box::new(canonicalise(*l, alphabet))),
                        None => None,
                    };
                    let right = match node.right {
                        Some(r) => Some(Box::new(canonicalise(*r, alphabet))),
                        None => None,
                    };
                    BSTree {
                        value: Literal::OR,
                        left,
                        right,
                    }
                }
                OpType::AND => {
                    let left = match node.left {
                        Some(l) => Some(Box::new(canonicalise(*l, alphabet))),
                        None => None,
                    };
                    let right = match node.right {
                        Some(r) => Some(Box::new(canonicalise(*r, alphabet))),
                        None => None,
                    };
                    BSTree {
                        value: Literal::AND,
                        left,
                        right,
                    }
                }
                OpType::KLEENE => {
                    let left = match node.left {
                        Some(l) => Some(Box::new(canonicalise(*l, alphabet))),
                        None => None,
                    };
                    let right = match node.right {
                        Some(r) => Some(Box::new(canonicalise(*r, alphabet))),
                        None => None,
                    };
                    BSTree {
                        value: Literal::KLEENE,
                        left,
                        right,
                    }
                }
                OpType::QM => {
                    let left = Some(Box::new(BSTree {
                        value: Literal::Symbol(EPSILON_VALUE),
                        left: None,
                        right: None,
                    }));
                    //it the node has a ? DEFINITELY it has only a left children
                    let right = Some(Box::new(canonicalise(*node.left.unwrap(), alphabet)));
                    BSTree {
                        value: Literal::OR,
                        left,
                        right,
                    }
                }
                OpType::PL => {
                    //it the node has a + DEFINITELY it has only a left children
                    let left = canonicalise(*node.left.unwrap(), alphabet);
                    let right = BSTree {
                        value: Literal::KLEENE,
                        left: Some(Box::new(left.clone())),
                        right: None,
                    };
                    BSTree {
                        value: Literal::AND,
                        left: Some(Box::new(left)),
                        right: Some(Box::new(right)),
                    }
                }
                n => panic!("Unexpected operation {}", n),
            }
        }
    }
}

/// Performs the thompson construction on a regex canonical parse tree to obtain an NFA.
///
/// - `start_index`: Starts assigning indices to NFA nodes from this number.
/// - `production`: This is the announced production index for the current parse tree.
fn thompson_construction(prod: &CanonicalTree, start_index: usize, production: usize) -> NFA {
    let mut index = start_index;
    let mut visit = vec![prod];
    let mut todo = Vec::new();
    let mut done = Vec::<NFA>::new();
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
                let mut alphabet = HashSet::new();
                if val != EPSILON_VALUE {
                    alphabet.insert(val);
                }
                pushme = NFA {
                    states_no: 2,
                    transition: hashmap! {
                        (index, val) => hashset!{index+1},
                    },
                    alphabet,
                    start: index,
                    accept: hashmap! {index+1 => production},
                };
                index += 2;
            }
            Literal::KLEENE => {
                let new_start = index;
                let new_end = index + 1;
                index += 2;
                let mut first = done.pop().unwrap();
                for acc in first.accept {
                    first
                        .transition
                        .insert((acc.0, EPSILON_VALUE), hashset! {first.start, new_end});
                }
                first
                    .transition
                    .insert((new_start, EPSILON_VALUE), hashset! {first.start, new_end});
                first.start = new_start;
                first.accept = hashmap! {new_end => production};
                first.states_no += 2;
                pushme = first;
            }
            Literal::AND => {
                let second = done.pop().unwrap();
                let mut first = done.pop().unwrap();
                first.transition.extend(second.transition);
                for acc in first.accept {
                    first
                        .transition
                        .insert((acc.0, EPSILON_VALUE), hashset! {second.start});
                }
                first.accept = second.accept;
                first.alphabet = first.alphabet.union(&second.alphabet).cloned().collect();
                first.states_no += second.states_no;
                pushme = first;
            }
            Literal::OR => {
                let new_start = index;
                let new_end = index + 1;
                index += 2;
                let second = done.pop().unwrap();
                let mut first = done.pop().unwrap();
                first.transition.extend(second.transition);
                first.transition.insert(
                    (new_start, EPSILON_VALUE),
                    hashset! {first.start, second.start},
                );
                for acc in first.accept.into_iter().chain(second.accept.into_iter()) {
                    first
                        .transition
                        .insert((acc.0, EPSILON_VALUE), hashset! {new_end});
                }
                first.start = new_start;
                first.alphabet = first.alphabet.union(&second.alphabet).cloned().collect();
                first.accept = hashmap! {new_end => production};
                first.states_no += second.states_no + 2;
                pushme = first;
            }
            Literal::Acc(_) => panic!("Accept state not allowed in thompson construction!"),
        }
        done.push(pushme);
    }
    done.pop().unwrap()
}

/// Transforms a NFA to a DFA using subset construction algorithm.
///
/// Guaranteed to have a move on every symbol for every node.
fn subset_construction(nfa: &NFA) -> DFA {
    let mut ds_marked = BTreeSet::new();
    let mut ds_unmarked = Vec::new();
    let mut indices = HashMap::new();
    let mut index = 0 as usize;
    let mut transition = HashMap::new();
    let mut accept = HashMap::new();

    let s0 = sc_epsilon_closure(hashset! {nfa.start}, &nfa.transition);
    indices.insert(s0.clone(), index);
    ds_unmarked.push(s0);
    index += 1;
    while let Some(t) = ds_unmarked.pop() {
        let t_idx = *indices.get(&t).unwrap();
        ds_marked.insert(t.clone());
        for symbol in nfa.alphabet.iter() {
            let sym = *symbol;
            let mov = sc_move(&t, sym, &nfa.transition);
            let u = sc_epsilon_closure(mov, &nfa.transition);
            let u_idx;
            if !u.is_empty() {
                //check if node has already been created
                if !indices.contains_key(&u) {
                    u_idx = index;
                    indices.insert(u.clone(), u_idx);
                    index += 1;
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
            transition.entry((node, *symbol)).or_insert(sink);
        }
    }
    DFA {
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
    set: HashSet<usize>,
    transition_table: &HashMap<(usize, char), HashSet<usize>>,
) -> BTreeSet<usize> {
    let mut stack = set.iter().copied().collect::<Vec<_>>();
    let mut closure = set;
    while let Some(t) = stack.pop() {
        if let Some(eset) = transition_table.get(&(t, EPSILON_VALUE)) {
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
fn sc_accepting(set: &BTreeSet<usize>, accepting: &HashMap<usize, usize>) -> Option<usize> {
    let mut productions = BTreeSet::new();
    for node in accepting {
        if set.contains(&node.0) {
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
    set: &BTreeSet<usize>,
    symbol: char,
    transition: &HashMap<(usize, char), HashSet<usize>>,
) -> HashSet<usize> {
    let mut ret = HashSet::new();
    for node in set {
        if let Some(t) = transition.get(&(*node, symbol)) {
            ret = ret.union(t).cloned().collect::<HashSet<_>>();
        }
    }
    ret
}

/// Merges different canonical trees into a single canonical tree with multiple accepting nodes.
/// Accepting states are labeled with a new node in the canonical tree.
fn merge_regex_trees(nodes: Vec<CanonicalTree>) -> CanonicalTree {
    let mut roots = Vec::new();
    for node in nodes.into_iter().enumerate() {
        let right = BSTree {
            value: Literal::Acc(node.0),
            left: None,
            right: None,
        };
        let root = BSTree {
            value: Literal::AND,
            left: Some(Box::new(node.1)),
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

/// Performs a direct DFA construction from the canonical tree without using an intermediate NFA.
/// Refers to "Compilers, principle techniques and tools" of A.Aho et al. (p.179 on 2nd edition).
///
/// Guaranteed to have a move on every symbol for every node.
fn direct_construction(node: CanonicalTree) -> DFA {
    let helper = build_dc_helper(&node, 0);
    let mut indices = vec![EPSILON_VALUE; helper.value.index + 1];
    let mut followpos = vec![BTreeSet::new(); helper.value.index + 1];
    let mut accepting = HashMap::new();
    dc_assign_index_to_literal(&helper, &mut indices, &mut accepting);
    dc_compute_followpos(&helper, &mut followpos);
    dc_build_graph(helper.value.firstpos, &followpos, &indices, &accepting)
}

/// Part of the direct DFA construction:
///
/// Computes `firstpos`, `lastpos` and `nullpos` for the parse tree.
///
/// - `node`: the root of the tree (it's a recursive function)
/// - `start_index`: starting index of the output DFA (0 for the first invocation)
fn build_dc_helper(node: &CanonicalTree, start_index: usize) -> BSTree<DCHelper> {
    //postorder because I need to build it bottom up
    let mut index = start_index;
    let mut children = [&node.left, &node.right]
        .iter()
        .map(|x| match x {
            Some(c) => {
                let helper = build_dc_helper(&*c, index);
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
            if *val == EPSILON_VALUE {
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
fn dc_compute_followpos(node: &BSTree<DCHelper>, graph: &mut Vec<BTreeSet<usize>>) {
    if let Some(l) = &node.left {
        dc_compute_followpos(&*l, graph);
    }
    if let Some(r) = &node.right {
        dc_compute_followpos(&*r, graph);
    }
    match &node.value.ttype {
        Literal::Symbol(_) => {}
        Literal::Acc(_) => {}
        Literal::OR => {}
        Literal::AND => {
            let c1 = &**node.left.as_ref().unwrap();
            let c2 = &**node.right.as_ref().unwrap();
            for i in &c1.value.lastpos {
                graph[*i] = graph[*i].union(&c2.value.firstpos).cloned().collect();
            }
        }
        Literal::KLEENE => {
            for i in &node.value.lastpos {
                graph[*i] = graph[*i].union(&node.value.firstpos).cloned().collect();
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
    indices: &mut Vec<char>,
    acc: &mut HashMap<usize, usize>,
) {
    if let Some(l) = &node.left {
        dc_assign_index_to_literal(&*l, indices, acc);
    }
    if let Some(r) = &node.right {
        dc_assign_index_to_literal(&*r, indices, acc);
    }
    match &node.value.ttype {
        Literal::Symbol(val) => indices[node.value.index] = *val,
        Literal::Acc(prod) => {
            acc.insert(node.value.index, *prod);
        }
        _ => {}
    }
}

/// Part of the direct DFA construction:
///
/// Performs the actual DFA construction given the starting set, followpos set, the assigned indices
/// and the accepting nodes.
fn dc_build_graph(
    start: BTreeSet<usize>,
    followpos: &[BTreeSet<usize>],
    indices: &[char],
    accepting: &HashMap<usize, usize>,
) -> DFA {
    let mut done = hashmap! {
        start.clone() => 0
    };
    let mut index = 1 as usize;
    let mut unmarked = vec![start];
    let mut tran = HashMap::new();
    let mut accept = HashMap::new();
    let alphabet = indices
        .iter()
        .filter(|x| **x != EPSILON_VALUE)
        .copied()
        .collect::<HashSet<char>>();
    while let Some(node_set) = unmarked.pop() {
        for letter in &alphabet {
            let u = node_set
                .iter()
                .filter(|x| indices[**x] == *letter)
                .flat_map(|x| &followpos[*x])
                .cloned()
                .collect::<BTreeSet<_>>();
            let u_idx;
            if let Some(got) = done.get(&u) {
                u_idx = *got;
            } else {
                u_idx = index;
                index += 1;
                if let Some(acc_prod) = u.iter().flat_map(|x| accepting.get(x)).min() {
                    accept.insert(u_idx, *acc_prod);
                }
                unmarked.push(u.clone());
                done.insert(u, u_idx);
            }
            let set_idx = *done.get(&node_set).unwrap();
            tran.insert((set_idx, *letter), u_idx);
        }
    }
    DFA {
        alphabet,
        states_no: index,
        start: 0,
        transition: tran,
        accept,
    }
}

/// Given a DFA returns the DFA with the minimum number of nodes.
///
/// **REQUIRED** to have a move on every symbol for every node.
///
/// Again the source of this algorithm is "Compilers, principle techniques and tools" of
/// A.Aho et al. (p.180 on 2nd edition).
fn min_dfa(dfa: DFA) -> DFA {
    let mut partitions = init_partitions(&dfa);
    let mut positions = HashMap::new();
    for (partition_index, partition) in partitions.iter().enumerate() {
        for node in partition {
            positions.insert(*node, partition_index);
        }
    }
    while partitions.len() < dfa.states_no {
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
                    positions.insert(*node, old_partitions.len() + new_idx);
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
fn init_partitions(dfa: &DFA) -> Vec<HashSet<usize>> {
    let mut announced_max = std::usize::MIN;
    let mut acc = HashSet::new();
    for announced in &dfa.accept {
        acc.insert(*announced.0);
        announced_max = announced_max.max(*announced.1);
    }
    let accepting_no = announced_max + 1;
    let nacc = (0 as usize..)
        .take(dfa.states_no)
        .collect::<HashSet<_>>()
        .difference(&acc)
        .cloned()
        .collect::<HashSet<_>>();
    // this is a DFA for lexical analysis so I need to further split acc by announced rule
    let mut ret = vec![HashSet::new(); accepting_no];
    for announced in &dfa.accept {
        ret[(*announced.1)].insert(*announced.0);
    }
    ret.push(nacc); //add the non_accepting partition to the end
    ret
}

/// Part of the min DFA algorithm:
///
/// Splits a partition if two nodes goes to different partitions on the same symbol.
fn split_partition(
    partition: HashSet<usize>,
    position: &HashMap<usize, usize>,
    dfa: &DFA,
) -> (HashSet<usize>, HashSet<usize>) {
    let mut split = HashSet::new();
    if partition.len() > 1 {
        for symbol in &dfa.alphabet {
            let mut iter = partition.iter();
            let first = *iter.next().unwrap();
            let expected_target = *position
                .get(dfa.transition.get(&(first, *symbol)).unwrap())
                .unwrap();
            for node in iter {
                let target = *position
                    .get(dfa.transition.get(&(*node, *symbol)).unwrap())
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
fn remap(partitions: Vec<HashSet<usize>>, positions: HashMap<usize, usize>, dfa: DFA) -> DFA {
    //first record in which partitions is every node
    let mut new_trans = HashMap::new();
    let mut accept = HashMap::new();
    let mut in_degree = vec![0 as usize; partitions.len()];
    let mut out_degree = vec![0 as usize; partitions.len()];
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
            out_degree[new_source] += 1;
            in_degree[new_target] += 1;
        }
        new_trans.insert((new_source, letter), new_target);
    }
    // start = partition of previous start
    let start = *positions.get(&(0 as usize)).unwrap();
    // remove unreachable states (and non-accepting sinks)
    //broken: no in-edges and no start state OR no out-edges and not accepting, excluding self-loops
    let broken_states = (0 as usize..)
        .take(partitions.len())
        .filter(|x| {
            (in_degree[*x] == 0 && *x != start) || (out_degree[*x] == 0 && !accept.contains_key(x))
        })
        .collect::<BTreeSet<_>>();
    new_trans = new_trans
        .into_iter()
        .filter(|x| !broken_states.contains(&(x.0).0) && !broken_states.contains(&x.1))
        .collect();
    DFA {
        states_no: partitions.len() - broken_states.len(),
        transition: new_trans,
        alphabet: dfa.alphabet,
        accept,
        start,
    }
}

#[cfg(test)]
#[path = "tests/lexer.rs"]
mod tests;
