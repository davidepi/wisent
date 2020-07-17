use std::collections::{BTreeSet, HashMap, HashSet};
use std::iter::{Enumerate, Peekable};
use std::str::Chars;

use crate::grammar::Grammar;

const EPSILON_VALUE: char = '\u{107FE1}';
const ANY_VALUE: char = '\u{10A261}';

#[derive(Clone)]
pub struct BSTree<T> {
    pub value: T,
    pub left: Option<Box<BSTree<T>>>,
    pub right: Option<Box<BSTree<T>>>,
}

type NFANode = usize;
pub struct NFA {
    states_no: usize,
    transition: HashMap<(NFANode, char), BTreeSet<NFANode>>,
    alphabet: HashSet<char>,
    start: NFANode,
    accept: BTreeSet<(NFANode, usize)>,
}

type DFANode = usize;
pub struct DFA {
    states_no: usize,
    transition: HashMap<(DFANode, char), DFANode>,
    accept: BTreeSet<(DFANode, usize)>,
}

impl NFA {
    pub fn is_empty(&self) -> bool {
        self.states_no == 0
    }

    pub fn nodes(&self) -> usize {
        self.states_no
    }

    pub fn edges(&self) -> usize {
        self.transition.iter().fold(0, |acc, x| x.1.len() + acc)
    }
}

impl DFA {
    pub fn is_empty(&self) -> bool {
        self.states_no == 0
    }

    pub fn nodes(&self) -> usize {
        self.states_no
    }

    pub fn edges(&self) -> usize {
        self.transition.len()
    }
}

impl std::fmt::Display for NFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "digraph{{start[shape=point];")?;
        for state in &self.accept {
            write!(
                f,
                "{}[shape=doublecircle;xlabel=\"ACC({})\"];",
                state.0, state.1
            )?;
        }
        write!(f, "start->{};", &self.start)?;
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
                write!(f, "{}->{}[label=\"{}\"];", source, target, symbol)?;
            }
        }
        write!(f, "}}")
    }
}

impl std::fmt::Display for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "digraph{{start[shape=point];")?;
        for state in &self.accept {
            write!(
                f,
                "{}[shape=doublecircle;xlabel=\"ACC({})\"];",
                state.0, state.1
            )?;
        }
        write!(f, "start->0;")?;
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
            write!(f, "{}->{}[label=\"{}\"];", source, trans.1, symbol)?;
        }
        write!(f, "}}")
    }
}

fn sc_epsilon_closure(
    set: &BTreeSet<NFANode>,
    tt: &HashMap<(NFANode, char), BTreeSet<NFANode>>,
) -> BTreeSet<NFANode> {
    let mut stack = set.iter().copied().collect::<Vec<_>>();
    let mut closure = set.iter().copied().collect::<BTreeSet<_>>();
    while let Some(t) = stack.pop() {
        if let Some(eset) = tt.get(&(t, EPSILON_VALUE)) {
            closure = closure.union(eset).cloned().collect::<BTreeSet<_>>();
            stack.extend(eset.iter());
        }
    }
    closure
}

pub fn subset_construction(nfa: &NFA) -> DFA {
    let mut ds_marked = BTreeSet::new();
    let mut ds_unmarked = Vec::new();
    let mut indices = HashMap::new();
    let mut index = 0 as usize;
    let alpha = nfa
        .alphabet
        .iter()
        .chain(&[ANY_VALUE])
        .collect::<HashSet<_>>();
    let mut transition = HashMap::new();
    let mut accept = BTreeSet::new();

    let s0 = sc_epsilon_closure(&btreeset! {nfa.start}, &nfa.transition);
    indices.insert(s0.clone(), index);
    ds_unmarked.push(s0);
    index += 1;
    while let Some(t) = ds_unmarked.pop() {
        let t_idx = *indices.get(&t).unwrap();
        ds_marked.insert(t.clone());
        for symbol in alpha.iter() {
            let sym = **symbol;
            let mov = sc_move(&t, sym, &nfa.transition);
            let u = sc_epsilon_closure(&mov, &nfa.transition);
            let u_idx;
            if !u.is_empty() {
                //check if node has already been created
                if !indices.contains_key(&u) {
                    u_idx = index;
                    indices.insert(u.clone(), u_idx);
                    index += 1;
                    if let Some(accepted_production) = sc_accepting(&u, &nfa.accept) {
                        accept.insert((u_idx, accepted_production));
                    }
                    ds_unmarked.push(u);
                } else {
                    u_idx = *indices.get(&u).unwrap();
                }
                transition.insert((t_idx, sym), u_idx);
            }
        }
    }
    DFA {
        states_no: index,
        transition,
        accept,
    }
}

fn sc_accepting(set: &BTreeSet<NFANode>, accepting: &BTreeSet<(NFANode, usize)>) -> Option<usize> {
    let mut productions = BTreeSet::new();
    for node in accepting {
        if set.contains(&node.0) {
            productions.insert(node.1);
        }
    }
    if !productions.is_empty() {
        Some(*productions.iter().next().unwrap()) //get smallest value (production appearing first)
    } else {
        None
    }
}

fn sc_move(
    set: &BTreeSet<NFANode>,
    symbol: char,
    tt: &HashMap<(NFANode, char), BTreeSet<NFANode>>,
) -> BTreeSet<NFANode> {
    let mut ret = BTreeSet::new();
    for node in set {
        if let Some(t) = tt.get(&(*node, symbol)) {
            ret = ret.union(t).cloned().collect::<BTreeSet<_>>();
        }
    }
    ret
}

fn thompson_construction(prod: &BSTree<Literal>, start_index: usize, production: usize) -> NFA {
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
            Literal::Value(val) => {
                pushme = NFA {
                    states_no: 2,
                    transition: hashmap! {
                        (index, val) => btreeset!{index+1},
                    },
                    alphabet: hashset! {val},
                    start: index,
                    accept: btreeset! {(index+1, production)},
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
                        .insert((acc.0, EPSILON_VALUE), btreeset! {first.start, new_end});
                }
                first
                    .transition
                    .insert((new_start, EPSILON_VALUE), btreeset! {first.start, new_end});
                first.start = new_start;
                first.accept = btreeset! {(new_end, production)};
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
                        .insert((acc.0, EPSILON_VALUE), btreeset! {second.start});
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
                    btreeset! {first.start, second.start},
                );
                for acc in first.accept.into_iter().chain(second.accept.into_iter()) {
                    first
                        .transition
                        .insert((acc.0, EPSILON_VALUE), btreeset! {new_end});
                }
                first.start = new_start;
                first.alphabet = first.alphabet.union(&second.alphabet).cloned().collect();
                first.accept = btreeset! {(new_end, production)};
                first.states_no += second.states_no + 2;
                pushme = first;
            }
            Literal::Acc(_) => panic!("Accept state not allowed in thompson construction!"),
        }
        done.push(pushme);
    }
    done.pop().unwrap()
}

pub fn transition_table_nfa(grammar: &Grammar) -> NFA {
    let parse_trees = grammar
        .iter_term()
        .map(|x| gen_parse_tree(x))
        .map(expand_literals)
        .collect::<Vec<_>>();
    let alphabet = parse_trees
        .iter()
        .flat_map(get_alphabet)
        .collect::<HashSet<char>>();
    let canonical_tree = parse_trees
        .into_iter()
        .map(|x| canonicalise(x, &alphabet))
        .collect::<Vec<_>>();
    let mut index = 0 as usize; //used to keep unique node indices
    let mut thompson_nfa = canonical_tree
        .iter()
        .enumerate()
        .map(|x| {
            let nfa = thompson_construction(x.1, index, x.0);
            index += nfa.nodes();
            nfa
        })
        .collect::<Vec<_>>();
    //merge productions into a single NFA
    if thompson_nfa.len() > 1 {
        let start_transition = thompson_nfa
            .iter()
            .map(|x| x.start)
            .collect::<BTreeSet<_>>();
        //FIXME: this clone is not particularly efficient (even though I expect nodes in the order of hundredth)
        let accept = thompson_nfa
            .iter()
            .flat_map(|x| x.accept.clone())
            .collect::<BTreeSet<_>>();
        let mut transition_table = thompson_nfa
            .into_iter()
            .flat_map(|x| x.transition)
            .collect::<HashMap<_, _>>();
        transition_table.insert((index, EPSILON_VALUE), start_transition);
        NFA {
            states_no: index + 1,
            transition: transition_table,
            alphabet,
            start: index,
            accept,
        }
    } else {
        thompson_nfa.pop().unwrap()
    }
}

pub fn transition_table_dfa(grammar: &Grammar) -> DFA {
    let parse_trees = grammar
        .iter_term()
        .map(|x| gen_parse_tree(x))
        .map(expand_literals)
        .collect::<Vec<_>>();
    let alphabet = parse_trees
        .iter()
        .flat_map(get_alphabet)
        .collect::<HashSet<char>>();
    let canonical_tree = parse_trees
        .into_iter()
        .map(|x| canonicalise(x, &alphabet))
        .collect::<Vec<_>>();
    let merged_tree = collect_productions(canonical_tree);
    direct_construction(merged_tree)
}

impl<T> std::fmt::Display for BSTree<T>
where
    T: std::fmt::Display,
{
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

#[derive(Copy, Clone)]
pub(super) struct RegexOp<'a> {
    pub(super) r#type: OpType,
    pub(super) value: &'a str,
    pub(super) priority: u8,
}

impl std::fmt::Display for RegexOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub(super) enum ExLiteral {
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

#[derive(PartialEq, Debug, Copy, Clone)]
pub(super) enum Literal {
    Value(char),
    Acc(usize),
    KLEENE,
    AND,
    OR,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Value(i) => write!(f, "{}", i),
            Literal::Acc(i) => write!(f, "ACC({})", i),
            Literal::KLEENE => write!(f, "*"),
            Literal::AND => write!(f, "&"),
            Literal::OR => write!(f, "|"),
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub(super) enum OpType {
    KLEENE,
    QM,
    PL,
    LP,
    RP,
    NOT,
    OR,
    AND,
    ID,
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

fn combine_nodes<'a>(operands: &mut Vec<BSTree<RegexOp<'a>>>, operators: &mut Vec<RegexOp<'a>>) {
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

fn tokenize(regex: &str) -> Vec<RegexOp> {
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

fn implicit_concatenation(last: &OpType, current: &OpType) -> bool {
    let last_is_kleene_family =
        *last == OpType::KLEENE || *last == OpType::PL || *last == OpType::QM;
    let cur_is_lp_or_id = *current == OpType::LP || *current == OpType::ID;
    (last_is_kleene_family && cur_is_lp_or_id)
        || (*last == OpType::RP && (*current == OpType::NOT || cur_is_lp_or_id))
        || (*last == OpType::ID && (*current == OpType::NOT || cur_is_lp_or_id))
}

pub(super) fn gen_parse_tree(regex: &str) -> BSTree<RegexOp> {
    let mut operands = Vec::new();
    let mut operators: Vec<RegexOp> = Vec::new();
    let tokens = tokenize(&regex);
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
            OpType::LP => operators.push(operator),
            OpType::RP => {
                while !operators.is_empty() && operators.last().unwrap().r#type != OpType::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
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

pub(super) fn expand_literals(node: BSTree<RegexOp>) -> BSTree<ExLiteral> {
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

fn expand_literal_node(literal: &str) -> BSTree<ExLiteral> {
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

pub(super) fn get_alphabet(node: &BSTree<ExLiteral>) -> HashSet<char> {
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

pub(super) fn canonicalise(node: BSTree<ExLiteral>, alphabet: &HashSet<char>) -> BSTree<Literal> {
    match node.value {
        ExLiteral::Value(i) => BSTree {
            value: Literal::Value(i),
            left: None,
            right: None,
        },
        ExLiteral::AnyValue => {
            let mut chars = alphabet
                .iter()
                .chain(&[ANY_VALUE])
                .map(|c| BSTree {
                    value: Literal::Value(*c),
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
                    let mut diff = alphabet
                        .difference(&subnode_alphabet)
                        .chain(&[ANY_VALUE])
                        .map(|c| BSTree {
                            value: Literal::Value(*c),
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
                    //if this panics is probably some weird nonsense regexp such as ~.
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
                        value: Literal::Value(EPSILON_VALUE),
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

struct DCHelper {
    ttype: Literal,
    index: usize,
    nullable: bool,
    firstpos: BTreeSet<usize>,
    lastpos: BTreeSet<usize>,
}

fn build_dc_helper(node: &BSTree<Literal>, start_index: usize) -> BSTree<DCHelper> {
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
        Literal::Value(val) => {
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
fn compute_followpos(node: &BSTree<DCHelper>, graph: &mut Vec<BTreeSet<usize>>) {
    if let Some(l) = &node.left {
        compute_followpos(&*l, graph);
    }
    if let Some(r) = &node.right {
        compute_followpos(&*r, graph);
    }
    match &node.value.ttype {
        Literal::Value(_) => {}
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

fn retrieve_idx_and_acc(
    node: &BSTree<DCHelper>,
    indices: &mut Vec<char>,
    acc: &mut HashMap<usize, usize>,
) {
    if let Some(l) = &node.left {
        retrieve_idx_and_acc(&*l, indices, acc);
    }
    if let Some(r) = &node.right {
        retrieve_idx_and_acc(&*r, indices, acc);
    }
    match &node.value.ttype {
        Literal::Value(val) => indices[node.value.index] = *val,
        Literal::Acc(prod) => {
            acc.insert(node.value.index, *prod);
        }
        _ => {}
    }
}

fn direct_sc(
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
    let mut accept = BTreeSet::new();
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
                    accept.insert((u_idx, *acc_prod));
                }
                unmarked.push(u.clone());
                done.insert(u, u_idx);
            }
            let set_idx = *done.get(&node_set).unwrap();
            tran.insert((set_idx, *letter), u_idx);
        }
    }
    DFA {
        states_no: index + 1,
        transition: tran,
        accept,
    }
}

fn direct_construction(node: BSTree<Literal>) -> DFA {
    let helper = build_dc_helper(&node, 0);
    let mut indices = vec![EPSILON_VALUE; helper.value.index + 1];
    let mut followpos = vec![BTreeSet::new(); helper.value.index + 1];
    let mut accepting = HashMap::new();
    retrieve_idx_and_acc(&helper, &mut indices, &mut accepting);
    compute_followpos(&helper, &mut followpos);
    direct_sc(helper.value.firstpos, &followpos, &indices, &accepting)
}

fn collect_productions(nodes: Vec<BSTree<Literal>>) -> BSTree<Literal> {
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

// fn min_dfa(dfa: DFA) -> DFA {
//     let acc = dfa.accept.iter().map(|x| x.0).collect::<BTreeSet<_>>();
//     let non_acc = (0 as usize..)
//         .take(dfa.states_no)
//         .difference(&accepting)
//         .collect::<BTreeSet<_>>();
//     let mut partitions = btreeset! {acc, non_acc};
//     let mut new_partitions;
//     loop {
//
//     }
// }
