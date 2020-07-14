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
    start: NFANode,
    accept: BTreeSet<(NFANode, usize)>,
}

impl NFA {
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
                }
                write!(f, "{}->{}[label=\"{}\"];", source, target, symbol)?;
            }
        }
        write!(f, "}}")
    }
}

fn thompson_construction(prod: &BSTree<Literal>, start_index: usize, production: usize) -> NFA {
    let mut index = start_index;
    let mut visit = vec![prod];
    let mut todo = Vec::new();
    let mut done = Vec::<NFA>::new();
    //first transform the parse tree into a stack, this will be the processing order
    while let Some(node) = visit.pop() {
        if let Some(l) = &node.left {
            visit.push(*&l);
        }
        if let Some(r) = &node.right {
            visit.push(*&r);
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
                first.accept = btreeset! {(new_end, production)};
                first.states_no += second.states_no + 2;
                pushme = first;
            }
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
    let ret;
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
        ret = NFA {
            states_no: index + 1,
            transition: transition_table,
            start: index,
            accept,
        }
    } else {
        ret = thompson_nfa.pop().unwrap();
    }
    ret
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
    KLEENE,
    AND,
    OR,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Value(i) => write!(f, "{}", i),
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
