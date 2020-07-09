use std::collections::HashSet;
use std::iter::{Enumerate, Peekable};
use std::str::Chars;

use crate::error::ParseError;

pub(super) struct BSTree<T> {
    value: T,
    left: Option<Box<BSTree<T>>>,
    right: Option<Box<BSTree<T>>>,
}

pub(super) struct RegexOperation<'a> {
    r#type: OpType,
    value: &'a str,
    priority: u8,
}

pub(super) enum Literal {
    Value(char),
    AnyValue,
    Operation(OpType),
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

fn consume_counting_until(it: &mut Enumerate<Peekable<Chars>>, until: char) -> usize {
    let mut escapes = 0;
    let mut skipped = 0;
    let mut last_char = '\0';
    for skip in it {
        last_char = skip.1;
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
        '.' => &input[..2],
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

fn combine_nodes<'a>(
    operands: &mut Vec<BSTree<RegexOperation<'a>>>,
    operators: &mut Vec<RegexOperation<'a>>,
) {
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

fn tokenize(regex: &str) -> Vec<RegexOperation> {
    let mut tokenz = Vec::<RegexOperation>::new();
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
            tokenz.push(RegexOperation {
                r#type: OpType::AND,
                value: "&",
                priority: 2,
            })
        }
        tokenz.push(RegexOperation {
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

pub(super) fn gen_parse_tree(regex: &str) -> BSTree<RegexOperation> {
    let mut operands = Vec::new();
    let mut operators: Vec<RegexOperation> = Vec::new();
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
    let tree = operands.pop().unwrap();
    tree
}

pub(super) fn tree_json(node: &BSTree<RegexOperation>, stream: &mut String) {
    stream.push('{');
    stream.push_str(&format!("\"val\":\"{}\"", &node.value.value));
    if let Some(left) = &node.left {
        stream.push_str(",\"left\":");
        tree_json(left, stream);
    }
    if let Some(right) = &node.right {
        stream.push_str(",\"right\":");
        tree_json(right, stream);
    }
    stream.push('}');
}

fn collect_alphabet(parse_tree: BSTree<RegexOperation>) -> (BSTree<Literal>, HashSet<char>) {
    let mut todo_stack = vec![parse_tree];
    let mut new_stack = Vec::new();
    let mut literals = HashSet::new();
    while let Some(node) = todo_stack.pop() {
        if node.value.r#type == OpType::ID {
        } else {
            if let Some(right) = node.right {
                todo_stack.push(*right);
            }
            if let Some(left) = node.left {
                todo_stack.push(*left);
            }
        }
    }
    (new_stack.pop().unwrap(), literals)
}

pub(super) fn expand_literal_node(literal: &str, char_set: &mut HashSet<char>) -> BSTree<Literal> {
    if literal == "." {
        return BSTree {
            value: Literal::AnyValue,
            left: None,
            right: None,
        };
    }
    let mut charz = Vec::new();
    let mut iter = literal.chars();
    let start = iter.next().unwrap();
    let end;
    let mut last = '\x00';
    let mut set_op;
    if start == '[' {
        end = ']';
        set_op = OpType::OR;
    } else {
        end = '\'';
        set_op = OpType::AND;
    }
    while let Some(char) = iter.next() {
        let mut pushme = char;
        if char == '\\' {
            //escaped char
            pushme = unescape_character(iter.next().unwrap(), &mut iter);
            last = pushme;
            char_set.insert(pushme);
            charz.push(BSTree {
                value: Literal::Value(pushme),
                left: None,
                right: None,
            });
        } else if set_op == OpType::OR && char == '-' {
            //set in form a-z, A-Z, 0-9, etc..
            let from = last as u32 + 1;
            let until = iter.next().unwrap() as u32 + 1; //included
            for i in from..until {
                pushme = std::char::from_u32(i).unwrap();
                char_set.insert(pushme);
                charz.push(BSTree {
                    value: Literal::Value(pushme),
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
            char_set.insert(pushme);
            charz.push(BSTree {
                value: Literal::Value(pushme as char),
                left: None,
                right: None,
            });
        }
    }
    //check possible range in form 'a'..'z', at this point I ASSUME this can be a literal only
    //and the syntax has already been checked.
    if let Some(char @ '.') = iter.next() {
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
            char_set.insert(pushme);
            charz.push(BSTree {
                value: Literal::Value(pushme),
                left: None,
                right: None,
            });
        }
    }
    while charz.len() > 2 {
        let left = charz.pop().unwrap();
        let right = charz.pop().unwrap();
        let new = BSTree {
            value: Literal::Operation(set_op),
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
            //FIXME: the ANTLR reference specifies that an unicode character should be encoded as
            //       \uXXXX, however, an unicode char can have also 5 digits 🤔.
            //       For example this emoji is U+1F914
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
