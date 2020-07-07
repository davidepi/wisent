use std::iter::{Enumerate, Peekable};
use std::str::Chars;

use crate::error::ParseError;

pub(super) struct TreeNode<T> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

pub(super) struct Operator<'a> {
    r#type: OpType,
    value: &'a str,
    priority: u8,
}

#[derive(PartialEq, Debug)]
enum OpType {
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

fn consume_counting_until(
    it: &mut Enumerate<Peekable<Chars>>,
    until: char,
) -> Result<usize, ParseError> {
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
    if last_char == until {
        Ok(skipped)
    } else {
        Err(ParseError::SyntaxError {
            message: format!("Unmatched {}, reached end of file", until),
        })
    }
}

//No clippy, this is not more readable.
#[allow(clippy::useless_let_if_seq)]
fn read_token<'a>(
    input: &'a str,
    first: char,
    it: &mut Enumerate<Peekable<Chars>>,
) -> Result<&'a str, ParseError> {
    match first {
        '.' => Ok(&input[..2]),
        '[' => {
            let counted = consume_counting_until(it, ']')?;
            Ok(&input[..counted + 2]) //2 is to match [], plus all the bytes counted inside
        }
        '\'' => {
            let mut counted = consume_counting_until(it, '\'')?;
            let mut id = &input[..counted + 2]; //this is a valid literal. check for range ''..''
            if input.len() > counted + 5 && &input[(counted + 2)..(counted + 5)] == "..\'" {
                //this is actually a range ''..'' so first advance the iterator by 3 pos
                it.next();
                it.next();
                it.next();
                //then update the counter for the literal length by accounting also the new lit.
                counted += consume_counting_until(it, '\'')?;
                id = &input[..counted + 6];
            }
            Ok(id)
        }
        _ => Err(ParseError::SyntaxError {
            message: format!("Unsupported literal {}", input),
        }),
    }
}

fn combine_nodes<'a>(
    operands: &mut Vec<TreeNode<Operator<'a>>>,
    operators: &mut Vec<Operator<'a>>,
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
    let ret = TreeNode {
        value: operator,
        left,
        right,
    };
    operands.push(ret);
}

fn tokenize(regex: &str) -> Result<Vec<Operator>, ParseError> {
    let mut tokenz = Vec::<Operator>::new();
    let mut balanced = 0;
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
                balanced += 1;
            }
            ')' => {
                tp = OpType::RP;
                val = &regex[index..index + 1];
                priority = 5;
                balanced -= 1;
            }
            _ => {
                tp = OpType::ID;
                priority = 0;
                val = read_token(&regex[index..], char, &mut iter)?;
            }
        };
        if !tokenz.is_empty() && implicit_concatenation(&tokenz.last().unwrap().r#type, &tp) {
            tokenz.push(Operator {
                r#type: OpType::AND,
                value: "&",
                priority: 2,
            })
        }
        tokenz.push(Operator {
            r#type: tp,
            value: val,
            priority,
        });
    }
    if balanced == 0 {
        Ok(tokenz)
    } else {
        Err(ParseError::SyntaxError {
            message: format!("Unmatched parentheses in {}", regex),
        })
    }
}

fn implicit_concatenation(last: &OpType, current: &OpType) -> bool {
    let last_is_kleene_family =
        *last == OpType::KLEENE || *last == OpType::PL || *last == OpType::QM;
    let cur_is_lp_or_id = *current == OpType::LP || *current == OpType::ID;
    (last_is_kleene_family && cur_is_lp_or_id)
        || (*last == OpType::RP && (*current == OpType::NOT || cur_is_lp_or_id))
        || (*last == OpType::ID && (*current == OpType::NOT || cur_is_lp_or_id))
}

pub(super) fn gen_parse_tree(regex: &str) -> Result<TreeNode<Operator>, ParseError> {
    let mut operands = Vec::new();
    let mut operators: Vec<Operator> = Vec::new();
    let tokens = tokenize(&regex)?;
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
                let leaf = TreeNode {
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
    Ok(tree)
}

pub(super) fn tree_json(node: &TreeNode<Operator>, stream: &mut String) {
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
