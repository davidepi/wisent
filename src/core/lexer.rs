use std::iter::{Enumerate, Peekable};
use std::str::Chars;
use std::vec;

use regex::Regex;

use crate::error::ParseError;

pub(super) struct TreeNode<T> {
    node_type: Operators,
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

#[derive(PartialEq)]
enum Operators {
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
                counted = counted + consume_counting_until(it, '\'')?;
                id = &input[..counted + 6];
            }
            Ok(id)
        }
        _ => Err(ParseError::SyntaxError {
            message: format!("Unsupported literal {}", "all"),
        }),
    }
}

fn combine_nodes(operands: &mut Vec<TreeNode<&str>>, operators: &mut Vec<Operators>) {
    let operator = operators.pop().unwrap();
    let left;
    let right;
    if operator == Operators::OR {
        //binary operator
        right = Some(Box::new(operands.pop().unwrap()));
        left = Some(Box::new(operands.pop().unwrap()));
    } else {
        // unary operator
        left = Some(Box::new(operands.pop().unwrap()));
        right = None;
    };
    let ret = TreeNode {
        node_type: operator,
        value: "",
        left,
        right,
    };
    operands.push(ret);
}

pub(super) fn gen_parse_tree(regex: &str) -> Result<TreeNode<&str>, ParseError> {
    let mut operands = Vec::new();
    let mut operators = Vec::new();
    let mut iter = regex.chars().peekable().enumerate();
    let mut consecutive_ids = false; //if last was ID and current is ID add a concatenation
    while let Some((index, char)) = iter.next() {
        let operator = match char {
            '*' => Operators::KLEENE,
            '|' => Operators::OR,
            '?' => Operators::QM,
            '+' => Operators::PL,
            '~' => Operators::NOT,
            '(' => Operators::LP,
            ')' => Operators::RP,
            _ => Operators::ID,
        };
        if operator != Operators::ID && consecutive_ids {
            consecutive_ids = false;
        }
        match operator {
            //operators after operand -> solve last in stack and ALSO this (as I have the operand)
            Operators::KLEENE | Operators::QM | Operators::PL => {
                if !operators.is_empty() && *operators.last().unwrap() != Operators::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.push(operator);
                combine_nodes(&mut operands, &mut operators);
            }
            //operators before operand, solve last in stack and push this
            Operators::OR | Operators::NOT => {
                if !operators.is_empty() && *operators.last().unwrap() != Operators::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.push(operator);
            }
            Operators::LP => operators.push(operator),
            Operators::RP => {
                while !operators.is_empty() && *operators.last().unwrap() != Operators::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
            Operators::ID => {
                let leaf = TreeNode {
                    node_type: Operators::ID,
                    value: read_token(&regex[index..], char, &mut iter)?,
                    left: None,
                    right: None,
                };
                operands.push(leaf);
                if consecutive_ids {
                    operators.push(Operators::AND);
                    combine_nodes(&mut operands, &mut operators);
                }
                consecutive_ids = true;
            }
            Operators::AND => {}
        }
    }
    //solve all remaining operators (usually unary ones)
    while !operators.is_empty() {
        combine_nodes(&mut operands, &mut operators);
    }
    //concatenate all the remaining nodes
    while operands.len() > 1 {
        let right = Some(Box::new(operands.pop().unwrap()));
        let left = Some(Box::new(operands.pop().unwrap()));
        let node = TreeNode {
            node_type: Operators::AND,
            value: "",
            left,
            right,
        };
        operands.push(node);
    }
    let tree = operands.pop().unwrap();
    Ok(tree)
}

pub(super) fn tree_json(node: &TreeNode<&str>, stream: &mut String) {
    stream.push('{');
    if node.node_type != Operators::ID {
        stream.push_str("\"id\":\"");
        match node.node_type {
            Operators::OR => stream.push('|'),
            Operators::PL => stream.push('+'),
            Operators::QM => stream.push('?'),
            Operators::NOT => stream.push('~'),
            Operators::KLEENE => stream.push('*'),
            Operators::AND => stream.push('&'),
            _ => {}
        }
        stream.push_str("\",\"val\":[");
        if let Some(left) = &node.left {
            tree_json(left, stream);
        } else {
            stream.push_str("{}");
        }
        stream.push(',');
        if let Some(right) = &node.right {
            tree_json(right, stream);
        } else {
            stream.push_str("{}");
        }
        stream.push(']');
    } else {
        stream.push_str("\"id\":\"ID\",\"val\":\"");
        stream.push_str(node.value);
        stream.push('"');
    }
    stream.push('}');
}
