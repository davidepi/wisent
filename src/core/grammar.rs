use std::collections::HashMap;
use std::{iter::Peekable, str::Chars};

use crate::error::ParseError;

#[derive(Debug)]
pub struct Grammar {
    //TODO: whis entire struct will be changed probably
    pub productions: Vec<String>,
    fragments: HashMap<String, String>,
}

pub fn parse_grammar(path: &str) -> Result<Grammar, ParseError> {
    let grammar_content = std::fs::read_to_string(path)?;
    let productions = retrieve_productions(&grammar_content);
    let productions_no_nl = remove_newlines(productions);
    let grammar = replace_fragments(productions_no_nl);
    //TODO: used for debug, removeme
    // for fragment in &grammar.fragments {
    //     println!("{} -> {}", fragment.0, fragment.1);
    // }
    Ok(grammar)
}

/// Replaces all `\n` and `\r` chars with a whitespace.
/// ## Arguments
/// * `production` The vector of String representing each production
/// ## Returns
/// A vector of Strings containing the input productions withouth `\r` or `\n`
fn remove_newlines(productions: Vec<String>) -> Vec<String> {
    let ret = productions
        .into_iter()
        .map(|s| {
            s.chars()
                .map(|c| match c {
                    '\n' => ' ',
                    '\r' => ' ',
                    _ => c,
                })
                .collect()
        })
        .collect();
    ret
}

fn replace_fragments(productions: Vec<String>) -> Grammar {
    let mut fragments = HashMap::new();
    let mut prod_wo_frags = Vec::new();
    // find all fragments and put them in an hash map. A fragment start with
    // the `fragment` keyword.
    for production in productions {
        if production.starts_with("fragment") && production.chars().nth(8).unwrap().is_whitespace()
        {
            let fragment = &production[8..];
            let mut splitter = fragment.splitn(2, ':');
            let name = splitter.next().unwrap().trim().to_string();
            //TODO: maybe check syntax of names? (first letter uppercase)
            let rule = splitter.next().unwrap().trim().to_string();
            fragments.insert(name, rule);
        } else {
            prod_wo_frags.push(production);
        }
    }
    for fragment in &fragments {
        if fragment.0.chars().next().unwrap().is_lowercase() {
            //TODO: need a custom error type :'(
        }
    }
    Grammar {
        productions: prod_wo_frags,
        fragments,
    }
}

/// Retrieves every production from a `.g4` grammar.
/// This effectively works by removing every comment and then splitting over ;
/// tokens that are not quoted, although in this functions is implemented as a
/// single pass.
/// The comments removed are the multiline `/*`-`*/` and single line `//`, `#`.
/// ## Arguments
/// * `content` - A string containing the original `.g4` grammar content.
/// ## Returns
/// A vector of string representing the productions of the original grammar.
/// Each element represents a single production.
fn retrieve_productions(content: &String) -> Vec<String> {
    let mut productions = Vec::new();
    let mut ret = String::new();
    let mut it = content.chars().peekable();
    while let Some(letter) = it.next() {
        //get lookahead
        let lookahead = match it.peek() {
            Some(l) => *l,
            None => '\0',
        };
        match letter {
            '/' => {
                if lookahead == '*' {
                    it.next(); //position to the lookahead
                               //skip until */ is found
                    while let Some(skip) = it.next() {
                        if skip == '*' {
                            if let Some(lahead) = it.peek() {
                                if *lahead == '/' {
                                    it.next(); //skip also the lookahead
                                    break;
                                }
                            }
                        }
                    }
                } else if lookahead == '/' {
                    consume_line(&mut it);
                } else {
                    ret.push(letter);
                }
            }
            '\'' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, '\'')
            }
            '[' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, ']');
            }
            ';' => {
                // end of the production and start of a new one
                // don't push the ; as it's not needed from now on
                productions.push(ret.trim().to_string());
                ret.clear();
            }
            _ => ret.push(letter),
        }
    }
    //remove productions without colon. This should remove the grammar XX; stmt.
    //very naive as a proper check for escaped char will be performed later.
    productions = productions
        .into_iter()
        .filter(|s| s.contains(':'))
        .collect();
    productions
}

/// Advances the iterator until the next `\n` character.
/// Alsoi the last `\n` is discarded.
/// ## Arguments
/// * `it` The iterator that will be advanced
/// * `ret` The string where the final \n will be appended
fn consume_line(it: &mut Peekable<Chars>) {
    while let Some(skip) = it.next() {
        if skip == '\n' {
            break;
        }
    }
}

/// Advances the iterator until the given character and appends all the
/// encountered characters. This function takes into account also escape
/// character, so if the given charcater is ', this won't  stop in case a \' is
/// encountered.
/// ## Arguments
/// * `it` - The iterator that will be advanced
/// * `ret` - The string where the various character will be appended
/// * `until` - The character that will stop the method. Escaped versions of
/// this character won't be considered
fn append_until(it: &mut Peekable<Chars>, ret: &mut String, until: char) {
    let mut escapes = 0;
    while let Some(push) = it.next() {
        ret.push(push);
        if push == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if push == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
    }
}
