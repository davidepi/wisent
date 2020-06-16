use std::collections::HashMap;
use std::{iter::Peekable, str::Chars};

use regex::Regex;

use crate::error::ParseError;

#[derive(Debug)]
pub struct Grammar {
    pub terminals: HashMap<String, String>,
    pub non_terminals: HashMap<String, String>,
    fragments: HashMap<String, String>,
}

impl Grammar {
    /// Returns the total number of productions. This includes terminals and
    /// non-terminals but not fragments.
    /// ## Returns
    /// A number representing the sum of terminals and non-terminals productions
    pub fn len(self) -> usize {
        return self.terminals.len() + self.non_terminals.len();
    }
}

pub fn parse_grammar(path: &str) -> Result<Grammar, ParseError> {
    let grammar_content = std::fs::read_to_string(path)?;
    let productions = retrieve_productions(&grammar_content);
    let grammar = build_grammar(productions)?;
    // validate productions and categorize them into terminal or non-terminal
    // based on their uppercase or lowercase letter
    Ok(grammar)
}

/// Categorizes various productions into terminal, non-terminal and
/// fragments. Fragments will be reduced to a production in the form `S->'...'`.
/// Then returns a Grammar object.
/// ## Arguments
/// * `productions` - A vector of String where each String is a production
/// ending with `;`. This vector may contain fragments, but in this case the
/// fragment keyword must be passed as well.
/// ## Returns
/// A Result object with the following types:
/// * `Ok(Grammar)` - A Grammar object containing the parsed productions
/// * `Err(ParseError)` - A ParseError object containing a description of the
/// error
///
fn build_grammar(productions: Vec<String>) -> Result<Grammar, ParseError> {
    let mut terminals = HashMap::new();
    let mut non_terminals = HashMap::new();
    let mut fragments = HashMap::new();
    //the capt.group of this regex will be passed to p_re so I need to include ;
    let f_re = r"\s*fragment\s+((?:.|\n)+;)";
    let p_re = r"\s*(\w+)\s*:\s*((?:.|\n)+);";
    let re_fr = Regex::new(f_re).unwrap(); //fragment detection
    let re_pd = Regex::new(p_re).unwrap(); //production detection
    for production in &productions {
        let mut is_fragment = false;
        let mut prod = &production[..];
        if let Some(matches) = re_fr.captures(prod) {
            prod = matches.get(1).map_or("", |m| m.as_str());
            is_fragment = true;
        }
        match re_pd.captures(prod) {
            Some(matches) => {
                let name = matches.get(1).map_or("", |m| m.as_str()).to_string();
                let rule = matches.get(2).map_or("", |m| m.as_str()).to_string();
                if name.chars().next().unwrap().is_lowercase() {
                    if !is_fragment {
                        non_terminals.insert(name, rule);
                    } else {
                        return Err(ParseError::SyntaxError {
                            message: format!("Fragments should be lowercase {}", production),
                        });
                    }
                } else {
                    if !is_fragment {
                        terminals.insert(name, rule);
                    } else {
                        fragments.insert(name, rule);
                    }
                }
            }
            None => {
                return Err(ParseError::SyntaxError {
                    message: format!("Unknown production: {}", prod),
                });
            }
        }
    }
    Ok(Grammar {
        terminals,
        non_terminals,
        fragments,
    })
}

/// Retrieves every production from a `.g4` grammar.
/// This effectively works by removing every comment and then splitting over ;
/// tokens that are not quoted, although in this functions is implemented as a
/// single pass.
/// The comments removed are the multiline `/*`-`*/` and single line `//`, `#`.
/// ## Arguments
/// * `content` - A string containing the original `.g4` grammar content.
/// * `filename` - The name of the original file parsed.
/// ## Returns
/// A vector of string representing the productions of the original grammar.
/// Each element represents a single production.
fn retrieve_productions(content: &str) -> Vec<String> {
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
                ret.push(letter);
                productions.push(ret);
                ret = String::new();
            }
            _ => ret.push(letter),
        }
    }
    //remove productions without colon. This should remove the grammar XX; stmt.
    //very naive as a proper check for escaped char will be performed later.
    productions
        .into_iter()
        .filter(|s| s.contains(':'))
        .collect()
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
