use std::{iter::Peekable, str::Chars};

pub struct Grammar {
    //TODO: whis entire struct will be changed probably
    pub productions: Vec<String>,
}

pub fn parse_grammar(path: &str) -> std::io::Result<Grammar> {
    let grammar_content = std::fs::read_to_string(path)?;
    let productions = retrieve_productions(&grammar_content);
    return Ok(Grammar { productions });
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
            '#' => {
                consume_line(&mut it);
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
                ret.push(letter);
                productions.push(ret.trim().to_string());
                ret.clear();
            }
            _ => ret.push(letter),
        }
    }
    //remove the first production if it's grammar XX;
    if productions.len() > 0 && &productions[0][0..7] == "grammar" {
        productions.drain(0..1);
    }
    productions
}

/// Advances the iterator until the next `\n` character.
/// Only the last `\n` is appended to the string passed as input
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
