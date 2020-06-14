use std::{iter::Peekable, str::Chars};

pub struct Grammar {
    pub content: String, //TODO: whis will be changed probably
}

pub fn parse_grammar(path: &str) -> std::io::Result<Grammar> {
    let grammar_content = std::fs::read_to_string(path)?;
    let grammar_no_comments = remove_comments(&grammar_content);
    return Ok(Grammar {
        content: grammar_no_comments,
    });
}

/// Removes all the comments from a `.g4` grammar.
/// The comments removed are the multiline `/*`-`*/` and single line `//`, `#`.
/// ## Arguments
/// * `content` - A string containing the original `.g4` grammar content.
/// ## Returns
/// A new string representing the content of the original grammar, without
/// comments
fn remove_comments(content: &String) -> String {
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
                                    it.next(); //drop the lookahead
                                    break;
                                }
                            }
                        }
                    }
                } else if lookahead == '/' {
                    consume_line(&mut it, &mut ret);
                } else {
                    ret.push(letter);
                }
            }
            '#' => {
                consume_line(&mut it, &mut ret);
            }
            '\'' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, '\'')
            }
            '[' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, ']');
            }
            _ => ret.push(letter),
        }
    }
    ret
}

/// Advances the iterator until the next `\n` character.
/// Only the last `\n` is appended to the string passed as input
/// ## Arguments
/// * `it` The iterator that will be advanced
/// * `ret` The string where the final \n will be appended
fn consume_line(it: &mut Peekable<Chars>, ret: &mut String) {
    while let Some(skip) = it.next() {
        if skip == '\n' {
            ret.push(skip);
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
