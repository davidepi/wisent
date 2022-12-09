use super::{SymbolTable, UnicodeReader};
use crate::error::ParseError;
use crate::grammar::Action;
use crate::lexer::MultiDfa;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// How many characters are read when populating the buffer of the simulator.
/// In any case the simulator uses buffered reads, but each read requires some allocations, so it's
/// better to cache them anyway.
/// This size is expressed in bytes, but doe to how UTF8 works there may be up to 3 additional
/// bytes read.
const READ_SIZE: usize = 1024;

#[derive(Debug, Copy, Clone)]
/// Token retrieved by the lexical analyzer.
pub struct Token {
    /// The mode ID of the associated production.
    pub mode: u32,
    /// The production ID (in the grammar used to build the DFA) associated with this token.
    pub production: u32,
    /// Beginning of the token in number of bytes since the beginning of the stream.
    pub start: usize,
    /// End of the token in number of bytes since the beginning of the stream.
    /// If the last character of the token is a multi-byte character this value correspond to the
    /// last byte of the character.
    pub end: usize,
}

/// Lexical analyzer for a DFA
///
/// Simulates a Dfa with a given input and groups the input characters in tokens according to the
/// rules of the grammar passed to the [`Dfa`].
pub struct DfaSimulator<'a, I: Iterator<Item = Result<u8, std::io::Error>>> {
    /// Buffer storing the read characters converted using their symbol table id.
    buffer: VecDeque<u32>,
    /// Position of the next character to read from the internal buffer in amount of characters
    /// from buf start. Note that internal buffer is NOT equal to the input.
    forward_pos: usize,
    /// Current token position, in bytes from the beginning of the input.
    cur_pos: usize,
    /// Starting token position, in bytes, from the beginning of the input. This is almost always
    /// identical to cur_pos, unless the previous token had [Action::More].
    start_pos: usize,
    /// DFA containing the moves and alphabet
    mdfa: &'a MultiDfa,
    /// Current mode being simulated. Represented as stack to allow PUSHMODE and POPMODE
    current_mode: Vec<u32>,
    /// Current input being processed
    input: UnicodeReader<I>,
}

impl<'a, I: Iterator<Item = Result<u8, std::io::Error>>> DfaSimulator<'a, I> {
    /// Creates a new Lexical Analyzer with the given DFA and the given byte iterator.
    ///
    /// Consider using the methods [`tokenize_string`] and [`tokenize_file`] if fine grained
    /// control over each token is not needed.
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::{MultiDfa, DfaSimulator};
    /// # use std::io::{BufReader, Read};
    /// let grammar = Grammar::new(
    ///     &[("NUMBER", "([0-9])+").into(), ("WORD", "([a-z])+").into()],
    ///     &[],
    /// );
    /// let dfa = MultiDfa::new(&grammar);
    /// let input = "abc123";
    /// let simulator = DfaSimulator::new(&dfa, BufReader::new(input.as_bytes()).bytes());
    /// ```
    pub fn new(dfas: &'a MultiDfa, input: I) -> DfaSimulator<'a, I> {
        Self {
            buffer: VecDeque::with_capacity(READ_SIZE),
            forward_pos: 0,
            cur_pos: 0,
            start_pos: 0,
            mdfa: dfas,
            current_mode: vec![0],
            input: UnicodeReader::new(input),
        }
    }

    /// Runs the lexical analysis and retrieves the next token.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::{MultiDfa, DfaSimulator};
    /// # use std::io::{BufReader, Read};
    /// let grammar = Grammar::new(
    ///     &[("NUMBER", "([0-9])+").into(), ("WORD", "([a-z])+").into()],
    ///     &[],
    /// );
    /// let dfa = MultiDfa::new(&grammar);
    /// let input = "abc123";
    /// let mut chars = input.chars();
    /// let mut simulator = DfaSimulator::new(&dfa, BufReader::new(input.as_bytes()).bytes());
    /// let token = simulator.next_token().unwrap().unwrap();
    /// assert_eq!(token.production, 1);
    /// ```
    pub fn next_token(&mut self) -> Result<Option<Token>, ParseError> {
        let dfa = &self.mdfa[*self.current_mode.last().unwrap() as usize];
        let absolute_start_index = self.start_pos;
        let mut absolute_end_index = self.cur_pos;
        let mut state = dfa.start();
        let mut last_accepted = None;
        // Read as much as possible until the DFA halts
        while let Some((char_id, bytes_for_this_char)) = self.next_char()? {
            absolute_end_index += bytes_for_this_char as usize;
            if let Some(next) = dfa.moove(state, char_id) {
                //can advance
                state = next;
                if let Some(accepting_prod) = dfa.accepting(state) {
                    // if the new state is accepting, record the production and the current
                    // lexeme ending. DO NOT push the accepting state, as we try to greedily match
                    // other productions.
                    let actions = dfa.actions(state);
                    last_accepted = Some((
                        accepting_prod,
                        actions,
                        self.forward_pos,
                        absolute_end_index,
                    ));
                    if dfa.non_greedy(state) {
                        // if the current accepting state is non-greedy, immediately process acc
                        break;
                    }
                }
            } else {
                // DFA halts. stop reading chars and start solving accepting state.
                break;
            }
        }
        // Check if an accepting state was reached and handle it.
        if let Some((production, actions, lexeme_chars, last_valid_end)) = last_accepted {
            // backtrack the lookaheads to the last valid position
            self.cur_pos = last_valid_end;
            // no other move available, but there was a previous accepted production.
            // push the accepted production if the action is not Skip.
            let token = if !actions.contains(&Action::Skip) && !actions.contains(&Action::More) {
                Some(Token {
                    production,
                    start: absolute_start_index,
                    end: last_valid_end,
                    mode: *self.current_mode.last().unwrap(),
                })
            } else {
                None
            };
            if !actions.contains(&Action::More) {
                for _ in 0..lexeme_chars {
                    self.buffer.pop_front();
                }
                self.forward_pos = 0;
                self.start_pos = last_valid_end;
            } else {
                // 'MORE': forward_pos should not be reset, but at least the lookahead removed
                self.forward_pos = lexeme_chars;
            }
            // handle mode switching
            for action in actions {
                match action {
                    Action::Mode(m) => {
                        *self.current_mode.last_mut().unwrap() = *m;
                    }
                    Action::PushMode(m) => self.current_mode.push(*m),
                    Action::PopMode => {
                        if self.current_mode.len() == 1 {
                            return Err(ParseError::InternalError {
                                message: "attempt to pop last mode".to_string(),
                            });
                        } else {
                            self.current_mode.pop();
                        }
                    }
                    _ => (),
                }
            }
            if token.is_some() {
                Ok(token)
            } else {
                self.next_token()
            }
        } else {
            Ok(None)
        }
    }

    /// Reads the next character from the input in form of [`SymbolTable`] IDs.
    ///
    /// Takes care of buffering the read and handling the buffers.
    ///
    /// DO NOT change the `input` until EOF is returned!
    ///
    /// Returns the ID and the number of bytes used to represent it and None if EOF was reached.
    /// Return io::Error if some problems were encountered while reading the input.
    fn next_char(&mut self) -> Result<Option<(u32, u8)>, std::io::Error> {
        // checks if the buf needs refilling
        if self.forward_pos == self.buffer.len() {
            let chars = (&mut self.input)
                .take(READ_SIZE)
                .collect::<Result<Vec<char>, std::io::Error>>()?;
            let encoded_it = chars
                .into_iter()
                .map(|c| encode_char_len(c, self.mdfa.symbol_table()));
            self.buffer.extend(encoded_it);
        }
        if let Some(&char) = self.buffer.get(self.forward_pos) {
            // refilling successfull, return the next character
            self.forward_pos += 1;
            Ok(Some(decode_char_len(char)))
        } else {
            // refilling failed, EOF reached
            Ok(None)
        }
    }
}

/// In order to save space, the length of a char in bytes is written in the same value as the
/// symtable ID (only 2 bits are used)
fn encode_char_len(c: char, symtab: &SymbolTable) -> u32 {
    // 0x3EFFFFFF: symbol id
    // 0xC0000000: number of bytes to represent the char -1
    (symtab.symbol_id(c) & 0x3EFFFFFF) | ((c.len_utf8() - 1) as u32) << 30
}

/// Inverse of the encode_char_len: retrieves the original symtable ID and the character len in
/// bytes
fn decode_char_len(v: u32) -> (u32, u8) {
    (v & 0x3EFFFFFF, ((v & 0xC0000000) >> 30) as u8 + 1)
}

/// Tokenize a string with a given DFA.
///
/// This utility function is a wrapper that creates a [DfaSimulator], calls its
/// [next_token](DfaSimulator::next_token) method and returns all the tokens at once.
///
/// # Examples
/// Tokenize string:
/// ```
/// # use wisent::grammar::Grammar;
/// # use wisent::lexer::{MultiDfa, DfaSimulator, tokenize_string};
/// let grammar = Grammar::new(
///     &[("NUMBER", "([0-9])+").into(), ("WORD", "([a-z])+").into()],
///     &[],
/// );
/// let dfa = MultiDfa::new(&grammar);
/// let input = "abc123";
/// let result = tokenize_string(&dfa, &input).unwrap();
/// assert_eq!(result.len(), 2);
/// ```
pub fn tokenize_string(dfa: &MultiDfa, input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    if !input.is_empty() {
        let reader = BufReader::new(input.as_bytes());
        let mut simulator = DfaSimulator::new(dfa, reader.bytes());
        while let Some(token) = simulator.next_token()? {
            tokens.push(token);
        }
    }
    Ok(tokens)
}

/// Tokenize the content of a file with a given DFA.
///
/// This utility function is a wrapper that opens a File, creates a [DfaSimulator], calls its
/// [next_token](DfaSimulator::next_token) method and returns all the tokens at once.
///
/// Reads from the file are buffered, so the file content is never in memory all at once.
pub fn tokenize_file<P: AsRef<Path>>(dfa: &MultiDfa, path: P) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut simulator = DfaSimulator::new(dfa, reader.bytes());
    while let Some(token) = simulator.next_token()? {
        tokens.push(token);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::tokenize_string;
    use crate::error::ParseError;
    use crate::grammar::{Action, Grammar};
    use crate::lexer::simulator::READ_SIZE;
    use crate::lexer::{DfaSimulator, MultiDfa};
    use maplit::btreeset;
    use std::io::{BufReader, Read};
    use std::iter::repeat;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn simulator_next_char() -> Result<(), ParseError> {
        // input smaller than BUFFER_SIZE, extra logic for the buffer swap is in the tokenize
        // function
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let reader = BufReader::new(UTF8_INPUT.as_bytes()).bytes();
        let mut simulator = DfaSimulator::new(&dfa, reader);
        for _ in 0..UTF8_INPUT.chars().count() {
            assert!(simulator.next_char()?.is_some());
        }
        assert!(simulator.next_char()?.is_none());
        Ok(())
    }

    #[test]
    fn simulator_multiple_eof() -> Result<(), ParseError> {
        // after next_char returns eof once, additional calls return always eof
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let reader = BufReader::new(UTF8_INPUT.as_bytes()).bytes();
        let mut simulator = DfaSimulator::new(&dfa, reader);
        while simulator.next_char()?.is_some() {}
        assert!(simulator.next_char()?.is_none());
        assert!(simulator.next_char()?.is_none());
        assert!(simulator.next_char()?.is_none());
        Ok(())
    }

    #[test]
    fn simulator_tokenize_small() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let expected_prods = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let expected_start = vec![0, 9, 10, 23, 24, 29, 30, 36, 37, 48, 49];
        let expected_end = vec![9, 10, 23, 24, 29, 30, 36, 37, 48, 49, 53];
        let tokens = tokenize_string(&dfa, UTF8_INPUT)?;
        for (i, token) in tokens.into_iter().enumerate() {
            assert_eq!(token.production, expected_prods[i]);
            assert_eq!(token.start, expected_start[i]);
            assert_eq!(token.end, expected_end[i]);
        }
        Ok(())
    }

    #[test]
    fn simulator_tokenize_big() -> Result<(), ParseError> {
        // input bigger than BUFFER_SIZE to allow a single buffer swap
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let mut input = String::new();
        let mut prods = 1;
        while input.chars().count() < READ_SIZE as usize {
            input.push_str(UTF8_INPUT);
            prods += 10;
        }
        let tokens = tokenize_string(&dfa, &input)?;
        assert_eq!(tokens.len(), prods);
        Ok(())
    }

    #[test]
    fn simulator_single_lexeme_bigger_than_buffer() -> Result<(), ParseError> {
        let piece = UTF8_INPUT
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        let mut input = String::new();
        while input.chars().count() < 2 * READ_SIZE as usize {
            input.push_str(&piece);
        }
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let tokens = tokenize_string(&dfa, &input)?;
        assert_eq!(tokens.len(), 1);
        Ok(())
    }

    #[test]
    fn simulator_tokenize_match_longest() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("INT", "([0-9])+").into(),
                ("REAL", "([0-9])+'.'[0-9]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "123.456789";
        let tokens = tokenize_string(&dfa, input)?;
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 1);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 10);
        Ok(())
    }

    #[test]
    fn simulator_tokenize_match_longest_incomplete() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("INT", "([0-9])+").into(),
                ("REAL", "([0-9])+'.'[0-9]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "123.";
        let tokens = tokenize_string(&dfa, input)?;
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 0);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 3);
        Ok(())
    }

    #[test]
    fn simulator_tokenize_greedy_complete() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("COMMENT", "'/*'.*'*/'").into(),
                ("NUMBER", "[0-9]+").into(),
                ("SPACE", "[\r\n\t ]").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "/* test comment */ 123456 /* test comment 2 */";
        let tokens = tokenize_string(&dfa, input)?;
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 0);
        Ok(())
    }

    #[test]
    fn simulator_tokenize_greedy_incomplete() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("COMMENT", "'/*'.*'*/'").into(),
                ("NUMBER", "[0-9]+").into(),
                ("SPACE", "[\r\n\t ]").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "/* test comment */ 123456 ";
        let tokens = tokenize_string(&dfa, input)?;
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].production, 0);
        assert_eq!(tokens[1].production, 2);
        assert_eq!(tokens[2].production, 1);
        assert_eq!(tokens[3].production, 2);
        Ok(())
    }

    #[test]
    fn simulator_tokenize_nongreedy_kleene() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("COMMENT", "'/*'.*?'*/'").into(),
                ("SPACE", "[\r\n\t ]").into(),
                ("LITERAL", "'\"'~'\"'*'\"'").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "/* test comment */ \"/*this is not a comment*/\"";
        let tokens = tokenize_string(&dfa, input)?;
        assert_eq!(tokens.len(), 3, "The simulator greedily matched everything");
        Ok(())
    }

    #[test]
    fn simulator_backtrack_refresh() -> Result<(), ParseError> {
        // asserts that after backtracking the buffer 2 does not get refreshed again
        let grammar = Grammar::new(
            &[
                ("A", "'a'*").into(),
                ("AB", "'a'*'bbbbbb'").into(),
                ("BC", "'bbb'('c'*)").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = repeat('a')
            .take(READ_SIZE - 3)
            .chain(repeat('b').take(3))
            .chain(repeat('c').take(50))
            .collect::<String>();
        let tokens = tokenize_string(&dfa, &input)?;
        assert_eq!(tokens.len(), 2);
        Ok(())
    }

    #[test]
    fn tokenize_empty() -> Result<(), ParseError> {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "";
        let res = tokenize_string(&dfa, input)?;
        assert!(res.is_empty());
        Ok(())
    }

    #[test]
    fn tokenize_full() -> Result<(), ParseError> {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "aaabb";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 5);
        Ok(())
    }

    #[test]
    fn tokenize_err_no_partial() -> Result<(), ParseError> {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "d";
        let res = tokenize_string(&dfa, input)?;
        assert!(res.is_empty());
        Ok(())
    }

    #[test]
    fn tokenize_err_some_partial() -> Result<(), ParseError> {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "aad";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 2);
        Ok(())
    }

    #[test]
    fn tokenize_action_skip() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("ID", "[a-zA-Z]+").into(),
                ("INT", "[0-9]+").into(),
                ("WS", "[ \\t\\n\\t]+", btreeset! {Action::Skip}).into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "aad abc 123 bcd";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 4);
        assert_eq!(res[0].production, 0);
        assert_eq!(res[1].production, 0);
        assert_eq!(res[2].production, 1);
        assert_eq!(res[3].production, 0);
        Ok(())
    }

    #[test]
    fn tokenize_action_more() -> Result<(), ParseError> {
        let grammar = Grammar::new(
            &[
                ("PRIVATE", "'_'", btreeset! {Action::More}).into(),
                ("LETTERS", "[a-zA-Z]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "_test";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].production, 1);
        assert_eq!(res[0].start, 0);
        Ok(())
    }

    #[test]
    fn tokenize_action_mode() -> Result<(), ParseError> {
        let mut grammar = Grammar::new(
            &[("START_STRING", "'\"'", btreeset! {Action::Mode(1)}).into()],
            &[],
        );
        grammar.add_terminals(
            "STR".to_string(),
            &[
                ("TEXT", "~'\"'+").into(),
                ("END_STRING", "'\"'", btreeset! {Action::Mode(0)}).into(),
            ],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "\"string\"";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].mode, 0);
        assert_eq!(res[0].production, 0);
        assert_eq!(res[1].mode, 1);
        assert_eq!(res[1].production, 0);
        assert_eq!(res[2].mode, 1);
        assert_eq!(res[2].production, 1);
        Ok(())
    }

    #[test]
    fn tokenize_action_pushmode_popmode() -> Result<(), ParseError> {
        let mut grammar = Grammar::new(
            &[("OPEN_PAR", "'('", btreeset! {Action::PushMode(1)}).into()],
            &[],
        );
        grammar.add_terminals(
            "INSIDE".to_string(),
            &[
                ("OPEN_PAR", "'('", btreeset! {Action::PushMode(1)}).into(),
                ("CLOSE_PAR", "')'", btreeset! {Action::PopMode}).into(),
            ],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "((()))";
        let res = tokenize_string(&dfa, input)?;
        assert_eq!(res.len(), 6);
        assert_eq!(res[0].mode, 0);
        assert_eq!(res[1].mode, 1);
        Ok(())
    }

    #[test]
    fn tokenize_action_popmode_bottom_stack() {
        let mut grammar = Grammar::new(
            &[
                ("OPEN_PAR", "'('", btreeset! {Action::PushMode(1)}).into(),
                ("CLOSE_PAR", "')'", btreeset! {Action::PopMode}).into(),
            ],
            &[],
        );
        grammar.add_terminals(
            "INSIDE".to_string(),
            &[
                ("OPEN_PAR", "'('", btreeset! {Action::PushMode(1)}).into(),
                ("CLOSE_PAR", "')'", btreeset! {Action::PopMode}).into(),
            ],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "(()))))";
        let res = tokenize_string(&dfa, input);
        assert!(res.is_err());
    }
}
