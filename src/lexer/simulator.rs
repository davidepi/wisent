use super::SymbolTable;
use crate::lexer::Dfa;
use std::str::Chars;

/// Buffer size for the lexer. Each buffer stores the lookahead tokens (as u32) so we don't want to
/// store everything in memory but still keep a decently sized buffer.
const BUFFER_SIZE: usize = 1024;

#[derive(Debug, Copy, Clone)]
/// Token retrieved by the lexical analyzer.
pub struct Token {
    /// The production ID (in the grammar used to build the DFA) associated with this token.
    pub production: u32,
    /// Beginning of the token in number of bytes since the beginning of the stream.
    pub start: usize,
    /// End of the token in number of bytes since the beginning of the stream.
    /// If the last character of the token is a multi-byte character this value correspond to the
    /// last byte of the character.
    pub end: usize,
}

#[derive(Default, Copy, Clone)]
/// Simple to struct to name indices for the lexeme_pos and forward_pos
/// Just to remember what they represents, with a tuple it would be a mess
struct BufferIndexer {
    index: usize,
    buffer_index: u8,
}

/// Lexical analyzer for a DFA
///
/// Simulates a Dfa with a given input and groups the input characters in tokens according to the
/// rules of the grammar passed to the [`Dfa`].
pub struct DfaSimulator<'a> {
    /// Buffer and its len (capacity is BUFFER_SIZE, len is the second value in the tuple)
    buffers: [Vec<u32>; 2],
    /// true if the forward_pos backtracked to a different buffer
    backtracked: bool,
    /// Position of the next lexeme to read and the buffer where it is
    lexeme_pos: BufferIndexer,
    /// Position of the next lookahead character to read and the buffer where it is
    forward_pos: BufferIndexer,
    /// DFA containing the moves and alphabet
    dfa: &'a Dfa,
}

impl<'a> DfaSimulator<'a> {
    /// Creates a new Lexical Analyzer with the given DFA and the given input.
    pub fn new(dfa: &'a Dfa) -> DfaSimulator {
        Self {
            buffers: [
                Vec::with_capacity(BUFFER_SIZE),
                Vec::with_capacity(BUFFER_SIZE),
            ],
            backtracked: false,
            lexeme_pos: Default::default(),
            forward_pos: Default::default(),
            dfa,
        }
    }

    /// Runs the lexical analysis and retrieves the tokens composing a string.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::{Dfa, DfaSimulator};
    /// let grammar = Grammar::new(&["([0-9])+", "([a-z])+"], &[], &["NUMBER", "WORD"]);
    /// let dfa = Dfa::new(&grammar);
    /// let input = "abc123";
    /// let simulator = DfaSimulator::new(&dfa);
    /// let tokens = simulator.tokenize(input.chars());
    /// assert_eq!(tokens[0].production, 1);
    /// assert_eq!(tokens[1].production, 0);
    /// ```
    pub fn tokenize(mut self, mut input: Chars) -> Vec<Token> {
        self.init_tokenize(&mut input);
        let mut state = self.dfa.start();
        let mut location_start = 0;
        let mut location_end = 0;
        let mut last_accepted = None;
        let mut productions = Vec::new();
        loop {
            if let Some((char_id, bytes)) = self.next_char(&mut input) {
                location_end += bytes as usize;
                if let Some(next) = self.dfa.moove(state, char_id) {
                    //can advance
                    state = next;
                    if let Some(accepting_prod) = self.dfa.accepting(state) {
                        // if the new state is accepting, record the production and the current
                        // lexeme ending. DO NOT push the accepting state, as we try to greedily match
                        // other productions.
                        last_accepted = Some((accepting_prod, self.forward_pos, location_end));
                        if !self.dfa.non_greedy(state) {
                            // if the current accepting state is greedy, continue looping
                            continue;
                        }
                    } else {
                        continue; // avoid entering in the next if
                    }
                }
            }
            // this if serves to push the accepted state
            if let Some((production, last_valid_state, last_end)) = last_accepted {
                // no other move available, but there was a previous accepted production.
                // push the accepted production and roll back the head.
                productions.push(Token {
                    production,
                    start: location_start,
                    end: last_end,
                });
                state = self.dfa.start();
                if self.forward_pos.buffer_index != last_valid_state.buffer_index {
                    // the forward pos already loaded the next buffer, but is going back to
                    // previous one. The flag is used to avoid reloading another buffer again.
                    self.backtracked = true;
                }
                //FIXME: in case a lexeme ends exactly on the last cell of a buffer,
                //last_valid_state will point to the next cell (that does not exist).
                //next_char will then trigger a buffer extension instead of swap.
                self.lexeme_pos = last_valid_state;
                self.forward_pos = last_valid_state;
                location_start = last_end;
                location_end = last_end;
                last_accepted = None;
            } else {
                break; // no moves and no accepting state reached. halt.
            }
        }
        productions
    }

    /// initialize the buffers for a tokenization.
    /// next_char does not work for the initial refill
    fn init_tokenize(&mut self, input: &mut Chars) {
        self.buffers[0].clear();
        self.buffers[1].clear();
        self.backtracked = false;
        self.lexeme_pos = Default::default();
        self.forward_pos = Default::default();
        let chars_it = input
            .take(self.buffers[0].capacity())
            .map(|c| encode_char_len(c, self.dfa.symbol_table()));
        self.buffers[0].extend(chars_it);
    }

    /// Reads the next character from the input in form of [`SymbolTable`] IDs.
    ///
    /// Takes care of buffering the read and handling the buffers.
    ///
    /// DO NOT change the input `s` until EOF is returned!
    ///
    /// Returns the ID and the number of bytes used to represent it.
    fn next_char(&mut self, input: &mut Chars) -> Option<(u32, u8)> {
        let mut current_buffer = &mut self.buffers[self.forward_pos.buffer_index as usize];
        if self.forward_pos.index == current_buffer.len() {
            let mut next_buffer = (self.forward_pos.buffer_index + 1) % self.buffers.len() as u8;
            let mut forward_next = BufferIndexer {
                index: 0,
                buffer_index: next_buffer,
            };
            // backtracked is true if forward pos returned back to the previous buffer after
            // accepting a state. In that case the buffer does NOT need to be refilled, but the
            // buffer index in forward_pos needs to be changed.
            if !self.backtracked {
                let mut chars_it = input
                    .take(BUFFER_SIZE)
                    .map(|c| encode_char_len(c, self.dfa.symbol_table()))
                    .peekable();
                // return None if EOF
                chars_it.peek()?;
                if next_buffer != self.lexeme_pos.buffer_index {
                    //  swap buffers and refill the other one
                    self.buffers[next_buffer as usize].clear();
                } else {
                    // can't swap buffers as the current token is larger than an entire buffer.
                    // extend the current buffer and continue with current forward_pos.
                    forward_next = self.forward_pos;
                    next_buffer = self.forward_pos.buffer_index;
                };
                self.buffers[next_buffer as usize].extend(chars_it);
            }
            self.backtracked = false;
            self.forward_pos = forward_next;
            current_buffer = &mut self.buffers[next_buffer as usize];
        }
        // current buffer not full (or just refilled), just fetch the next character
        let char = current_buffer[self.forward_pos.index as usize];
        self.forward_pos.index += 1;
        Some(decode_char_len(char))
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

/// Error representing a partially complete tokenization. If this error is produced, the lexer
/// didn't reach EOF, but some tokens may have been recognized.
#[derive(Debug, Clone)]
pub struct IncompleteParse {
    /// The list of partially recognized tokens.
    pub partial: Vec<Token>,
}

/// Tokenize a string with a given DFA.
///
/// This utility function is a wrapper that creates a [DfaSimulator], calls its
/// [tokenize](DfaSimulator::tokenize) method and checks whether the simulator reached EOF or not.
///
/// If EOF was not reached, it is still possible to find the matched tokens inside the Error.
/// # Examples
/// Tokenize string:
/// ```
/// # use wisent::grammar::Grammar;
/// # use wisent::lexer::{Dfa, DfaSimulator, tokenize};
/// let grammar = Grammar::new(&["([0-9])+", "([a-z])+"], &[], &["NUMBER", "WORD"]);
/// let dfa = Dfa::new(&grammar);
/// let input = "abc123";
/// let result = tokenize(&dfa, &input);
/// assert!(result.is_ok());
/// ```
pub fn tokenize(dfa: &Dfa, input: &str) -> Result<Vec<Token>, IncompleteParse> {
    if !input.is_empty() {
        let tokens = DfaSimulator::new(dfa).tokenize(input.chars());
        if let Some(last) = tokens.last() {
            if last.end == input.len() {
                Ok(tokens)
            } else {
                Err(IncompleteParse { partial: tokens })
            }
        } else {
            Err(IncompleteParse { partial: tokens })
        }
    } else {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;
    use crate::lexer::simulator::BUFFER_SIZE;
    use crate::lexer::{Dfa, DfaSimulator};
    use std::iter::repeat;

    use super::tokenize;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn simulator_next_char() {
        // input smaller than BUFFER_SIZE, extra logic for the buffer swap is in the tokenize
        // function
        let grammar = Grammar::new(&["(~[ ])+", "' '+"], &[], &["NOT_SPACE", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(&dfa);
        simulator.init_tokenize(&mut reader);
        for _ in 0..UTF8_INPUT.chars().count() {
            assert!(simulator.next_char(&mut reader).is_some());
        }
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_multiple_eof() {
        // after next_char returns eof once, additional calls return always eof
        let grammar = Grammar::new(&["(~[ ])+", "' '+"], &[], &["NOT_SPACE", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(&dfa);
        simulator.init_tokenize(&mut reader);
        while simulator.next_char(&mut reader).is_some() {}
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_tokenize_small() {
        let grammar = Grammar::new(&["(~[ ])+", "' '+"], &[], &["NOT_SPACE", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let expected_prods = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let expected_start = vec![0, 9, 10, 23, 24, 29, 30, 36, 37, 48, 49];
        let expected_end = vec![9, 10, 23, 24, 29, 30, 36, 37, 48, 49, 53];
        let simulator = DfaSimulator::new(&dfa);
        let tokens = simulator.tokenize(UTF8_INPUT.chars());
        for (i, token) in tokens.into_iter().enumerate() {
            assert_eq!(token.production, expected_prods[i]);
            assert_eq!(token.start, expected_start[i]);
            assert_eq!(token.end, expected_end[i]);
        }
    }

    #[test]
    fn simulator_tokenize_big() {
        // input bigger than BUFFER_SIZE to allow a single buffer swap
        let grammar = Grammar::new(&["(~[ ])+", "' '+"], &[], &["NOT_SPACE", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut input = String::new();
        let mut prods = 1;
        while input.chars().count() < BUFFER_SIZE as usize {
            input.push_str(UTF8_INPUT);
            prods += 10;
        }
        let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
        assert_eq!(tokens.len(), prods);
    }

    #[test]
    fn simulator_single_lexeme_bigger_than_buffer() {
        let piece = UTF8_INPUT
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        let mut input = String::new();
        while input.chars().count() < 2 * BUFFER_SIZE as usize {
            input.push_str(&piece);
        }
        let grammar = Grammar::new(&["(~' ')+"], &[], &["NOT_SPACE"]);
        let dfa = Dfa::new(&grammar);
        let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn simulator_tokenize_match_longest() {
        let grammar = Grammar::new(&["([0-9])+", "([0-9])+'.'[0-9]+"], &[], &["INT", "REAL"]);
        let dfa = Dfa::new(&grammar);
        let input = "123.456789";
        let simulator = DfaSimulator::new(&dfa);
        let tokens = simulator.tokenize(input.chars()); //.into_iter().map(|t|t.production).collect::<Vec<_>>();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 1);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 10);
    }

    #[test]
    fn simulator_tokenize_match_longest_incomplete() {
        let grammar = Grammar::new(&["([0-9])+", "([0-9])+'.'[0-9]+"], &[], &["INT", "REAL"]);
        let dfa = Dfa::new(&grammar);
        let input = "123.";
        let simulator = DfaSimulator::new(&dfa);
        let tokens = simulator.tokenize(input.chars()); //.into_iter().map(|t|t.production).collect::<Vec<_>>();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 0);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 3);
    }

    #[test]
    fn simulator_tokenize_greedy_complete() {
        let grammar = Grammar::new(
            &["'/*'.*'*/'", "[0-9]+", "[\r\n\t ]"],
            &[],
            &["COMMENT", "NUMBER", "SPACE"],
        );
        let dfa = Dfa::new(&grammar);
        let simulator = DfaSimulator::new(&dfa);
        let input = "/* test comment */ 123456 /* test comment 2 */";
        let tokens = simulator.tokenize(input.chars());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 0);
    }

    #[test]
    fn simulator_tokenize_greedy_incomplete() {
        let grammar = Grammar::new(
            &["'/*'.*'*/'", "[0-9]+", "[\n\t ]"],
            &[],
            &["COMMENT", "NUMBER", "SPACE"],
        );
        let dfa = Dfa::new(&grammar);
        let simulator = DfaSimulator::new(&dfa);
        let input = "/* test comment */ 123456 ";
        let tokens = simulator.tokenize(input.chars());
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].production, 0);
        assert_eq!(tokens[1].production, 2);
        assert_eq!(tokens[2].production, 1);
        assert_eq!(tokens[3].production, 2);
    }

    #[test]
    fn simulator_tokenize_nongreedy_kleene() {
        let grammar = Grammar::new(
            &["'/*'.*?'*/'", "[\r\n\t ]", "'\"'~'\"'*'\"'"],
            &[],
            &["COMMENT", "SPACE", "LITERAL"],
        );
        let dfa = Dfa::new(&grammar);
        let simulator = DfaSimulator::new(&dfa);
        let input = "/* test comment */ \"/*this is not a comment*/\"";
        let tokens = simulator.tokenize(input.chars());
        assert_eq!(tokens.len(), 3, "The simulator greedily matched everything");
    }

    #[test]
    fn simulator_backtrack_refresh() {
        // asserts that after backtracking the buffer 2 does not get refreshed again
        let grammar = Grammar::new(
            &["'a'*", "'a'*'bbbbbb'", "'bbb'('c'*)"],
            &[],
            &["A", "AB", "BC"],
        );
        let dfa = Dfa::new(&grammar);
        let input = repeat('a')
            .take(BUFFER_SIZE - 3)
            .chain(repeat('b').take(3))
            .chain(repeat('c').take(50))
            .collect::<String>();
        let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn tokenize_empty() {
        let grammar = Grammar::new(&["'a'", "'b'"], &[], &["A", "B"]);
        let dfa = Dfa::new(&grammar);
        let input = "";
        let res = tokenize(&dfa, input);
        assert!(res.is_ok());
    }

    #[test]
    fn tokenize_full() {
        let grammar = Grammar::new(&["'a'", "'b'"], &[], &["A", "B"]);
        let dfa = Dfa::new(&grammar);
        let input = "aaabb";
        let res = tokenize(&dfa, input).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn tokenize_err_no_partial() {
        let grammar = Grammar::new(&["'a'", "'b'"], &[], &["A", "B"]);
        let dfa = Dfa::new(&grammar);
        let input = "d";
        let res = tokenize(&dfa, input);
        assert!(res.is_err());
        assert!(res.err().unwrap().partial.is_empty());
    }

    #[test]
    fn tokenize_err_some_partial() {
        let grammar = Grammar::new(&["'a'", "'b'"], &[], &["A", "B"]);
        let dfa = Dfa::new(&grammar);
        let input = "aad";
        let res = tokenize(&dfa, input);
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().partial.len(), 2);
    }
}
