use super::SymbolTable;
use crate::grammar::Action;
use crate::lexer::MultiDfa;
use std::collections::VecDeque;
use std::str::Chars;

/// How many characters are read when populating the buffer of the simulator.
/// In any case the simulator uses buffered reads.
const READ_SIZE: usize = 128;

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
pub struct DfaSimulator<'a> {
    /// Buffer storing the read characters converted using their symbol table id.
    buffer: VecDeque<u32>,
    /// Position of the next lookahead character to read (index of buffer)
    forward_pos: usize,
    /// DFA containing the moves and alphabet
    mdfa: &'a MultiDfa,
    /// Current mode being simulated. Represented as stack to allow PUSHMODE and POPMODE
    current_mode: Vec<u32>,
}

impl<'a> DfaSimulator<'a> {
    /// Creates a new Lexical Analyzer with the given DFA and the given input.
    pub fn new(dfas: &'a MultiDfa) -> DfaSimulator {
        Self {
            buffer: VecDeque::with_capacity(READ_SIZE),
            forward_pos: Default::default(),
            mdfa: dfas,
            current_mode: vec![0],
        }
    }

    /// Runs the lexical analysis and retrieves the tokens composing a string.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::{MultiDfa, DfaSimulator};
    /// let grammar = Grammar::new(
    ///     &[("NUMBER", "([0-9])+").into(), ("WORD", "([a-z])+").into()],
    ///     &[],
    /// );
    /// let dfa = MultiDfa::new(&grammar);
    /// let input = "abc123";
    /// let simulator = DfaSimulator::new(&dfa);
    /// let tokens = simulator.tokenize(input.chars());
    /// assert_eq!(tokens[0].production, 1);
    /// assert_eq!(tokens[1].production, 0);
    /// ```
    pub fn tokenize(mut self, mut input: Chars) -> Vec<Token> {
        let mut dfa = &self.mdfa[self.current_mode[0] as usize];
        let mut state = dfa.start();
        let mut absolute_start_index = 0;
        let mut absolute_end_index = 0;
        let mut last_accepted = None;
        let mut productions = Vec::new();
        loop {
            if let Some((char_id, bytes_for_this_char)) = self.next_char(&mut input) {
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
                        if !dfa.non_greedy(state) {
                            // if the current accepting state is greedy, continue looping
                            continue;
                        }
                    } else {
                        continue; // avoid entering in the next if
                    }
                }
            }
            // this if serves to push the accepted state
            if let Some((production, actions, lexeme_chars, last_valid_end)) = last_accepted {
                // backtrack the lookaheads to the last valid position
                absolute_end_index = last_valid_end;
                // no other move available, but there was a previous accepted production.
                // push the accepted production if the action is not Skip.
                if !actions.contains(&Action::Skip) && !actions.contains(&Action::More) {
                    productions.push(Token {
                        production,
                        start: absolute_start_index,
                        end: absolute_end_index,
                        mode: *self.current_mode.last().unwrap(),
                    });
                }
                if !actions.contains(&Action::More) {
                    for _ in 0..lexeme_chars {
                        self.buffer.pop_front();
                    }
                    absolute_start_index = absolute_end_index;
                    self.forward_pos = 0;
                } else {
                    // 'MORE': forward_pos should not be reset, but at least the lookahead removed
                    self.forward_pos = lexeme_chars;
                }
                last_accepted = None;
                // handle mode switching
                for action in actions {
                    match action {
                        Action::Mode(m) => {
                            let cur_mode = self.current_mode.last_mut().unwrap();
                            *cur_mode = *m;
                            dfa = &self.mdfa[*cur_mode as usize];
                        }
                        Action::PushMode(_) => todo!(),
                        Action::PopMode => todo!(),
                        _ => (),
                    }
                }
                state = dfa.start();
            } else {
                break; // no moves and no accepting state reached. halt.
            }
        }
        productions
    }

    /// Reads the next character from the input in form of [`SymbolTable`] IDs.
    ///
    /// Takes care of buffering the read and handling the buffers.
    ///
    /// DO NOT change the `input` until EOF is returned!
    ///
    /// Returns the ID and the number of bytes used to represent it.
    fn next_char(&mut self, input: &mut Chars) -> Option<(u32, u8)> {
        // checks if the buf needs refilling
        if self.forward_pos == self.buffer.len() {
            let chars_it = input
                .take(READ_SIZE)
                .map(|c| encode_char_len(c, self.mdfa.symbol_table()));
            self.buffer.extend(chars_it);
        }
        if let Some(&char) = self.buffer.get(self.forward_pos) {
            // refilling successfull, return the next character
            self.forward_pos += 1;
            Some(decode_char_len(char))
        } else {
            // refilling failed, EOF reached
            None
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
/// # use wisent::lexer::{MultiDfa, DfaSimulator, tokenize};
/// let grammar = Grammar::new(
///     &[("NUMBER", "([0-9])+").into(), ("WORD", "([a-z])+").into()],
///     &[],
/// );
/// let dfa = MultiDfa::new(&grammar);
/// let input = "abc123";
/// let result = tokenize(&dfa, &input);
/// assert!(result.is_ok());
/// ```
pub fn tokenize(dfa: &MultiDfa, input: &str) -> Result<Vec<Token>, IncompleteParse> {
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
    use crate::grammar::{Action, Grammar};
    use crate::lexer::simulator::READ_SIZE;
    use crate::lexer::{DfaSimulator, MultiDfa};
    use maplit::btreeset;
    use std::iter::repeat;

    use super::tokenize;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn simulator_next_char() {
        // input smaller than BUFFER_SIZE, extra logic for the buffer swap is in the tokenize
        // function
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(&dfa);
        for _ in 0..UTF8_INPUT.chars().count() {
            assert!(simulator.next_char(&mut reader).is_some());
        }
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_multiple_eof() {
        // after next_char returns eof once, additional calls return always eof
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(&dfa);
        while simulator.next_char(&mut reader).is_some() {}
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_tokenize_small() {
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
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
        while input.chars().count() < 2 * READ_SIZE as usize {
            input.push_str(&piece);
        }
        let grammar = Grammar::new(
            &[("NOT_SPACE", "(~[ ])+").into(), ("SPACE", "' '+").into()],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn simulator_tokenize_match_longest() {
        let grammar = Grammar::new(
            &[
                ("INT", "([0-9])+").into(),
                ("REAL", "([0-9])+'.'[0-9]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
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
        let grammar = Grammar::new(
            &[
                ("INT", "([0-9])+").into(),
                ("REAL", "([0-9])+'.'[0-9]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
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
            &[
                ("COMMENT", "'/*'.*'*/'").into(),
                ("NUMBER", "[0-9]+").into(),
                ("SPACE", "[\r\n\t ]").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let simulator = DfaSimulator::new(&dfa);
        let input = "/* test comment */ 123456 /* test comment 2 */";
        let tokens = simulator.tokenize(input.chars());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].production, 0);
    }

    #[test]
    fn simulator_tokenize_greedy_incomplete() {
        let grammar = Grammar::new(
            &[
                ("COMMENT", "'/*'.*'*/'").into(),
                ("NUMBER", "[0-9]+").into(),
                ("SPACE", "[\r\n\t ]").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
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
            &[
                ("COMMENT", "'/*'.*?'*/'").into(),
                ("SPACE", "[\r\n\t ]").into(),
                ("LITERAL", "'\"'~'\"'*'\"'").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let simulator = DfaSimulator::new(&dfa);
        let input = "/* test comment */ \"/*this is not a comment*/\"";
        let tokens = simulator.tokenize(input.chars());
        assert_eq!(tokens.len(), 3, "The simulator greedily matched everything");
    }

    #[test]
    fn simulator_backtrack_refresh() {
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
        let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn tokenize_empty() {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "";
        let res = tokenize(&dfa, input);
        assert!(res.is_ok());
    }

    #[test]
    fn tokenize_full() {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "aaabb";
        let res = tokenize(&dfa, input).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn tokenize_err_no_partial() {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "d";
        let res = tokenize(&dfa, input);
        assert!(res.is_err());
        assert!(res.err().unwrap().partial.is_empty());
    }

    #[test]
    fn tokenize_err_some_partial() {
        let grammar = Grammar::new(&[("A", "'a'").into(), ("B", "'b'").into()], &[]);
        let dfa = MultiDfa::new(&grammar);
        let input = "aad";
        let res = tokenize(&dfa, input);
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().partial.len(), 2);
    }

    #[test]
    fn tokenize_action_skip() {
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
        let res = tokenize(&dfa, input).expect("Tokenization failed");
        assert_eq!(res.len(), 4);
        assert_eq!(res[0].production, 0);
        assert_eq!(res[1].production, 0);
        assert_eq!(res[2].production, 1);
        assert_eq!(res[3].production, 0);
    }

    #[test]
    fn tokenize_action_more() {
        let grammar = Grammar::new(
            &[
                ("PRIVATE", "'_'", btreeset! {Action::More}).into(),
                ("LETTERS", "[a-zA-Z]+").into(),
            ],
            &[],
        );
        let dfa = MultiDfa::new(&grammar);
        let input = "_test";
        let res = tokenize(&dfa, input).expect("Tokenization failed");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].production, 1);
        assert_eq!(res[0].start, 0);
    }

    #[test]
    fn tokenize_action_mode() {
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
        let res = tokenize(&dfa, input).expect("Tokenization failed");
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].mode, 0);
        assert_eq!(res[0].production, 0);
        assert_eq!(res[1].mode, 1);
        assert_eq!(res[1].production, 0);
        assert_eq!(res[2].mode, 1);
        assert_eq!(res[2].production, 1);
    }
}
