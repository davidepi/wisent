use crate::lexer::Dfa;
use std::str::Chars;

/// Buffer size for the lexer. Each buffer stores the lookahead tokens (as u32) so we don't want to
/// store everything in memory but still keep a decently sized buffer.
const BUFFER_SIZE: usize = 1024;

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
pub struct DfaSimulator {
    /// Buffer and its len (capacity is BUFFER_SIZE, len is the second value in the tuple)
    buffers: [Vec<u32>; 2],
    /// Position of the next lexeme to read and the buffer where it is
    lexeme_pos: BufferIndexer,
    /// Position of the next lookahead character to read and the buffer where it is
    forward_pos: BufferIndexer,
    dfa: Dfa,
}

impl DfaSimulator {
    /// Creates a new Lexical Analyzer with the given DFA and the given input.
    /// # Examples
    /// Retrieving letters and numbers from a string:
    /// **TODO:** write an example when the public interface is available
    pub fn new(dfa: Dfa) -> Self {
        Self {
            buffers: [
                Vec::with_capacity(BUFFER_SIZE),
                Vec::with_capacity(BUFFER_SIZE),
            ],
            lexeme_pos: Default::default(),
            forward_pos: Default::default(),
            dfa,
        }
    }

    pub fn tokenize(&mut self, input: &mut Chars) -> Vec<u32> {
        self.init_tokenize(input);
        let mut state = self.dfa.start();
        let mut last_accepted = None;
        let mut productions = Vec::new();
        while let Some(char_id) = self.next_char(input) {
            if let Some(next) = self.dfa.moove(state, char_id) {
                //can advance
                state = next;
                if let Some(accepting_prod) = self.dfa.accepting(state) {
                    // if the new state is accepting, record the production and the current
                    // lexeme ending. DO NOT push the accepting state, as we try to greedily match
                    // other productions.
                    last_accepted = Some((accepting_prod, self.forward_pos));
                }
            } else if let Some((accepted_prod, last_valid_state)) = last_accepted {
                // no other move available, but there was a previous accepted production.
                // push the accepted production and roll back the head.
                productions.push(accepted_prod);
                state = self.dfa.start();
                self.lexeme_pos = last_valid_state;
                self.forward_pos = last_valid_state;
            } else {
                break; // halt
            }
        }
        if let Some((production, _)) = last_accepted {
            productions.push(production);
        }
        productions
    }

    /// initialize the buffers for a tokenization.
    /// next_char does not work for the initial refill
    fn init_tokenize(&mut self, input: &mut Chars) {
        self.buffers[0].clear();
        self.buffers[1].clear();
        self.lexeme_pos = Default::default();
        self.forward_pos = Default::default();
        let chars_it = input
            .take(self.buffers[0].capacity())
            .map(|c| self.dfa.symbol_table().symbol_id(c));
        self.buffers[0].extend(chars_it);
    }

    /// Reads the next character from the input in form of [`SymbolTable`] IDs.
    ///
    /// Takes care of buffering the read and handling the buffers.
    ///
    /// DO NOT change the input `s` until EOF is returned!
    fn next_char(&mut self, input: &mut Chars) -> Option<u32> {
        let mut current_buffer = &mut self.buffers[self.forward_pos.buffer_index as usize];
        if self.forward_pos.index == current_buffer.len() {
            let mut chars_it = input
                .take(BUFFER_SIZE)
                .map(|c| self.dfa.symbol_table().symbol_id(c))
                .peekable();
            // return if EOF
            chars_it.peek()?;
            // current buffer full: swap buffers and refill the other one
            let forward_next = if self.lexeme_pos.buffer_index == self.forward_pos.buffer_index {
                let next_buffer = (self.forward_pos.buffer_index + 1) % self.buffers.len() as u8;
                self.buffers[next_buffer as usize].clear();
                BufferIndexer {
                    index: 0,
                    buffer_index: next_buffer,
                }
            } else {
                // can't swap buffers as the current token is larger than an entire buffer.
                // extend the current buffer (this is not good for speed, as the
                // buffer should be doubled, but this should not happen often)
                self.forward_pos
            };
            self.forward_pos = forward_next;
            let next_buffer = &mut self.buffers[self.forward_pos.buffer_index as usize];
            next_buffer.extend(chars_it);
            current_buffer = next_buffer;
        }
        // current buffer not full (or just refilled), just fetch the next character
        let char = current_buffer[self.forward_pos.index as usize];
        self.forward_pos.index += 1;
        Some(char)
    }
}

#[cfg(test)]
mod tests {
    use super::DfaSimulator;
    use crate::grammar::Grammar;
    use crate::lexer::simulator::BUFFER_SIZE;
    use crate::lexer::Dfa;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn simulator_input_small() {
        // input smaller than BUFFER_SIZE
        let grammar = Grammar::new(&["(~[0-9])+", "' '*"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(dfa);
        simulator.init_tokenize(&mut reader);
        for _ in 0..UTF8_INPUT.chars().count() {
            assert!(simulator.next_char(&mut reader).is_some());
        }
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_multiple_eof() {
        // after next_char returns eof once, additional calls return always eof
        let grammar = Grammar::new(&["(~[0-9])+", "' '*"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(dfa);
        simulator.init_tokenize(&mut reader);
        while simulator.next_char(&mut reader).is_some() {}
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_input_big() {
        // input bigger than BUFFER_SIZE to allow a single buffer swap
        // bigger inputs requires moving also the lexeme_pos
        let grammar = Grammar::new(&["(~[0-9])+", "' '*"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut input = String::new();
        while input.chars().count() < BUFFER_SIZE as usize {
            input.push_str(UTF8_INPUT);
        }
        let mut reader = input.chars();
        let mut simulator = DfaSimulator::new(dfa);
        simulator.init_tokenize(&mut reader);
        for _ in 0..input.chars().count() {
            assert!(simulator.next_char(&mut reader).is_some());
        }
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_lexeme_bigger_than_two_buffers() {
        let piece = UTF8_INPUT
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        let mut input = String::new();
        while input.chars().count() < 2 * BUFFER_SIZE as usize {
            input.push_str(&piece);
        }
        let mut reader = input.chars();
        let grammar = Grammar::new(&["(~' ')+"], &[], &["NOT_SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut simulator = DfaSimulator::new(dfa);
        simulator.init_tokenize(&mut reader);
        for _ in 0..input.chars().count() {
            assert!(simulator.next_char(&mut reader).is_some());
        }
        assert!(simulator.next_char(&mut reader).is_none());
    }

    #[test]
    fn simulator_tokenize_non_greedy() {
        let grammar = Grammar::new(&["(~[0-9 ])+", "' '+"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let expected = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let mut simulator = DfaSimulator::new(dfa);
        let prods = simulator.tokenize(&mut reader);
        assert_eq!(prods, expected);
    }
}
