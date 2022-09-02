use crate::error::ParseError;
use crate::lexer::{Dfa, SymbolTable};
use std::io::Error as IOError;
use std::str::Chars;

const BUFFER_SIZE: u16 = 4096;

struct CharBuffer {
    data: [u32; BUFFER_SIZE as usize],
    len: u16,
}

impl Default for CharBuffer {
    fn default() -> Self {
        Self {
            data: [0; BUFFER_SIZE as usize],
            len: 0,
        }
    }
}

#[derive(Default, Copy, Clone)]
/// Simple to struct to name indices for the lexeme_pos and forward_pos
/// Just to remember what they represents, with a tuple it would be a mess
struct BufferIndexer {
    index: u16,
    buffer_index: u8,
}

/// Lexical analyzer for a DFA
///
/// Simulates a Dfa with a given input and groups the input characters in tokens according to the
/// rules of the grammar passed to the [`Dfa`].
pub struct DfaSimulator {
    /// Buffer and its len (capacity is BUFFER_SIZE, len is the second value in the tuple)
    buffers: [CharBuffer; 2],
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
            buffers: [Default::default(), Default::default()],
            lexeme_pos: Default::default(),
            forward_pos: Default::default(),
            dfa,
        }
    }

    pub fn tokenize<R: Utf8CharReader>(&mut self, input: &mut R) -> Result<Vec<u32>, ParseError> {
        self.init_tokenize(input)?;
        let mut state = self.dfa.start();
        let mut last_accepted = None;
        let mut productions = Vec::new();
        while let Some(char_id) = self.next_char(input)? {
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
        Ok(productions)
    }

    /// initialize the buffers for a tokenization.
    fn init_tokenize<R: Utf8CharReader>(&mut self, input: &mut R) -> Result<(), ParseError> {
        let symtable = self.dfa.symbol_table();
        refill_buffer(&mut self.buffers[0], input, symtable)?;
        self.lexeme_pos = Default::default();
        self.forward_pos = Default::default();
        Ok(())
    }

    /// Reads the next character from the input in form of [`SymbolTable`] IDs.
    ///
    /// Takes care of buffering the read and handling the buffers.
    ///
    /// DO NOT change the input `s` until EOF is returned!
    fn next_char<R: Utf8CharReader>(&mut self, input: &mut R) -> Result<Option<u32>, ParseError> {
        let mut current_buffer = &self.buffers[self.forward_pos.buffer_index as usize];
        if self.forward_pos.index == current_buffer.len {
            // refill buffer
            // swap buffers and refill the other one
            let next_index = (self.forward_pos.buffer_index + 1) % self.buffers.len() as u8;
            let count_bytes = refill_buffer(
                &mut self.buffers[next_index as usize],
                input,
                self.dfa.symbol_table(),
            )?;
            if count_bytes == 0 {
                // EOF
                return Ok(None);
            } else {
                if self.lexeme_pos.buffer_index != self.forward_pos.buffer_index {
                    // lexeme is longer than an entire buffer, and buffer was overwritten
                    // can't continue
                    return Err(ParseError::LexerSimulationError {
                        message: "lexeme size exceeded the maximum supported size".to_string(),
                    });
                }
                self.forward_pos.buffer_index = next_index;
                self.forward_pos.index = 0;
                current_buffer = &self.buffers[self.forward_pos.buffer_index as usize];
            }
        }
        // if arrives here, then forward_pos < buffer len
        let char = current_buffer.data[self.forward_pos.index as usize];
        self.forward_pos.index += 1;
        Ok(Some(char))
    }
}

/// Refill the simulator buffer by taking symbols from `r` and converting them to their ID in the
/// symbol table.
/// Returns the amount of bytes read. The amount of chars read, instead, will be written in the
/// `len` parameter of the `CharBuffer`
fn refill_buffer<R: Utf8CharReader>(
    buf: &mut CharBuffer,
    input: &mut R,
    symtab: &SymbolTable,
) -> Result<usize, IOError> {
    let string = input.read_chars(BUFFER_SIZE as usize)?;
    buf.len = 0;
    for (symbol, buf_val) in string.chars().zip(buf.data.iter_mut()) {
        *buf_val = symtab.symbol_id(symbol);
        buf.len += 1;
    }
    Ok(string.len())
}

/// Trait used to read a given amount of UTF-8 characters from different sources.
///
/// This trait is similar to [`Chars`] but handle the conversion from `u8` to `char` in most cases.
pub trait Utf8CharReader {
    /// Reads the given amount of UTF-8 character from the implementor.
    ///
    /// If not enough characters are available, this method returns all the available ones.
    /// # Examples
    /// Reading from [`str`] or [`String`]:
    /// ```
    /// use wisent::lexer::Utf8CharReader;
    ///
    /// let string = "🦀: «Hello!»";
    /// let mut chars = string.chars();
    /// assert_eq!(chars.read_chars(1).unwrap(), "🦀");
    /// assert!(chars.read_chars(2).is_ok());
    /// assert_eq!(chars.read_chars(8).unwrap(), "«Hello!»");
    /// ```
    fn read_chars(&mut self, chars_no: usize) -> Result<String, IOError>;
}

impl Utf8CharReader for Chars<'_> {
    fn read_chars(&mut self, chars_no: usize) -> Result<String, IOError> {
        Ok(self.take(chars_no).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ParseError;
    use crate::grammar::Grammar;
    use crate::lexer::simulator::BUFFER_SIZE;
    use crate::lexer::{Dfa, Utf8CharReader};
    use std::io::Error as IOError;

    use super::DfaSimulator;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn utf8charreader_read_too_much_characters() -> Result<(), IOError> {
        let mut input = UTF8_INPUT.chars();
        assert_eq!(input.read_chars(6)?, "Příliš");
        assert_eq!(input.read_chars(999)?, " žluťoučký kůň úpěl ďábelské ódy");
        Ok(())
    }

    #[test]
    fn utf8charrreader_continue_reading_characters() -> Result<(), IOError> {
        let mut input = UTF8_INPUT.chars();
        assert!(input.read_chars(999).is_ok());
        assert_eq!(input.read_chars(999)?, "".to_string());
        Ok(())
    }

    #[test]
    fn simulator_input_small() -> Result<(), ParseError> {
        // input smaller than BUFFER_SIZE
        let grammar = Grammar::new(&["(~[0-9])+", "' '*"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let mut simulator = DfaSimulator::new(dfa);
        for _ in 0..38 {
            assert!(simulator.next_char(&mut reader)?.is_some());
        }
        assert!(simulator.next_char(&mut reader)?.is_none());
        Ok(())
    }

    #[test]
    fn simulator_input_big() -> Result<(), ParseError> {
        // input bigger than BUFFER_SIZE to allow a buffer swap
        // bigger inputs requires moving also the lexeme_pos
        let grammar = Grammar::new(&["(~[0-9])+", "' '*"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut input = String::new();
        while input.chars().count() < BUFFER_SIZE as usize {
            input.push_str(UTF8_INPUT);
        }
        let mut simulator = DfaSimulator::new(dfa);
        let mut reader = input.chars();
        for _ in 0..4104 {
            assert!(simulator.next_char(&mut reader)?.is_some());
        }
        assert!(simulator.next_char(&mut reader)?.is_none());
        Ok(())
    }

    #[test]
    fn simulator_input_lexeme_bigger_than_buffer() -> Result<(), ParseError> {
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
        for _ in 0..8192 {
            assert!(simulator.next_char(&mut reader)?.is_some());
        }
        let err = simulator.next_char(&mut reader);
        match err {
            Ok(_) => panic!("test should return error"),
            Err(e) => match e {
                ParseError::LexerSimulationError { .. } => Ok(()),
                _ => panic!("wrong error type"),
            },
        }
    }

    #[test]
    fn simulator_tokenize_non_greedy() -> Result<(), ParseError> {
        let grammar = Grammar::new(&["(~[0-9 ])+", "' '+"], &[], &["NO_NUMBER", "SPACE"]);
        let dfa = Dfa::new(&grammar);
        let mut reader = UTF8_INPUT.chars();
        let expected = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let mut simulator = DfaSimulator::new(dfa);
        let prods = simulator.tokenize(&mut reader)?;
        assert_eq!(prods, expected);
        Ok(())
    }
}
