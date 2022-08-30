/// Error wrapping all the possible kind of errors encountered during parsing.
///
/// The possible errors can be:
/// * `IOError` - containing an `std::io::Error`, this kind of error can arise
/// when opening the grammar on disks fail.
/// * `SyntaxError` - An error containing a `message` String that can arise when
/// parsing fails due to syntax errors in the input grammar.
/// * `DeserializeError` - An error arising during deserialization.
#[derive(Debug)]
pub enum ParseError {
    IOError(std::io::Error),
    SyntaxError { message: String },
    DeserializeError { message: String },
    LexerSimulationError { message: String },
}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::IOError(e)
    }
}

impl std::error::Error for ParseError {}

impl std::fmt::Display for ParseError {
    fn fmt(&self, buffer: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ParseError::IOError(e) => write!(buffer, "IOError: {}", e),
            ParseError::SyntaxError { message } => write!(buffer, "SyntaxError: {}", message),
            ParseError::DeserializeError { message } => {
                write!(buffer, "DeserializeError: {}", message)
            }
            ParseError::LexerSimulationError { message } => {
                write!(buffer, "LexerSimulationError: {}", message)
            }
        }
    }
}
