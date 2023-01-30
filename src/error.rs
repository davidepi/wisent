/// Error wrapping all the possible kind of errors encountered during parsing.
///
/// The possible errors can be:
/// * `IOError` - containing an `std::io::Error`, this kind of error can arise
/// when opening the grammar on disks fail.
/// * `SyntaxError` - An error containing a `message` String that can arise when
/// parsing fails due to syntax errors in the input grammar.
/// * `LLError` - An error arising when the original grammar can not be converted into a LL
/// grammar. This can happen if the grammar is left-recursive or if there are FIRST or FOLLOW
/// conflicts.
/// * `DeserializeError` - An error arising during deserialization.
/// * `InternalError` - TODO
#[derive(Debug)]
pub enum ParseError {
    IOError(std::io::Error),
    SyntaxError { message: String },
    LLError { message: String },
    ParsingError { message: String },
    DeserializeError { message: String },
    InternalError { message: String },
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
            ParseError::LLError { message } => write!(buffer, "Grammar is not LL: {}", message),
            ParseError::ParsingError { message } => write!(buffer, "ParsingError: {}", message),
            ParseError::DeserializeError { message } => {
                write!(buffer, "DeserializeError: {}", message)
            }
            ParseError::InternalError { message } => write!(buffer, "InternalError: {}", message),
        }
    }
}
