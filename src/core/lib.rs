/// Module containing the errors that may arise during the parser generation
pub mod error;
/// Module responsible of parsing and understanding the grammar file
pub mod grammar;
/// Module responsible of generating the lexer transition table
pub mod lexer;

#[macro_use]
extern crate maplit;
