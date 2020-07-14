/// Module containing the errors that may arise during the parser generation
pub mod error;

/// Module responsible of parsing and understanding the grammar file
pub mod grammar;
#[cfg(test)]
mod grammar_tests;

pub mod lexer;
#[cfg(test)]
mod lexer_tests;

#[macro_use]
extern crate maplit;
