/// Module containing the errors that may arise during the parser generation.
pub mod error;
/// Module providing the definition of a Grammar.
pub mod grammar;
/// Module responsible of parsing an ANTLR grammar file without using anything outside the stl.
mod grammar_bootstrap;
/// Module responsible of generating the lexer transition table.
pub mod lexer;

#[macro_use]
extern crate maplit;
