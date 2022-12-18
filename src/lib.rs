/// Module containing the errors that may arise during the parser generation.
pub mod error;
/// Module providing the definition of a Grammar.
// contains also the bootstrapping code to read an ANTLR grammar.
pub mod grammar;
/// Module responsible of generating the lexer transition table and running it.
pub mod lexer;
/// Module responsible of generating the parser transition table and running it.
pub mod parser;
