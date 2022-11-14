/// build.rs can not self-reference products of the crate being built.
/// This crate uses the crate's DFA to parse the grammar used by the parser generator.
/// This means that generating the implementation of multiple "grammar parsers" is not possible
/// until an existing version of this crates is ready.

use wisent::{grammar::Grammar, lexer::{Dfa, GraphvizDot}};
use std::env;

fn main() {
    bootstrap_antlr();
}

fn bootstrap_antlr() {
    const TERMINALS: [(&str, &str);25] = [
    ("JAVADOC_COMMENT", "'/**'(~[*]~[/])*'*/'"),
    ("BLOCK_COMMENT", "'/*'(~[*]~[/])*'*/'"),
    ("LINE_COMMENT", "'//'~[\\r\\n]*"),
    ("QUOTED", "'\\''~['\\r\\n\\\\]*'\\''"),
    ("FRAGMENT", "'fragment'"),
    ("IMPORT", "'import'"),
    ("LEXER", "'lexer'"),
    ("PARSER", "'parser'"),
    ("GRAMMAR","'grammar'"),
    ("COLON", "':'"),
    ("SEMI", "';'"),
    ("COMMA", "','"),
    ("LPAR", "'('"),
    ("RPAR", "')'"),
    ("LBRACE", "'{'"),
    ("RBRACE", "'}'"),
    ("ARROW", "'->'"),
    ("OR", "'|'"),
    ("TWO_DOTS", "'..'"),
    ("TILDE", "'~'"),
    ("STAR", "'*'"),
    ("PLUS", "'+'"),
    ("QUESTION", "'?'"),
    ("MINUS", "'-'"),
    ("WHITESPACE", "[\\t\\r\\n\\f ]+"),

];
    let grammar = Grammar::new(&TERMINALS, &[]);
    let dfa = Dfa::new(&grammar);
}
