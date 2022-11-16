use crate::error::ParseError;
use maplit::hashmap;
use std::collections::{BTreeSet, HashMap};
use std::io::ErrorKind;

use self::bootstrap::bootstrap_parse_string;

// load the manually written ANTLR g4 parser
mod bootstrap;

/// Struct representing a grammar production.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Production {
    /// Name of the production.
    pub head: String,
    /// Body of the production.
    pub body: String,
    /// If this production belongs to a lexer, this field contains the lexer actions.
    pub actions: Option<BTreeSet<Action>>,
}

impl<S: Into<String>> From<(S, S)> for Production {
    fn from((head, body): (S, S)) -> Self {
        Self {
            head: head.into(),
            body: body.into(),
            actions: None,
        }
    }
}

impl<S: Into<String>> From<(S, S, BTreeSet<Action>)> for Production {
    fn from((head, body, actions): (S, S, BTreeSet<Action>)) -> Self {
        Self {
            head: head.into(),
            body: body.into(),
            actions: Some(actions),
        }
    }
}

/// Struct representing a parsed grammar.
///
/// This struct stores terminal and non-terminal productions.
/// This struct also record the lexer actions for each terminal production, but drops any embedded
/// action as they are language dependent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    //vector containing the bodies of the terminal productions.
    //the first dimension of the array represent the lexer mode (context).
    terminals: Vec<Vec<Production>>,
    //vector containing the bodies of the non-terminal productions
    non_terminals: Vec<Production>,
    // map a mode name to a specific index, used in the first dimension of this struct lexer rules
    modes_index: HashMap<String, usize>,
}

impl Default for Grammar {
    fn default() -> Self {
        Self {
            modes_index: hashmap! {"DEFAULT_MODE".to_owned() => 0},
            terminals: Default::default(),
            non_terminals: Default::default(),
        }
    }
}

impl Grammar {
    /// Constructs an empty Grammar.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let grammar = Grammar::empty();
    ///
    /// assert!(grammar.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self::default()
    }

    /// Constructs a new Grammar with the given terminals and non terminals.
    ///
    /// The set of terminals and non terminals will be added to the `DEFAULT_MODE`.
    ///
    /// No checks will be performed on the productions naming and no recursion will be resolved.
    ///
    /// Using [`Grammar::parse_antlr`] or [`Grammar::parse_grammar`] instead of this method is
    /// **HIGHLY** recommended.
    pub fn new(terminals: &[Production], non_terminals: &[Production]) -> Self {
        Grammar {
            terminals: vec![terminals.to_vec()],
            non_terminals: non_terminals.to_vec(),
            ..Default::default()
        }
    }

    /// Adds lexer productions to the grammar.
    ///
    /// Adds additional lexer productions to the grammar with the given mode.
    ///
    /// This method has the same limitations of [`Grammar::new`], being intended for debug
    /// purposes, and [`Grammar::parse_antlr`] or [`Grammar::parse_grammar`] are suggested.
    ///
    /// The terminal tuple is similar to the one explained in [`Grammar::new`], with the
    /// addition of the action set for each recognized terminal.
    pub fn add_terminals(&mut self, mode: String, terminals: &[Production]) {
        let next_mode = self.modes_index.len();
        let mode_index = *self.modes_index.entry(mode).or_insert(next_mode);
        self.terminals[mode_index].extend_from_slice(terminals);
    }

    /// Returns the total number of productions.
    ///
    /// This includes terminals and non-terminals, but not fragments.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///          letter: LETTER_UP | LETTER_LO;
    ///          word: word letter | letter;
    ///          LETTER_UP: [A-Z];
    ///          LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    /// assert_eq!(grammar.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len_term() + self.len_nonterm()
    }

    /// Returns the total number of terminal productions, in all modes.
    ///
    /// Note that fragments are excluded from the count, as they are merged within the terminals and
    /// non-terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///          letter: LETTER_UP | LETTER_LO;
    ///          word: word letter | letter;
    ///          LETTER_UP: [A-Z];
    ///          LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    /// assert_eq!(grammar.len_term(), 2);
    /// ```
    pub fn len_term(&self) -> usize {
        self.terminals.iter().fold(0, |acc, term| acc + term.len())
    }

    /// Returns the total number of terminal production in the given mode.
    ///
    /// Note that fragments are excluded from the count, as they are merged within the terminals and
    /// non-terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "lexer grammar Strings;
    ///          LQUOTE : '\"' -> more, mode(STR) ;
    ///          WS : [ \r\t\n]+ -> skip ;
    ///          mode STR;
    ///          STRING : '\"' -> mode(DEFAULT_MODE) ; // token we want parser to see
    ///          TEXT : . -> more ; // collect more text for string";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    ///
    /// assert_eq!(grammar.len_term(), 4);
    /// assert_eq!(grammar.len_term_in_mode("DEFAULT_MODE"), 2);
    /// assert_eq!(grammar.len_term_in_mode("STR"), 2);
    /// ```
    pub fn len_term_in_mode(&self, mode: &str) -> usize {
        if let Some(mode_index) = self.modes_index.get(mode) {
            self.terminals[*mode_index].len()
        } else {
            0
        }
    }

    /// Returns the total number of non-terminal productions.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///          letter: LETTER_UP | LETTER_LO;
    ///          LETTER_UP: [A-Z];
    ///          LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    /// assert_eq!(grammar.len_nonterm(), 1);
    /// ```
    pub fn len_nonterm(&self) -> usize {
        self.non_terminals.len()
    }

    /// Checks if the grammar has no productions.
    ///
    /// Returns true if the grammar has exactly 0 productions, false otherwise. This comprises both
    /// terminals and non terminals, in all modes.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let grammar = Grammar::empty();
    /// assert!(grammar.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len_term() == 0 && self.non_terminals.is_empty()
    }

    /// Returns an iterator over the modes of this grammar.
    ///
    /// Iterates by name the various modes used by the lexer of this grammar.
    /// # Examples
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "lexer grammar Strings;
    ///          LQUOTE : '\"' -> more, mode(STR) ;
    ///          WS : [ \r\t\n]+ -> skip ;
    ///          mode STR;
    ///          STRING : '\"' -> mode(DEFAULT_MODE) ; // token we want parser to see
    ///          TEXT : . -> more ; // collect more text for string";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    ///
    /// assert_eq!(grammar.iter_modes().count(), 2);
    /// ```
    pub fn iter_modes(&self) -> impl Iterator<Item = &str> {
        self.modes_index.keys().map(String::as_str)
    }

    /// Returns an iterator over the terminals slice in DEFAULT_MODE.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "lexer grammar Strings;
    ///          LQUOTE : '\"' -> more, mode(STR) ;
    ///          WS : [ \r\t\n]+ -> skip ;
    ///          mode STR;
    ///          STRING : '\"' -> mode(DEFAULT_MODE) ; // token we want parser to see
    ///          TEXT : . -> more ; // collect more text for string";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    ///
    /// assert_eq!(grammar.iter_term().count(), 2);
    /// ```
    pub fn iter_term(&self) -> impl Iterator<Item = &Production> {
        if let Some(terminals) = self.terminals.get(0) {
            terminals.iter()
        } else {
            [].iter()
        }
    }

    /// Returns an iterator over the terminals slice in the given mode.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "lexer grammar Strings;
    ///          LQUOTE : '\"' -> more, mode(STR) ;
    ///          WS : [ \r\t\n]+ -> skip ;
    ///          mode STR;
    ///          STRING : '\"' -> mode(DEFAULT_MODE) ; // token we want parser to see
    ///          TEXT : . -> more ; // collect more text for string";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    ///
    /// assert_eq!(grammar.iter_term_in_mode("STR").count(), 2);
    /// ```
    pub fn iter_term_in_mode(&self, mode: &str) -> impl Iterator<Item = &Production> {
        if let Some(index) = self.modes_index.get(mode) {
            if let Some(terminals) = self.terminals.get(*index) {
                terminals.iter()
            } else {
                [].iter()
            }
        } else {
            [].iter()
        }
    }

    /// Returns an iterator over the non-terminals slice.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///          letter:LT_LO | LT_UP;
    ///          LT_LO: [a-z];
    ///          LT_U: [A-Z];";
    /// let grammar = Grammar::parse_antlr(g).unwrap();
    /// let mut iterator = grammar.iter_nonterm();
    ///
    /// assert_eq!(iterator.next().unwrap().body, "LT_LO | LT_UP");
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter_nonterm(&self) -> impl Iterator<Item = &Production> {
        self.non_terminals.iter()
    }

    /// Builds a grammar by reading a given file.
    ///
    /// This method constructs and initializes a Grammar class by parsing an external specification
    /// written in a `.g4` file.
    ///
    /// A path pointing to the `.g4` file is expected as input.
    ///
    /// In case the file cannot be found or contains syntax errors a ParseError is returned.
    /// # Errors
    /// Returns error if the given path is missing the extension.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// # use wisent::grammar::Grammar;
    /// let grammar = Grammar::parse_grammar("Rust.g4").unwrap();
    /// ```
    pub fn parse_grammar<P: AsRef<std::path::Path>>(path: P) -> Result<Grammar, ParseError> {
        if let Some(ext) = path.as_ref().extension() {
            let grammar_content = std::fs::read_to_string(path.as_ref())?;
            let extension = ext.to_str().unwrap();
            match extension {
                "g4" => Self::parse_antlr(&grammar_content),
                _ => Err(ParseError::IOError(std::io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unsupported file type {}", extension),
                ))),
            }
        } else {
            Err(ParseError::IOError(std::io::Error::new(
                ErrorKind::InvalidInput,
                "filename is missing the extension",
            )))
        }
    }

    /// Builds a grammar from a String with the content in ANTLR syntax.
    ///
    /// This method constructs and initializes a Grammar class by parsing a String following the
    /// ANTLR4 specification.
    ///
    /// A ParseError is returned in case the String contains syntax errors.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let cont = "grammar g; letter:[a-z];";
    /// let grammar = Grammar::parse_antlr(cont).unwrap();
    /// assert_eq!(grammar.len(), 1);
    /// ```
    pub fn parse_antlr(content: &str) -> Result<Grammar, ParseError> {
        bootstrap_parse_string(content)
    }
}

/// Enum representing the possible lexer actions supported by the ANTLR lexer.
///
/// These actions are default operations that aims to give language-independent instruction to the
/// lexer.
///
/// These action are expressed after a production in the form `head: body -> action;` where action
/// can assume only specific values.
///
/// A brief documentation is provided for each action, but the user should refer to the ANTLR
/// reference.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Action {
    /// Action telling the lexer to not return the matched token.
    Skip,
    /// Action telling the lexer to match the current rule but continue collecting tokens.
    More,
    /// Action assigning a specific type for the matched token.
    /// The type is passed as a String parameter.
    Type(String),
    /// Action telling the lexer to switch to a specific channel after matching the token.
    /// The name of the channel is passed as a String parameter.
    Channel(String),
    /// After matching the token, the lexer will switch to the mode passed as String. Only rules
    /// matching the newly passed mode will be matched.
    Mode(String),
    /// Same behaviour of `Action::MODE` but the mode is pushed on a stack, to be later popped by
    /// `Action::POPMODE`.
    PushMode(String),
    /// After matching the token, pop a mode from the mode stack and continue matching tokens using
    /// the mode on the top of the stack.
    PopMode,
}
