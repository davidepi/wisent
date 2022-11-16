use crate::error::ParseError;
use std::collections::BTreeSet;

/// Code used to parse an ANTLR grammar with and ad-hoc parser
mod bootstrap;
use bootstrap::bootstrap_parse_string;

/// Struct representing a parsed grammar.
///
/// This struct stores terminal and non-terminal productions in the form `head`:`body`; and allows
/// to access every `body` given a particular `head`
/// This struct also record the lexer actions for each terminal production, but drops any embedded
/// action as they are language dependent.
#[derive(Debug, Default, Clone)]
pub struct Grammar {
    //vector containing the bodies of the terminal productions
    pub(crate) terminals: Vec<String>,
    //lexer actions (ANTLR-specific feature for g4 grammars)
    pub(crate) actions: Vec<BTreeSet<Action>>,
    //vector containing the bodies of the non-terminal productions
    pub(crate) non_terminals: Vec<String>,
    //vector recording the name of each terminal production
    names_terminals: Vec<String>,
    //vector recording the name of each non-terminal production
    names_non_terminals: Vec<String>,
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
    /// No checks will be performed on the productions naming and no recursion will be resolved.
    /// This method is used mostly for debug purposes, and [`parse_grammar()`] or
    /// [`parse_string()`]
    /// should be used.
    ///
    /// The following arguments are expected:
    /// * `terminals` - A slice of strings representing the terminal productions' bodies.
    ///                 The tuple represents (*<terminal name>* | *<terminal>*)
    /// * `non_terminals` - A slice of strings representing the non_terminal productions' bodies.
    ///                     The tuple represents (*<non_terminal name>* | *<non_terminal>*)
    /// order. First all the terminals are read, then the non-terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let terminals = vec![("LETTER_LOWERCASE", "[a-z]"), ("LETTER_UPPERCASE", "[A-Z]")];
    /// let non_terminals = vec![
    ///     ("letter", "LETTER_UPPERCASE | LETTER_LOWERCASE"),
    ///     ("word", "word letter | letter"),
    /// ];
    /// let grammar = Grammar::new(&terminals, &non_terminals);
    /// ```
    pub fn new(terminals: &[(&str, &str)], non_terminals: &[(&str, &str)]) -> Grammar {
        let terminals_no = terminals.len();
        let (names_terminals, terminals): (Vec<String>, Vec<String>) = terminals
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .unzip();
        let (names_non_terminals, non_terminals): (Vec<String>, Vec<String>) = non_terminals
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .unzip();
        Grammar {
            terminals,
            non_terminals,
            actions: vec![BTreeSet::new(); terminals_no],
            names_terminals,
            names_non_terminals,
        }
    }

    /// Returns the total number of productions.
    ///
    /// This includes terminals and non-terminals, but not fragments.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_UP: [A-Z];
    ///     LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// assert_eq!(grammar.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.terminals.len() + self.non_terminals.len()
    }

    /// Returns the total number of terminal productions.
    ///
    /// Note that fragments are excluded from the count, as they are merged within the terminals and
    /// non-terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_UP: [A-Z];
    ///     LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// assert_eq!(grammar.len_term(), 2);
    /// ```
    pub fn len_term(&self) -> usize {
        self.terminals.len()
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
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// assert_eq!(grammar.len_nonterm(), 1);
    /// ```
    pub fn len_nonterm(&self) -> usize {
        self.non_terminals.len()
    }

    /// Checks if the grammar has no productions.
    ///
    /// Returnts true if the grammar has exactly 0 productions, false otherwise. This comprises both
    /// terminals and non terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let grammar = Grammar::new(Vec::new().as_slice(), Vec::new().as_slice());
    /// assert!(grammar.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty() && self.non_terminals.is_empty()
    }

    /// Returns the name of the nth terminal production.
    ///
    /// Productions are expressed in the form `head: body;` and assigned an index (the order in
    /// which they appear in the grammar file. This method takes that index  and returns the `head`
    /// for terminals or `None` if the index was not found.
    ///
    /// The index correspond to the one found within the result of the [`Grammar::iter_term`]
    /// method.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    /// LETTER_UPPERCASE: [A-Z];
    /// LETTER_LOWERCASE: [a-z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// let head = grammar.name_term(1).unwrap();
    /// assert_eq!(head, "LETTER_LOWERCASE");
    /// ```
    pub fn name_term(&self, index: usize) -> Option<&String> {
        self.names_terminals.get(index)
    }

    /// Returns the name of the nth non-terminal production.
    ///
    /// Productions are expressed in the form `head: body;` and assigned an index (the order in
    /// which they appear in the grammar file. This method takes that index  and returns the `head`
    /// for terminals or `None` if the index was not found.
    ///
    /// The index correspond to the one found within the result of the [`Grammar::iter_nonterm`]
    /// method.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///          letter: LETTER_UP | LETTER_LO;
    ///          LETTER_UP: [A-Z];
    ///          LETTER_LO: [a-z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// let head = grammar.name_nonterm(0).unwrap();
    /// assert_eq!(head, "letter");
    /// ```
    pub fn name_nonterm(&self, index: usize) -> Option<&String> {
        self.names_non_terminals.get(index)
    }

    /// Returns the lexer action for a given terminal index.
    ///
    /// Lexer actions are an ANTLR-specific, language independent feature useful only to the lexer.
    /// For more information refer to the ANTLR specification.
    ///
    /// This method exprects the production index as input and returns a set containing the actions
    /// for the given production.
    ///
    /// If the requested index is out of bounds a panic will be thrown.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::{Grammar,Action};
    /// let g = "grammar g;
    ///    LETTER_UPPERCASE: [A-Z] -> more, mode( NEW_MODE);
    ///     LETTER_LOWERCASE: [a-z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// let mut actions0 = grammar.action(0).iter();
    /// let actions1 = grammar.action(1);
    /// assert_eq!(*actions0.next().unwrap(), wisent::grammar::Action::MORE);
    /// assert_eq!(
    ///     *actions0.next().unwrap(),
    ///     Action::MODE("NEW_MODE".to_owned())
    /// );
    /// assert!(actions1.is_empty());
    /// ```
    pub fn action(&self, index: usize) -> &BTreeSet<Action> {
        &self.actions[index]
    }

    /// Returns an iterator over the terminals slice.
    ///
    /// This method is just a wrapper of `iter()` and as such does not take ownership.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_LO: [a-z];
    ///     LETTER_UP: [A-Z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// let mut iterator = grammar.iter_term();
    /// assert_eq!(iterator.next(), Some(&"[a-z]".to_owned()));
    /// assert_eq!(iterator.next(), Some(&"[A-Z]".to_owned()));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter_term(&self) -> std::slice::Iter<String> {
        self.terminals.iter()
    }

    /// Returns an iterator over the non-terminals slice.
    ///
    /// This method is just a wrapper of `iter()` and as such does not take ownership.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let g = "grammar g;
    ///     letter:LT_LO | LT_UP;
    ///     LT_LO: [a-z];
    ///     LT_U: [A-Z];";
    /// let grammar = Grammar::parse_string(g).unwrap();
    /// let mut iterator = grammar.iter_nonterm();
    /// assert_eq!(iterator.next(), Some(&"LT_LO | LT_UP".to_owned()));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter_nonterm(&self) -> std::slice::Iter<String> {
        self.non_terminals.iter()
    }

    /// Builds a grammar from an ANTLR `.g4` file.
    ///
    /// This method constructs and initializes a Grammar class by parsing an external specification
    /// written in a `.g4` file.
    ///
    /// A path pointing to the `.g4` file is expected as input.
    ///
    /// In case the file cannot be found or contains syntax errors a ParseError is returned.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// # use wisent::grammar::Grammar;
    /// let grammar = Grammar::parse_grammar("Rust.g4").unwrap();
    /// ```
    pub fn parse_grammar(path: &str) -> Result<Grammar, ParseError> {
        let grammar_content = std::fs::read_to_string(path)?;
        Self::parse_string(&grammar_content[..])
    }

    /// Builds a grammar from a String with the content of an ANTLR `.g4` file.
    ///
    /// This method constructs and initializes a Grammar class by parsing a String following the
    /// ANTLR `.g4` specification.
    ///
    /// A ParseError is returned in case the String contains syntax errors.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// let cont = "grammar g; letter:[a-z];";
    /// let grammar = Grammar::parse_string(cont).unwrap();
    /// assert_eq!(grammar.len(), 1);
    /// ```
    pub fn parse_string(content: &str) -> Result<Grammar, ParseError> {
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
    SKIP,
    /// Action telling the lexer to match the current rule but continue collecting tokens.
    MORE,
    /// Action assigning a specific type for the matched token.
    /// The type is passed as a String parameter.
    TYPE(String),
    /// Action telling the lexer to switch to a specific channel after matching the token.
    /// The name of the channel is passed as a String parameter.
    CHANNEL(String),
    /// After matching the token, the lexer will switch to the mode passed as String. Only rules
    /// matching the newly passed mode will be matched.
    MODE(String),
    /// Same behaviour of `Action::MODE` but the mode is pushed on a stack, to be later popped by
    /// `Action::POPMODE`.
    PUSHMODE(String),
    /// After matching the token, pop a mode from the mode stack and continue matching tokens using
    /// the mode on the top of the stack.
    POPMODE,
}
