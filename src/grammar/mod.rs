use crate::error::ParseError;
use std::collections::{BTreeSet, HashMap};
use std::ops::Index;

/// Code used to parse an ANTLR grammar with and ad-hoc parser
mod bootstrap;
use bootstrap::bootstrap_parse_string;

/// Struct representing a parsed grammar.
///
/// This struct stores terminal and non-terminal productions in the form `head`:`body`; and allows
/// to access every `body` given a particular `head`
/// This struct also record the lexer actions for each terminal production, but drops any embedded
/// action as they are language dependent.
pub struct Grammar {
    //vector containing the bodies of the terminal productions
    pub(crate) terminals: Vec<String>,
    //lexer actions (ANTLR-specific feature for g4 grammars)
    pub(crate) actions: Vec<BTreeSet<Action>>,
    //vector containing the bodies of the non-terminal productions
    pub(crate) non_terminals: Vec<String>,
    //map assigning a tuple (index, is_terminal?) to the productions' heads
    pub(crate) names: HashMap<String, (usize, bool)>,
}

impl Grammar {
    /// Constructs a new Grammar with the given terminals and non terminals.
    ///
    /// No checks will be performed on the productions naming and no recursion will be resolved.
    /// This method is used mostly for debug purposes, and `parse_grammar()` or `parse_string()`
    /// should be used.
    ///
    /// The following arguments are expected:
    /// * `terminals` - A slice of strings representing the terminal productions' bodies.
    /// * `non_terminals` - A slice of strings representing the non_terminal productions' bodies.
    /// * `names` - A slice of strings representing the names of every terminal and non terminal in
    /// order. First all the terminals are read, then the non-terminals.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let terminals = vec!["[a-z]", "[A-Z]"];
    /// let non_terminals = vec![
    ///     "LETTER_UPPERCASE | LETTER_LOWERCASE",
    ///     "word letter | letter",
    /// ];
    /// let names = vec!["LETTER_LOWERCASE", "LETTER_UPPERCASE", "letter", "word"];
    /// let grammar = wisent::grammar::Grammar::new(&terminals, &non_terminals, &names);
    /// ```
    pub fn new(terminals: &[&str], non_terminals: &[&str], names: &[&str]) -> Grammar {
        let terms = terminals.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let nterms = non_terminals
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        let mut map = HashMap::new();
        for (idx, item) in names.iter().enumerate() {
            let term = idx < terminals.len();
            map.insert(item.to_string(), (idx, term));
        }
        Grammar {
            terminals: terms,
            non_terminals: nterms,
            actions: Vec::new(),
            names: map,
        }
    }

    /// Returns the total number of productions.
    ///
    /// This includes terminals and non-terminals, but not fragments.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_UP: [A-Z];
    ///     LETTER_LO: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
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
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_UP: [A-Z];
    ///     LETTER_LO: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
    /// assert_eq!(grammar.len_term(), 2);
    /// ```
    pub fn len_term(&self) -> usize {
        self.terminals.len()
    }

    /// Returns the total number of non-terminal productions.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let g = "grammar g;
    ///          letter: LETTER_UP | LETTER_LO;
    ///          LETTER_UP: [A-Z];
    ///          LETTER_LO: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
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
    /// let grammar = wisent::grammar::Grammar::new(
    ///     Vec::new().as_slice(),
    ///     Vec::new().as_slice(),
    ///     Vec::new().as_slice(),
    /// );
    /// assert!(grammar.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty() && self.non_terminals.is_empty()
    }

    /// Returns the production body associated to a given head.
    ///
    /// Productions are expressed in the form `head: body;`. This method takes the `head` and
    /// returns the given `body` or None if the production does not exists.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let g = "grammar g;
    /// LETTER_UPPERCASE: [A-Z];
    /// LETTER_LOWERCASE: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
    /// let body = grammar.get("LETTER_LOWERCASE").unwrap();
    /// assert_eq!(body, "[a-z]");
    /// ```
    pub fn get(&self, head: &str) -> Option<&str> {
        if let Some(found) = self.names.get(head) {
            if found.1 {
                Some(&self.terminals[found.0])
            } else {
                Some(&self.non_terminals[found.0])
            }
        } else {
            None
        }
    }

    /// Returns the lexer action for a given terminal name.
    ///
    /// Lexer actions are an ANTLR-specific, language independent feature useful only to the lexer.
    /// For more information refer to the ANTLR specification.
    ///
    /// This method exprects the production name as input and returns a set containing the actions
    /// for the given production.
    ///
    /// If the requested name does not exists, None is returned.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let g = "grammar g;
    ///    LETTER_UPPERCASE: [A-Z] -> more, mode( NEW_MODE);
    ///     LETTER_LOWERCASE: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
    /// let act_lo = grammar.action("LETTER_LOWERCASE").unwrap();
    /// let mut act_up = grammar.action("LETTER_UPPERCASE").unwrap().iter();
    /// assert_eq!(*act_up.next().unwrap(), wisent::grammar::Action::MORE);
    /// assert_eq!(
    ///     *act_up.next().unwrap(),
    ///     wisent::grammar::Action::MODE("NEW_MODE".to_owned())
    /// );
    /// assert!(act_lo.is_empty());
    /// ```
    pub fn action(&self, head: &str) -> Option<&BTreeSet<Action>> {
        if let Some(found) = self.names.get(head) {
            Some(&self.actions[found.0])
        } else {
            None
        }
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
    /// let g = "grammar g;
    ///    LETTER_UPPERCASE: [A-Z] -> more, mode( NEW_MODE);
    ///     LETTER_LOWERCASE: [a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
    /// let mut actions0 = grammar.action_nth(0).iter();
    /// let actions1 = grammar.action_nth(1);
    /// assert_eq!(*actions0.next().unwrap(), wisent::grammar::Action::MORE);
    /// assert_eq!(
    ///     *actions0.next().unwrap(),
    ///     wisent::grammar::Action::MODE("NEW_MODE".to_owned())
    /// );
    /// assert!(actions1.is_empty());
    /// ```
    pub fn action_nth(&self, index: usize) -> &BTreeSet<Action> {
        &self.actions[index]
    }

    /// Returns an iterator over the terminals slice.
    ///
    /// This method is just a wrapper of `iter()` and as such does not take ownership.
    /// # Examples
    /// Basic usage:
    /// ```
    /// let g = "grammar g;
    ///     letter: LETTER_UP | LETTER_LO;
    ///     word: word letter | letter;
    ///     LETTER_LO: [a-z];
    ///     LETTER_UP: [A-Z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
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
    /// let g = "grammar g;
    ///     letter:LT_LO | LT_UP;
    ///     LT_LO: [a-z];
    ///     LT_U: [A-Z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(g).unwrap();
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
    /// let grammar = wisent::grammar::Grammar::parse_grammar("Rust.g4").unwrap();
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
    /// let cont = "grammar g; letter:[a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(cont).unwrap();
    /// assert_eq!(grammar.len(), 1);
    /// ```
    pub fn parse_string(content: &str) -> Result<Grammar, ParseError> {
        bootstrap_parse_string(content)
    }
}

impl Index<usize> for Grammar {
    type Output = String;

    fn index(&self, index: usize) -> &Self::Output {
        let idx;
        if index < self.terminals.len() {
            idx = index;
            &self.terminals[idx]
        } else {
            idx = index - self.terminals.len();
            &self.non_terminals[idx]
        }
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
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
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
