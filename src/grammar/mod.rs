use crate::error::ParseError;
use maplit::hashmap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt::Write;
use std::io::ErrorKind;
use std::path::Path;

mod bootstrap;
use crate::grammar::bootstrap::bootstrap_grammar;

/// Struct representing a lexer production.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexerProduction {
    /// Name of the production.
    pub head: String,
    /// Body of the production.
    pub body: Tree<LexerRuleElement>,
    /// If this production belongs to a lexer, this field contains the lexer actions.
    pub actions: BTreeSet<Action>,
}

/// Struct representing a parser production.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserProduction {
    /// Name of the production.
    pub head: String,
    /// Body of the production.
    pub body: Tree<ParserRuleElement>,
}

/// Struct representing a parsed grammar.
///
/// This struct stores terminal and non-terminal productions.
/// This struct also record the lexer actions for each terminal production, but drops any embedded
/// action as they are language dependent.
///
/// Additionally, multiple lexer modes can be used in this grammar: each mode has different set of
/// rules, and switching from one modes to the other ones can be done with lexer actions. More info
/// can be found on the [ANTLR
/// specification](https://github.com/antlr/antlr4/blob/master/doc/lexer-rules.md#lexical-modes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    //vector containing the bodies of the terminal productions.
    //the first dimension of the array represent the lexer mode (context).
    terminals: Vec<Vec<LexerProduction>>,
    //vector containing the bodies of the non-terminal productions
    non_terminals: Vec<ParserProduction>,
    // map a mode name to a specific index, used in the first dimension of this struct lexer rules
    modes_index: HashMap<String, u32>,
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
            self.terminals[*mode_index as usize].len()
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

    /// Returns the amount of modes used in this grammar.
    /// # Examples
    /// ```
    /// panic!()
    /// ```
    pub fn len_modes(&self) -> usize {
        self.modes_index.len()
    }

    /// Returns an iterator over the modes of this grammar.
    ///
    /// The modes are guaranteed to be indexed by their ID.
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
        let mut vec = self
            .modes_index
            .iter()
            .map(|(k, v)| (v, k.as_str()))
            .collect::<Vec<_>>();
        vec.sort_unstable();
        vec.into_iter().map(|(_, v)| v)
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
    pub fn iter_term(&self) -> impl Iterator<Item = &LexerProduction> {
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
    pub fn iter_term_in_mode(&self, mode: &str) -> impl Iterator<Item = &LexerProduction> {
        if let Some(index) = self.modes_index.get(mode) {
            if let Some(terminals) = self.terminals.get(*index as usize) {
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
    pub fn iter_nonterm(&self) -> impl Iterator<Item = &ParserProduction> {
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
        let grammar_content = std::fs::read_to_string(path.as_ref())?;
        match path.as_ref().extension().and_then(|ext| ext.to_str()) {
            Some("g4") | Some("g") => Self::parse_antlr(&grammar_content),
            Some("bootstrap") => Self::parse_bootstrap(&grammar_content),
            _ => Err(ParseError::IOError(std::io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "unsupported grammar or missing extension: {:?}",
                    path.as_ref()
                ),
            ))),
        }
    }

    /// Builds a grammar using a simil-ANTLR syntax.
    ///
    /// The main purpose of this method is to read a grammar without depending on this crate
    /// itself. This is done by manually implementing a recursive descent parser.
    ///
    /// The recognized grammar is similar to ANTLR in syntax, with a few key difference to
    /// simplify the manual implementation of the parser:
    /// - `=` instead of `:` (originally this was an EBNF grammar)
    /// - no modes or lexer actions are supported.
    /// - no escaping is possible.
    /// - ranges must be fully specified (`[abc]` instead of `[a-c]`).
    /// - no unicode is supported.
    /// - no fragments are supported.
    /// - the syntax 'a'..'c' is not supported.
    pub fn parse_bootstrap(content: &str) -> Result<Grammar, ParseError> {
        bootstrap_grammar(content)
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
        todo!()
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
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum Action {
    /// Action telling the lexer to not return the matched token.
    Skip,
    /// Action telling the lexer to match the current rule but continue collecting tokens.
    More,
    /// Action assigning a specific type for the matched token.
    /// **Not supported in this implementation**
    Type,
    /// Action telling the lexer to switch to a specific channel after matching the token.
    /// **Not supported in this implementation**
    Channel,
    /// After matching the token, the lexer will switch to the mode passed as String. Only rules
    /// matching the newly passed mode will be matched.
    Mode(u32),
    /// Same behaviour of `Action::MODE` but the mode is pushed on a stack, to be later popped by
    /// `Action::POPMODE`.
    PushMode(u32),
    /// After matching the token, pop a mode from the mode stack and continue matching tokens using
    /// the mode on the top of the stack.
    PopMode,
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Skip => write!(f, "SKIP"),
            Action::More => write!(f, "MORE"),
            Action::Type => write!(f, "TYPE"),
            Action::Channel => write!(f, "CHANNEL"),
            Action::Mode(m) => write!(f, "MODE({})", m),
            Action::PushMode(p) => write!(f, "PUSHMODE({})", p),
            Action::PopMode => write!(f, "POPMODE"),
        }
    }
}

/// The type of operation that can be found in a lexer production.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum LexerOp {
    /// Kleene star operator `*`.
    Kleene,
    /// Question mark operator `?`.
    Qm,
    /// Plus sign operator `+`.
    Pl,
    /// Not sign operator `~`.
    Not,
    /// Alternation between elements `|`.
    Or,
    /// Concatenation of elements `&`.
    And,
}

impl std::fmt::Display for LexerOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LexerOp::Kleene => write!(f, "*"),
            LexerOp::Qm => write!(f, "?"),
            LexerOp::Pl => write!(f, "+"),
            LexerOp::Not => write!(f, "~"),
            LexerOp::Or => write!(f, "|"),
            LexerOp::And => write!(f, "&"),
        }
    }
}

/// The elements that can be found in a lexer production.
///
/// A literal should be represented as a concatenation of 1-element [`CharSet`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexerRuleElement {
    /// A set containing `T`. `T` is usually a character or its ID in the [`SymbolTable`].
    CharSet(BTreeSet<char>),
    /// The *any value* operator `.`.
    AnyValue,
    /// An operation between the other elements.
    Operation(LexerOp),
}

impl std::fmt::Display for LexerRuleElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerRuleElement::CharSet(i) => {
                write!(f, "[")?;
                for charz in i {
                    write!(f, "{}", charz)?;
                }
                write!(f, "]")
            }
            LexerRuleElement::AnyValue => write!(f, "."),
            LexerRuleElement::Operation(tp) => write!(f, "{}", tp),
        }
    }
}

/// The elements that can be found in a parser production.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParserRuleElement {
    NonTerminal(String),
    Terminal(String),
    Empty,
    Operation(LexerOp),
}

impl std::fmt::Display for ParserRuleElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParserRuleElement::NonTerminal(name) => write!(f, "NT[{}]", name),
            ParserRuleElement::Terminal(name) => write!(f, "T[{}]", name),
            ParserRuleElement::Empty => write!(f, "ε"),
            ParserRuleElement::Operation(tp) => write!(f, "{}", tp),
        }
    }
}

/// Trait used to represents various object in [Graphviz Dot notation](https://graphviz.org/).
pub trait GraphvizDot {
    /// Returns a graphviz dot representation of the object as string.
    /// # Examples
    /// Implementation of the [`Dfa`] class:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::lexer::{MultiDfa, GraphvizDot};
    /// let grammar = Grammar::new(
    ///     &[("LETTER_A", "'a'").into(), ("LETTER_B", "'b'*").into()],
    ///     &[],
    /// );
    /// let dfa = MultiDfa::new(&grammar);
    /// dfa.to_dot();
    /// ```
    fn to_dot(&self) -> String;

    /// Writes a graphviz dot representation to to file.
    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_dot())
    }
}

/// A Tree data structure.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Tree<T> {
    /// The value contained in the tree.
    value: T,
    /// The right child of the tree.
    children: Vec<Tree<T>>,
}

impl<T> Tree<T> {
    /// Creates a new leaf node with the given value.
    pub fn new_leaf(value: T) -> Self {
        Self {
            value,
            children: Vec::new(),
        }
    }
    /// Creates a new node with the given value and children.
    pub fn new_node(value: T, children: Vec<Tree<T>>) -> Self {
        Self { value, children }
    }

    /// Retrieves the value contained inside the node.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Retrieves the mutable value contained inside the node.
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Pushes a child to the current node
    pub fn add_child(&mut self, child: Self) {
        self.children.push(child)
    }

    /// Returns a reference to the nth child.
    pub fn child(&self, nth: usize) -> Option<&Self> {
        self.children.get(nth)
    }

    /// Returns a mutable reference to the nth child.
    pub fn child_mut(&mut self, nth: usize) -> Option<&mut Self> {
        self.children.get_mut(nth)
    }

    /// Consumes the node and iterates its children.
    pub fn into_children(self) -> impl DoubleEndedIterator<Item = Self> {
        self.children.into_iter()
    }

    /// Iterator visiting references to the node children.
    pub fn children(&self) -> impl DoubleEndedIterator<Item = &Self> {
        self.children.iter()
    }

    /// Returns the amount of children contained in this node.
    pub fn children_len(&self) -> usize {
        self.children.len()
    }

    /// Iterator visiting mutable references to the node children.
    pub fn children_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Self> {
        self.children.iter_mut()
    }

    /// Iterator performing a pre-order depth-first visit of the Tree
    pub fn dfs_preorder(&self) -> impl DoubleEndedIterator<Item = &Self> {
        let mut order = Vec::new();
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            order.push(node);
            stack.extend(node.children().rev());
        }
        order.into_iter()
    }

    /// Iterator performing a pre-order breadth-first visit of the Tree
    pub fn bfs_postorder(&self) -> impl DoubleEndedIterator<Item = &Self> {
        let mut order = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(self);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            queue.extend(node.children())
        }
        order.into_iter()
    }
}

impl<T: std::fmt::Display> GraphvizDot for Tree<T> {
    fn to_dot(&self) -> String {
        let mut retval = "digraph Tree {\n".to_string();
        let mut next_id = 0;
        let mut nodes = vec![(self, next_id)];
        next_id += 1;
        while let Some((node, id)) = nodes.pop() {
            writeln!(retval, "    {}[label=\"{}\"];", id, node.value).unwrap();
            for child in node.children() {
                nodes.push((child, next_id));
                writeln!(retval, "    {}->{}", id, next_id).unwrap();
                next_id += 1;
            }
        }
        retval.push('}');
        retval
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Tree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut stack = vec![(0, self)];
        while let Some((indent, node)) = stack.pop() {
            write!(f, "{}{:?}", " ".repeat(indent * 4), node.value())?;
            stack.extend(node.children().rev().map(|x| (indent + 1, x)));
            if !stack.is_empty() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
