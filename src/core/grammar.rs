use std::collections::{BTreeSet, HashMap};
use std::ops::Index;

use regex::Regex;

use crate::error::ParseError;

#[derive(Debug)]
/// Struct representing a parsed grammar.
/// This struct stores terminal and non-terminal productions in the form `head`:`body`; and allows
/// to access every `body` given a particular `head`
/// This struct also record the lexer actions for each terminal production, but drops any embedded
/// action as they are language dependent.
pub struct Grammar {
    //vector containing the bodies of the terminal productions
    terminals: Vec<String>,
    //lexer actions (ANTLR-specific feature for g4 grammars)
    actions: Vec<BTreeSet<Action>>,
    //vector containing the bodies of the non-terminal productions
    non_terminals: Vec<String>,
    //map assigning a tuple (index, is_terminal?) to the productions' heads
    names: HashMap<String, (usize, bool)>,
}

impl Grammar {
    /// Constructs a new Grammar with the given terminals and non terminals.
    /// No checks will be performed on the productions naming and no recursion will be resolved.
    /// This method is used mostly for debug purposes, and `parse_grammar()` or `parse_string()`
    /// should be used.
    /// # Arguments
    /// * `terminals` - A slice of strings representing the terminal productions' bodies.
    /// * `non_terminals` - A slice of strings representing the non_terminal productions' bodies.
    /// * `names` - A slice of strings representing the names of every terminal and non terminal in
    /// order. First all the terminals are read, then the non-terminals.
    /// # Returns
    /// The newly created Grammar struct.
    /// # Examples
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

    /// Returns the total number of productions. This includes terminals and non-terminals but not
    /// fragments.
    /// # Returns
    /// A number representing the sum of terminals and non-terminals productions.
    /// # Examples
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
    /// Note that fragments are excluded from the count, as they are merged within the terminals and
    /// non-terminals.
    /// # Returns
    /// A number representing the amount of terminals.
    /// # Examples
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
    /// # Returns
    /// A number representing the amount of non-terminals.
    /// # Examples
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
    /// This comprises both terminals and non terminals.
    /// # Returns
    /// True if the grammar has exactly 0 productions, false otherwise.
    /// # Examples
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
    /// Productions are expressed in the form `head: body;`. This method takes the `head` and
    /// returns the given `body` or None if the production does not exists.
    /// # Arguments
    /// * `head` - The head, or name, of the production.
    /// # Returns
    /// An option containing the given production body or None if the production does not exists.
    /// # Examples
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
    /// Lexer actions are an ANTLR-specific, language independent feature useful only to the lexer.
    /// For more information refer to the ANTLR specification.
    /// # Arguments
    /// * `head` - The head, or name, of the terminal production.
    /// # Returns
    /// A string representing the requested action. The action is guaranteed to be returned WITHOUT
    /// any space or the arrow `->`. For example the production `[a-z] -> skip` will return `skip`
    /// as action.
    /// If the action does not exists, empty string is returned
    /// If the requested name does not exists, None is returned
    /// # Examples
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
    /// Lexer actions are an ANTLR-specific, language independent feature useful only to the lexer.
    /// For more information refer to the ANTLR specification.
    /// # Arguments
    /// * `index` - The index of the terminal production
    /// # Returns
    /// A string representing the requested action. The action is guaranteed to be returned WITHOUT
    /// any space or the arrow `->`. For example the production `[a-z] -> skip` will return `skip`
    /// as action.
    /// If the action does not exists, empty string is returned
    /// # Examples
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
    /// This method is just a wrapper of `iter()` and as such does not take ownership.
    /// # Returns
    /// An iterator over the terminals slice.
    /// # Examples
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
    /// This method is just a wrapper of `iter()` and as such does not take ownership.
    /// # Returns
    /// An iterator over the non-terminals slice.
    /// # Examples
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
    /// This method constructs and initializes a Grammar class by parsing an external specification
    /// written in a `.g4` file.
    /// This method effectively reads the files and forward the content to `parse_string()`
    /// # Arguments
    /// * `path` - The path pointing to the `.g4` file.
    /// # Results
    /// The newly constructed grammar or an error.
    /// # Errors
    /// A ParseError in case the file cannot be found or contains syntax errors.
    /// # Examples
    /// ```no_run
    /// let grammar = wisent::grammar::Grammar::parse_grammar("Rust.g4").unwrap();
    /// ```
    pub fn parse_grammar(path: &str) -> Result<Grammar, ParseError> {
        let grammar_content = std::fs::read_to_string(path)?;
        Self::parse_string(&grammar_content[..])
    }

    /// Builds a grammar from a String with the content of an ANTLR `.g4` file.
    /// This method constructs and initializes a Grammar class by parsing a String following the
    /// ANTLR `.g4` specification
    /// # Arguments
    /// * `content` - The content of the `.g4` file.
    /// # Results
    /// The newly constructed grammar or an error.
    /// # Errors
    /// A ParseError in case the String contains syntax errors.
    /// # Examples
    /// ```
    /// let cont = "grammar g; letter:[a-z];";
    /// let grammar = wisent::grammar::Grammar::parse_string(cont).unwrap();
    /// assert_eq!(grammar.len(), 1);
    /// ```
    pub fn parse_string(content: &str) -> Result<Grammar, ParseError> {
        let productions = retrieve_productions(&content);
        let grammar_rec = split_head_body(productions)?;
        let grammar_not_rec = resolve_terminals_dependencies(grammar_rec)?;
        let grammar = reindex(grammar_not_rec);
        Ok(grammar)
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

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
/// Enum representing the possible lexer actions supported by the ANTLR lexer. These actions are
/// default operations that aims to give language-independent instruction to the lexer.
///
/// These action are expressed after a production in the form `head: body -> action;` where action
/// can assume only specific values.
///
/// A brief documentation is provided for each action, but the user should refer to the ANTLR
/// reference.
pub enum Action {
    ///Action telling the lexer to not return the matched token.
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

///struct used internally. hash map are less space efficient but easier to use during constructions
///at some point I also drop the fragments map, but I prefer to pass them around using a single
///struct.
struct GrammarInternal {
    ///`key`:`value`; map containing terminals.
    terminals: HashMap<String, String>,
    ///`key`:`value`; map containing non-terminals.
    non_terminals: HashMap<String, String>,
    ///`key`:`value`; map containing fragments.
    fragments: HashMap<String, String>,
    ///array containing every production `key` in the order they appear in the file.
    order: Vec<String>,
    //`key`:`value`; map containing terminal actions
    actions: HashMap<String, BTreeSet<Action>>,
}

/// Transforms the Unordered maps of the GrammarInternal into the indexed Vec of the Grammar.
/// GrammarInternal uses an Hash-based indexing because it's more convenient for productions removal
/// (i.e. fragments). Grammar instead, uses a more efficient numerical indexing. Given that the
/// index matters, especially for the lexer, this method transform a GrammarInternal into a Grammar
/// while keeping the original indexing
/// # Arguments
/// * `grammar` - A grammar represented with a GrammarInternal.
/// # Returns
/// A grammar represented with a `Grammar` struct.
fn reindex(mut grammar: GrammarInternal) -> Grammar {
    let mut terminals = Vec::with_capacity(grammar.terminals.len());
    let mut non_terminals = Vec::with_capacity(grammar.non_terminals.len());
    let mut names = HashMap::with_capacity(grammar.terminals.len() + grammar.non_terminals.len());
    let mut actions = Vec::with_capacity(grammar.terminals.len());
    //the new order will be: first every terminal, then every non-terminal. In the original order.
    for head in grammar.order {
        let idx;
        let term;
        if let Some(body) = grammar.terminals.remove(&head) {
            idx = terminals.len();
            term = true;
            terminals.push(body);
            actions.push(grammar.actions.remove(&head).unwrap()); //this should exist
        } else if let Some(body) = grammar.non_terminals.remove(&head) {
            idx = non_terminals.len();
            term = false;
            non_terminals.push(body);
        } else {
            panic!("Expected production to be either in terminals or non terminals.");
        }
        names.insert(head, (idx, term));
    }
    Grammar {
        terminals,
        non_terminals,
        names,
        actions,
    }
}

/// Categorizes various productions into terminal, non-terminal and fragments. Then splits them into
/// head and body, assuming productions in the form `head:body;`
/// # Arguments
/// * `productions` - A vector of String where each String is a production ending with `;`. This
/// vector may contain fragments, but in this case the fragment keyword must be passed as well.
/// # Returns
/// A GrammarInternal containing the built grammar, with recursive lexing rules and fragments yet to
/// be removed.
/// # Errors
/// SyntaxError if fragments does not start with uppercase letter
fn split_head_body(productions: Vec<String>) -> Result<GrammarInternal, ParseError> {
    let mut terminals = HashMap::new();
    let mut non_terminals = HashMap::new();
    let mut fragments = HashMap::new();
    let actions = HashMap::new();
    let mut order = Vec::new();
    //the capt.group of this regex will be passed to p_re so I need to include ;
    let f_re = r"\s*fragment\s+((?:.|\n)+;)";
    let p_re = r"\s*(\w+)\s*:\s*((?:.|\n)+);";
    let re_fr = Regex::new(f_re).unwrap(); //fragment detection
    let re_pd = Regex::new(p_re).unwrap(); //production detection
    for production in productions.iter() {
        let mut is_fragment = false;
        let mut prod = &production[..];
        if let Some(matches) = re_fr.captures(prod) {
            prod = matches.get(1).map_or("", |m| m.as_str());
            is_fragment = true;
        }
        match re_pd.captures(prod) {
            Some(matches) => {
                let name = matches.get(1).map_or("", |m| m.as_str()).to_string();
                let rule = matches.get(2).map_or("", |m| m.as_str()).to_string();
                if name.chars().next().unwrap().is_lowercase() {
                    if !is_fragment {
                        non_terminals.insert(name.to_owned(), rule);
                        order.push(name);
                    } else {
                        return Err(ParseError::SyntaxError {
                            message: format!("Fragments should be uppercase: {}", production),
                        });
                    }
                } else if !is_fragment {
                    terminals.insert(name.to_owned(), rule);
                    order.push(name);
                } else {
                    //no position recorded for fragments
                    fragments.insert(name, rule);
                }
            }
            None => {
                return Err(ParseError::SyntaxError {
                    message: format!("Unknown production: {}", prod),
                });
            }
        }
    }
    Ok(GrammarInternal {
        terminals,
        actions,
        non_terminals,
        fragments,
        order,
    })
}

/// Removes the fragments from lexer rules by effectively replacing them with their production body.
/// Additionally, solves the recursion in Lexer rules (this should not be a thing at all but it's
/// allowed by ANTLR...).
/// # Arguments
/// * `grammar` - A Grammar containing a set of terminals and non-terminals.
/// * `fragments` - An HashMap containing the fragments. The key is the head of the production and
/// the value is the body.
/// # Returns
/// A GrammarInternal object where each terminals does NOT contain recursive rules.
/// # Errors
/// SyntaxError if the terminals have cyclic dependencies or calls non-terminals productions.
fn resolve_terminals_dependencies(grammar: GrammarInternal) -> Result<GrammarInternal, ParseError> {
    let terms = merge_terminals_fragments(&grammar);
    let graph = build_terminals_dag(&terms, &grammar.non_terminals)?;
    let new_terminals = replace_terminals(&terms, &graph[0], &graph[1])?;
    Ok(GrammarInternal {
        terminals: new_terminals.0,
        actions: new_terminals.1,
        non_terminals: grammar.non_terminals,
        fragments: grammar.fragments,
        order: grammar.order,
    })
}

/// Helper struct used to contain an hash map and unique ids referencing to each key-value pair.
struct TerminalsFragmentsHelper<'a> {
    /// An hash map.
    prods: HashMap<&'a str, &'a str>,
    /// Maps each key of the map to an unique, sequential number.
    head2id: HashMap<&'a str, usize>,
    /// Maps each unique sequential number to a key of the map.
    id2head: Vec<&'a str>,
}

/// Merge together the fragments and the terminals in a single map and assigns a temporary index to
/// each production (this index will be used for dependency graph and is independent of the
/// production order index).
/// # Arguments
/// * `grammar` - The grammar from where terminals and fragments will be extracted.
/// # Returns
/// The generated merged struct.
fn merge_terminals_fragments(grammar: &GrammarInternal) -> TerminalsFragmentsHelper {
    let fragments_iter = grammar.fragments.iter().map(|(k, v)| (&k[..], &v[..]));
    let terminals_iter = grammar.terminals.iter().map(|(k, v)| (&k[..], &v[..]));
    let merge = fragments_iter
        .chain(terminals_iter)
        .collect::<HashMap<_, _>>();
    let mut head2id = HashMap::new();
    let mut id2head = vec![""; merge.len()];
    for (idx, terminal) in merge.iter().enumerate() {
        id2head[idx] = *terminal.0;
        head2id.insert(*terminal.0, idx);
    }
    TerminalsFragmentsHelper {
        prods: merge,
        head2id,
        id2head,
    }
}

/// Builds the dependency graph of each terminal rule and fragment rule in relation with the others.
/// # Arguments
/// * `terms` - The helper struct created with the `merge_terminals_fragments()` function.
/// * `nonterms` - An HashMap containing the key value pair for every non terminal production.
/// This map is used for error checking.
/// # Returns
/// Two arrays. Each index in the array correspond to a node ID (IDs can be found inside `term`) and
/// contains a set. The set for the two arrays contains:
/// * `[1]` - The dependencies in form of adjacency list: if a node A references productions B and
/// C, its adjacency list will contain the index of B and C.
/// * `[2] - For each node, the position in the body production (in bytes) where a recursive word
/// starts and ends. For example the body `'_' | DIGIT` will contain the indices where the word
/// `DIGIT` starts and ends. This will be useful to remove this word and replace it with the actual
/// production.
/// # Errors
/// SyntaxError if terminal rules refer non-terminal rules.
fn build_terminals_dag(
    terms: &TerminalsFragmentsHelper,
    nonterms: &HashMap<String, String>,
) -> Result<[Vec<BTreeSet<usize>>; 2], ParseError> {
    let mut graph = vec![BTreeSet::<usize>::new(); terms.prods.len()];
    //where each "recursive token" starts and ends
    let mut split = vec![BTreeSet::<usize>::new(); terms.prods.len()];
    let re = Regex::new(r"\w+").unwrap();
    for terminal in terms.prods.iter() {
        let term_head = *terminal.0;
        let term_body = *terminal.1;
        let term_id = *terms.head2id.get(term_head).unwrap(); //this DEFINITELY exists
        split[term_id].insert(0);
        split[term_id].insert((*terminal.1).len());
        for mat in re.find_iter(term_body) {
            // the terminal referenced (dependency to be satisfied)
            let dep = &term_body[mat.start()..mat.end()];
            //this is NOT guaranteed to exist! the regex can match literals!
            match terms.head2id.get(dep) {
                Some(dep_id) => {
                    //build the graph
                    graph[term_id].insert(*dep_id);
                    //and record the position in the string of the terminal
                    split[term_id].insert(mat.start());
                    split[term_id].insert(mat.end());
                }
                None => {
                    if nonterms.contains_key(dep) {
                        //more specific error in case a non-term is referenced
                        return Err(ParseError::SyntaxError {
                            message: format!(
                                "Lexer rule {} cannot reference Parser non-terminal {}",
                                term_head, dep
                            ),
                        });
                    }
                    //If I arrive here it is a terminal literal so skip it.
                }
            }
        }
    }
    Ok([graph, split])
}

/// Inner type used only in `replace_terminals()`. Check that function doc for a description.
type TerminalsActionsMap = (HashMap<String, String>, HashMap<String, BTreeSet<Action>>);

/// Given the results of `build_terminals_dag()`, this function actually replaces the terminals with
/// the actual production, in the correct order.
/// # Arguments
/// * `terms` - The helper struct created with the `merge_terminals_fragments()` function.
/// * `graph` - The dependencies graph created with the function `build_terminals_dag()`.
/// * `split` - The recursive rule positions obtained with the function `build_terminals_dag()`.
/// # Returns
/// Two HashMaps:
/// * the first containing every terminal rule (including the fragments) with no dependencies on
/// other terminal rules.
/// * the second containing every action for every terminal name
/// # Errors
/// SyntaxError if the productions form cycles or the lexer rules are illegal
fn replace_terminals(
    terms: &TerminalsFragmentsHelper,
    graph: &[BTreeSet<usize>],
    split: &[BTreeSet<usize>],
) -> Result<TerminalsActionsMap, ParseError> {
    let mut new_terminals = HashMap::<String, String>::new();
    let mut new_actions = HashMap::<String, BTreeSet<Action>>::new();
    if let Some(order) = topological_sort(graph) {
        for node in order {
            let mut body = *terms.prods.get(&terms.id2head[node]).unwrap();
            let replaced_body;
            if !graph[node].is_empty() {
                let mut last_split = 0_usize;
                //
                replaced_body = split[node]
                    .iter()
                    .map(|idx| {
                        let mut ret = String::with_capacity(*idx - last_split + 2);
                        //get the slice
                        let cur_slice = &body[last_split..*idx];
                        last_split = *idx;
                        //if is a head replace it with body
                        match new_terminals.get(cur_slice) {
                            Some(prod) => {
                                ret.push('(');
                                ret.push_str(prod);
                                ret.push(')');
                            }
                            None => {
                                ret.push_str(cur_slice);
                            }
                        };
                        ret
                    })
                    .collect::<Vec<_>>()
                    .join("");
                body = &replaced_body;
            }
            let clean_body = clean_ws(body);
            let action = get_actions(&clean_body)?;
            new_terminals.insert(terms.id2head[node].to_owned(), action.0.to_owned());
            new_actions.insert(terms.id2head[node].to_owned(), action.1);
        }
        Ok((new_terminals, new_actions))
    } else {
        Err(ParseError::SyntaxError {
            message: "Lexer contains cyclic productions!".to_owned(),
        })
    }
}

/// Given the body of a terminal production, splits it into actual body and lexer rule.
/// # Arguments
/// * `body` - The body of a terminal production.
/// # Returns
/// A tuple containing
/// * The body without the action. For example `a -> skip` will return `a`.
/// * A Set containing every Action for the current production. For example `a -> skip, popMode`
/// will return a set containing `Action::SKIP` and `Action::POPMODE`
/// # Errors
/// SyntaxError if the action is illegal
fn get_actions(body: &str) -> Result<(&str, BTreeSet<Action>), ParseError> {
    let mut action_builder = String::with_capacity(body.len());
    let mut body_iter = body.chars().rev().peekable();
    while let Some(letter) = body_iter.next() {
        action_builder.push(letter);
        match letter {
            '\'' => append_until(&mut body_iter, &mut action_builder, '\''),
            ']' => append_until(&mut body_iter, &mut action_builder, '['),
            '>' => {
                if let Some(next) = body_iter.peek() {
                    if *next == '-' {
                        //this is the boundary between lexer rules and actions
                        action_builder.push(*next);
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    //at this point action_builder either contains `->action` (reversed) or the entire string
    if action_builder.len() == body.len() {
        //entire string -> no action found
        Ok((body, BTreeSet::new()))
    } else {
        //action(s) will be converted to enum
        let clean_body = &body[..body.len() - action_builder.len()];
        action_builder = action_builder.chars().rev().collect(); //gather correct order
        let action_str = &action_builder[2..];
        let mut actions = BTreeSet::new();
        let action_split: Vec<&str> = action_str.split(',').collect();
        for act in action_split {
            actions.insert(match_action(act)?);
        }
        Ok((clean_body, actions))
    }
}

/// Match a specific action into an enum. Just a wrapper to split longer functions.
/// # Arguments
/// * `act` - The text representing a SINGLE action
/// # Returns
/// The action expressed as the Action enum
/// # Errors
/// SyntaxError if the action is illegal
fn match_action(act: &str) -> Result<Action, ParseError> {
    match act {
        "skip" => Ok(Action::SKIP),
        "more" => Ok(Action::MORE),
        "popMode" => Ok(Action::POPMODE),
        _ => {
            let last_is_par = match act.chars().last() {
                Some(char) => char == ')',
                None => false,
            };
            if act.starts_with("type(") && last_is_par {
                let arg = &act[5..act.len() - 1];
                Ok(Action::TYPE(arg.to_owned()))
            } else if act.starts_with("channel(") && last_is_par {
                let arg = &act[8..act.len() - 1];
                Ok(Action::CHANNEL(arg.to_owned()))
            } else if act.starts_with("mode(") && last_is_par {
                let arg = &act[5..act.len() - 1];
                Ok(Action::MODE(arg.to_owned()))
            } else if act.starts_with("pushMode(") && last_is_par {
                let arg = &act[9..act.len() - 1];
                Ok(Action::PUSHMODE(arg.to_owned()))
            } else {
                let message = format!("invalid action `{}`", act);
                Err(ParseError::SyntaxError { message })
            }
        }
    }
}

/// Performs a topological sort using an iterative DFS.
/// # Arguments
/// * `graph` - A graph represented as adjacency list.
/// # Returns
/// * `Some(value)` - a Vec containing the ordered indices if graph was a DAG.
/// * `None` - if the graph was not acyclic.
fn topological_sort(graph: &[BTreeSet<usize>]) -> Option<Vec<usize>> {
    //The idea is is the one described by Cormen et al. (2001), Mark record
    //if the DFS can reach node of the current branch and thus there is a cycle
    //In addition, being this function iterative, the `toprocess` array is used
    //to defer the node into post-order.
    #[derive(Clone, PartialEq)]
    enum Mark {
        NONE,
        //Node untouched
        TEMPORARY,
        //Current node being processed
        PERMANENT, //All the children of this node has been processed
    };
    let mut visited = vec![Mark::NONE; graph.len()];
    //Pair (node, All my neighbours have already been processed)
    let mut toprocess = Vec::with_capacity(graph.len());
    let mut ordered = Vec::with_capacity(graph.len());
    for n in 0..graph.len() {
        if visited[n] == Mark::NONE {
            toprocess.push((n, false)); //mark as not visited
        }
        while !(toprocess.is_empty()) {
            let node = toprocess.pop().unwrap();
            if node.1 {
                visited[node.0] = Mark::PERMANENT;
                ordered.push(node.0);
            } else {
                visited[node.0] = Mark::TEMPORARY;
                toprocess.push((node.0, true));
                let neighbours = &graph[node.0];
                for neighbour in neighbours {
                    if visited[*neighbour] == Mark::NONE {
                        toprocess.push((*neighbour, false));
                    } else if visited[*neighbour] == Mark::TEMPORARY {
                        return None;
                    } else {
                        //node already visited but does not form a cycle
                    }
                }
            }
        }
    }
    Some(ordered)
}

/// Retrieves every production from a `.g4` grammar.
/// This effectively works by removing every comment and then splitting over ; tokens that are not
/// quoted, although in this functions is implemented as a single pass.
/// The comments removed are the multi-line `/*`-`*/` and single line `//`, `#`.
/// # Arguments
/// * `content` - A string containing the original `.g4` grammar content.
/// * `filename` - The name of the original file parsed.
/// # Returns
/// A vector of string representing the productions of the original grammar.
/// Each element represents a single production.
fn retrieve_productions(content: &str) -> Vec<String> {
    let mut productions = Vec::new();
    let mut ret = String::new();
    let mut it = content.chars().peekable();
    while let Some(letter) = it.next() {
        //get lookahead
        let lookahead = match it.peek() {
            Some(l) => *l,
            None => '\0',
        };
        match letter {
            '/' => {
                if lookahead == '*' {
                    it.next(); //position to the lookahead
                               //skip until */ is found
                    while let Some(skip) = it.next() {
                        if skip == '*' {
                            if let Some(lahead) = it.peek() {
                                if *lahead == '/' {
                                    it.next(); //skip also the lookahead
                                    break;
                                }
                            }
                        }
                    }
                } else if lookahead == '/' {
                    consume_until(&mut it, '\n', false);
                } else {
                    ret.push(letter);
                }
            }
            '\'' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, '\'')
            }
            '[' => {
                ret.push(letter);
                append_until(&mut it, &mut ret, ']');
            }
            ';' => {
                ret.push(letter);
                productions.push(ret);
                ret = String::new();
            }
            _ => ret.push(letter),
        }
    }
    //remove the grammar XX; stmt.
    productions
        .into_iter()
        .filter(|s| !s.starts_with("grammar ") && s.contains(':'))
        .collect()
}

/// Removes whitespaces and embedded actions from a production in a context-aware fashion.
/// For example the production `[ ]+ | a -> channel (NAME) { embedded };` becomes
/// `[ ]+|a->channel(NAME)`
/// # Arguments
/// * `terminal` - The body of a terminal production.
/// # Returns
/// The body without spaces, excluding the one inside [] or '' blocks and without embedded rules
fn clean_ws(terminal: &str) -> String {
    let mut new = String::with_capacity(terminal.len());
    let mut it = terminal.chars().peekable();
    while let Some(letter) = it.next() {
        if !letter.is_whitespace() {
            new.push(letter);
            match letter {
                '\'' => append_until(&mut it, &mut new, '\''),
                '[' => append_until(&mut it, &mut new, ']'),
                '{' => {
                    new.pop();
                    consume_until(&mut it, '}', false);
                    //remove semantic predicate as well
                    if let Some(next) = it.peek() {
                        if *next == '?' {
                            it.next();
                        }
                    }
                }
                _ => {}
            }
        }
    }
    new
}

/// Advances the iterator until the given character.
/// This function can take into account also escape characters, so if the given character is ', this
/// won't  stop in case a \' is encountered.
/// # Arguments
/// * `it` - The iterator that will be advanced.
/// * `until` - The character that will stop the method. Escaped versions of this character won't be
/// considered.
/// * `allow_escapes` - true if escaped character won't stop the function, false otherwise
fn consume_until<T>(it: &mut T, until: char, allow_escapes: bool)
where
    T: Iterator<Item = char>,
{
    let mut escapes = 0;
    for skip in it {
        if skip == until {
            if !allow_escapes || escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if skip == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
    }
}

/// Advances the iterator until the given character and appends all the encountered characters.
/// This function takes into account also escape character, so if the given character is ', this
/// won't  stop in case a \' is encountered.
/// # Arguments
/// * `it` - The iterator that will be advanced.
/// * `ret` - The string where the various character will be appended.
/// * `until` - The character that will stop the method. Escaped versions of this character won't be
/// considered.
fn append_until<T>(it: &mut T, ret: &mut String, until: char)
where
    T: Iterator<Item = char>,
{
    let mut escapes = 0;
    for push in it {
        ret.push(push);
        if push == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if push == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
    }
}

#[cfg(test)]
#[path = "tests/grammar.rs"]
mod tests;
