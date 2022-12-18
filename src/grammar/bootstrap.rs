use crate::error::ParseError;
use crate::grammar::{Action, Grammar, LexerRuleElement, LexerOp, Tree};
use maplit::{hashmap, btreeset};
use regex::Regex;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::iter::Peekable;
use std::str::Chars;

use super::{LexerProduction, ParserProduction};

const SINGLE_QUOTE_AS_U8: u8 = b'\'';

/// Parse a .g4 grammar with a manually written, ad-hoc parser.
pub(super) fn bootstrap_parse_string(content: &str) -> Result<Grammar, ParseError> {
    let productions = retrieve_productions(content);
    let grammar_rec = split_head_body(productions)?;
    let grammar_no_lit = extract_literals_from_non_term(grammar_rec);
    let grammar_not_rec = resolve_terminals_dependencies(grammar_no_lit)?;
    let grammar = reindex(grammar_not_rec);
    Ok(grammar)
}

/// Struct used internally to represent a Grammar during construction.
/// Hash maps are less space efficient but easier to use during constructions.
/// At some point I also drop the fragments map, but I prefer to pass them around using a single
/// Struct.
struct GrammarInternal {
    ///`key`:`value`; map containing terminals.
    terminals: HashMap<String, String>,
    ///`key`:`value`; map containing non-terminals.
    non_terminals: HashMap<String, String>,
    ///`key`:`value`; map containing fragments.
    fragments: HashMap<String, String>,
    ///array containing every production `key` in the order they appear in the file.
    order: VecDeque<String>,
    //`key`:`value`; map containing terminal actions
    actions: HashMap<String, BTreeSet<Action>>,
    // key: mode name, value: mode index
    modes: HashMap<String, u32>,
}

/// Transforms the Unordered maps of the GrammarInternal into the indexed Vec of the Grammar.
///
/// GrammarInternal uses an Hash-based indexing because it's more convenient for productions removal
/// (i.e. fragments). Grammar instead, uses a more efficient numerical indexing. Given that the
/// index matters, especially for the lexer, this method transform a GrammarInternal into a Grammar
/// while keeping the original indexing.
fn reindex(mut grammar: GrammarInternal) -> Grammar {
    let mut terminals = Vec::with_capacity(grammar.terminals.len());
    let mut non_terminals = Vec::with_capacity(grammar.non_terminals.len());
    //the new order will be: first every terminal, then every non-terminal. In the original order.
    for head in grammar.order {
        if let Some(body) = grammar.terminals.remove(&head) {
            let actions = grammar.actions.remove(&head).unwrap(); // this should always exist
            let body = expand_literals(gen_precedence_tree(&body));
            terminals.push(LexerProduction {
                head,
                body,
                actions,
            });
        } else if let Some(body) = grammar.non_terminals.remove(&head) {
            non_terminals.push(ParserProduction { head, body });
        } else {
            panic!("Expected production to be either in terminals or non terminals.");
        }
    }
    Grammar {
        terminals: vec![terminals],
        non_terminals,
        modes_index: grammar.modes,
    }
}

/// Removes all literals from non terminals inside a grammar.
///
/// Given a grammar with some non-terminal productions, this method removes all literal from inside
/// the non-terminals. Literals must be enclosed within single quotes and, per ANTLR specification,
/// cannot contain escaped quotes (or at least this is my assumption after reading the
/// specification). The various literals are replaced with the corresponding terminal production, if
/// one exists, otherwise a new one is created with highest priority.
fn extract_literals_from_non_term(grammar: GrammarInternal) -> GrammarInternal {
    let mut new_terminals = HashMap::new();
    let mut terminals_rev = grammar
        .terminals
        .iter()
        .map(|x| (x.1[..].to_string(), x.0.clone()))
        .collect::<HashMap<_, _>>();
    let mut order = grammar.order;
    let mut non_terminals = HashMap::new();
    let mut literal_idx = 0;
    for nonterm in grammar.non_terminals.into_iter() {
        let key = nonterm.0;
        let value = nonterm.1;
        let mut replaced = String::new();
        let nonterm_as_bytes = value.as_bytes();
        let indices_of_quotes = nonterm_as_bytes
            .iter()
            .enumerate()
            .filter(|x| *x.1 == SINGLE_QUOTE_AS_U8)
            .map(|c| c.0)
            .chain(std::iter::once(0))
            .collect::<BTreeSet<_>>();
        let mut is_literal = nonterm_as_bytes[0] == SINGLE_QUOTE_AS_U8;
        let mut indices_iterator = indices_of_quotes.into_iter().peekable();
        // in this loop every substring (between the quotes) is iterated and checked
        while let Some(mut index0) = indices_iterator.next() {
            let index1 = *indices_iterator.peek().unwrap_or(&value.len());
            if nonterm_as_bytes[index0] == SINGLE_QUOTE_AS_U8 && index1 != value.len() {
                // without index1 != value.len() there would be errors in case index0 is the last
                // letter and a quote and peeking index1 results in value.len()
                index0 += 1;
            }
            if is_literal {
                let quoted_str = &value[index0 - 1..index1 + 1];
                if let Some(prod_name) = terminals_rev.get(quoted_str) {
                    replaced.push_str(prod_name);
                } else {
                    let prod_name = format!("#LITERAL{}", literal_idx);
                    literal_idx += 1;
                    replaced.push_str(&prod_name[..]);
                    new_terminals.insert(prod_name.clone(), quoted_str.to_string());
                    terminals_rev.insert(quoted_str.to_string(), prod_name.clone());
                    order.push_front(prod_name);
                }
            } else {
                let str = &value[index0..index1];
                replaced.push_str(str);
            }
            is_literal = !is_literal;
        }
        non_terminals.insert(key, replaced);
    }
    GrammarInternal {
        terminals: grammar.terminals.into_iter().chain(new_terminals).collect(),
        non_terminals,
        fragments: grammar.fragments,
        order,
        actions: grammar.actions,
        modes: grammar.modes,
    }
}

/// Categorizes various productions into terminal, non-terminal and fragments. Then splits them into
/// head and body, assuming productions in the form `head:body;`
///
/// Takes as input a vector of String where each String is a production ending with `;`. This
/// vector may contain fragments, but in this case the fragment keyword must be passed as well.
///
/// Returns SyntaxError if fragments does not start with uppercase letter
fn split_head_body(productions: Vec<String>) -> Result<GrammarInternal, ParseError> {
    let mut terminals = HashMap::new();
    let mut non_terminals = HashMap::new();
    let mut fragments = HashMap::new();
    let actions = HashMap::new();
    let modes = hashmap! {"DEFAULT_MODE".to_string() => 0};
    let mut order = VecDeque::new();
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
                        order.push_back(name);
                    } else {
                        return Err(ParseError::SyntaxError {
                            message: format!("Fragments should be uppercase: {}", production),
                        });
                    }
                } else if !is_fragment {
                    terminals.insert(name.to_owned(), rule);
                    order.push_back(name);
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
        modes,
    })
}

/// Removes the fragments from lexer rules by effectively replacing them with their production body.
///
/// Additionally, solves the recursion in Lexer rules (this should not be a thing at all but it's
/// allowed by ANTLR...).
///
/// Returns SyntaxError if terminals have cyclic dependencies or calls non-terminals productions.
fn resolve_terminals_dependencies(grammar: GrammarInternal) -> Result<GrammarInternal, ParseError> {
    let terms = merge_terminals_fragments(&grammar);
    let graph = build_terminals_dag(&terms, &grammar.non_terminals)?;
    let mut new_modes = grammar.modes.clone();
    let new_terminals = replace_terminals(&terms, &graph[0], &graph[1], &mut new_modes)?;
    Ok(GrammarInternal {
        terminals: new_terminals.0,
        actions: new_terminals.1,
        non_terminals: grammar.non_terminals,
        fragments: grammar.fragments,
        order: grammar.order,
        modes: new_modes,
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
///
/// The requested inputs are:
/// * `terms` - The helper struct created with the `merge_terminals_fragments()` function.
/// * `nonterms` - An HashMap containing the key value pair for every non terminal production.
/// This map is used for error checking.
///
/// This method returns two arrays. Each index in the array correspond to a node ID
/// (IDs can be found inside `term`) and contains a set. The set for the two arrays contains:
/// * `[1]` - The dependencies in form of adjacency list: if a node A references productions B and
/// C, its adjacency list will contain the index of B and C.
/// * `[2] - For each node, the position in the body production (in bytes) where a recursive word
/// starts and ends. For example the body `'_' | DIGIT` will contain the indices where the word
/// `DIGIT` starts and ends. This will be useful to remove this word and replace it with the actual
/// production.
///
/// Instead, a SyntaxError is returned if terminal rules refer non-terminal rules.
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
///
/// The input parametrs are:
/// * `terms` - The helper struct created with the `merge_terminals_fragments()` function.
/// * `graph` - The dependencies graph created with the function `build_terminals_dag()`.
/// * `split` - The recursive rule positions obtained with the function `build_terminals_dag()`.
///
/// The function returns two HashMaps:
/// * the first containing every terminal rule (including the fragments) with no dependencies on
/// other terminal rules.
/// * the second containing every action for every terminal name
///
/// A SyntaxError is returned if the productions form cycles or the lexer rules are illegal
fn replace_terminals(
    terms: &TerminalsFragmentsHelper,
    graph: &[BTreeSet<usize>],
    split: &[BTreeSet<usize>],
    modes: &mut HashMap<String, u32>,
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
            let action = get_actions(&clean_body, modes)?;
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
///
/// Return a tuple containing:
/// * The body without the action. For example `a -> skip` will return `a`.
/// * A Set containing every Action for the current production. For example `a -> skip, popMode`
/// will return a set containing `Action::SKIP` and `Action::POPMODE`
///
/// A SyntaxError is returned if the action is illegal.
fn get_actions<'a>(
    body: &'a str,
    modes: &mut HashMap<String, u32>,
) -> Result<(&'a str, BTreeSet<Action>), ParseError> {
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
            actions.insert(match_action(act, modes)?);
        }
        Ok((clean_body, actions))
    }
}

/// Matches a specific action into an enum. Just a wrapper to split longer functions.
///
/// Takes as input the text representing a SINGLE action and returns the action expressed with the
/// Action enum.
///
/// Returns SyntaxError if the action is illegal
fn match_action(act: &str, modes: &mut HashMap<String, u32>) -> Result<Action, ParseError> {
    match act {
        "skip" => Ok(Action::Skip),
        "more" => Ok(Action::More),
        "popMode" => Ok(Action::PopMode),
        _ => {
            let last_is_par = match act.chars().last() {
                Some(char) => char == ')',
                None => false,
            };
            let next_entry = modes.len() as u32;
            if act.starts_with("mode(") && last_is_par {
                let arg = &act[5..act.len() - 1];
                let index = *modes.entry(arg.to_string()).or_insert(next_entry);
                Ok(Action::Mode(index))
            } else if act.starts_with("pushMode(") && last_is_par {
                let arg = &act[9..act.len() - 1];
                let index = *modes.entry(arg.to_string()).or_insert(next_entry);
                Ok(Action::PushMode(index))
            } else if act.starts_with("channel(") && last_is_par {
                Ok(Action::Channel)
            } else if act.starts_with("type(") && last_is_par {
                Ok(Action::Type)
            } else {
                let message = format!("invalid action `{}`", act);
                Err(ParseError::SyntaxError { message })
            }
        }
    }
}

/// Performs a topological sort using an iterative DFS.
///
/// Takes as input a graph represented as adjacency list.
///
/// Returns a Vec containing the ordered indices if graph was a DAG or None is the graph contained
/// cycles.
fn topological_sort(graph: &[BTreeSet<usize>]) -> Option<Vec<usize>> {
    // The idea is is the one described by Cormen et al. (2001), Mark records
    // if the DFS can reach node of the current branch and thus there is a cycle
    // In addition, being this function iterative, the `toprocess` array is used
    // to defer the node into post-order.
    #[derive(Clone, PartialEq)]
    #[allow(clippy::upper_case_acronyms)]
    enum Mark {
        NONE,
        //Node untouched
        TEMPORARY,
        //Current node being processed
        PERMANENT, //All the children of this node has been processed
    }
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
///
/// This effectively works by removing every comment and then splitting over ; tokens that are not
/// quoted, although in this functions is implemented as a single pass.
/// The comments removed are the multi-line `/*`-`*/` and single line `//`, `#`.
///
/// Each element in the returned vector represents a single production.
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
///
/// The spaces not removed are the one inside [] or '' blocks and without embedded rules.
///
/// For example the production `[ ]+ | a -> channel (NAME) { embedded };` becomes
/// `[ ]+|a->channel(NAME)`
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
///
/// This function can take into account also escaped characters, so if the given character is ',
/// this won't  stop in case a \' is encountered.
///
/// Arguments:
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
///
/// This function takes into account also escape character, so if the given character is ', this
/// won't  stop in case a \' is encountered.
///
/// Arguments:
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

////////////// HANDLING LEXER PRODUCTIONS //////////

///Operators for a regex (and an operand, ID).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub(super) enum ProdToken<'a> {
    /// Kleenee star `*`.
    KLEENE,
    /// Question mark `?`.
    QM,
    /// Plus sign `+`.
    PL,
    /// Left parenthesis `(`.
    LP,
    /// Right parenthesis `)`.
    RP,
    /// Not `~`.
    NOT,
    /// Alternation of two operands.
    OR,
    /// Concatenation of two operands.
    AND,
    /// An Operand.
    Id(&'a str),
}

impl ProdToken<'_> {
    /// Returns the number of operands required for each operator.
    fn required_operands(&self) -> u8 {
        match self {
            ProdToken::LP | ProdToken::RP | ProdToken::Id(_) => 0,
            ProdToken::KLEENE | ProdToken::QM | ProdToken::PL | ProdToken::NOT => 1,
            ProdToken::OR | ProdToken::AND => 2,
        }
    }

    // returns true if the operator is instead an ID (simplifies syntax)
    fn is_id(&self) -> bool {
        matches!(self, ProdToken::Id(_))
    }

    /// Returns the operator precedence priority of this operator.
    fn priority(&self) -> u8 {
        match self {
            ProdToken::LP => 5,
            ProdToken::RP => 5,
            ProdToken::NOT => 4,
            ProdToken::KLEENE => 3,
            ProdToken::QM => 3,
            ProdToken::PL => 3,
            ProdToken::AND => 2,
            ProdToken::OR => 1,
            ProdToken::Id(_) => 0,
        }
    }
}

impl std::fmt::Display for ProdToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProdToken::KLEENE => write!(f, "*"),
            ProdToken::QM => write!(f, "?"),
            ProdToken::PL => write!(f, "+"),
            ProdToken::LP => write!(f, "("),
            ProdToken::RP => write!(f, ")"),
            ProdToken::NOT => write!(f, "~"),
            ProdToken::OR => write!(f, "|"),
            ProdToken::AND => write!(f, "&"),
            ProdToken::Id(i) => write!(f, "{}", i),
        }
    }
}

/// Parse tree for the regex operands, accounting for precedence.
type PrecedenceTree<'a> = Tree<ProdToken<'a>>;

/// Creates a parse tree with correct precedence given the input regex.
///
/// This is essentially the shunting yard algorithm.
/// All non-unary operators are left associative.
fn gen_precedence_tree(regex: &str) -> PrecedenceTree {
    let mut operands = Vec::new();
    let mut operators: Vec<ProdToken> = Vec::new();
    // first get a sequence of operands and operators
    let mut tokens = regex_to_operands(regex).into_iter().peekable();
    // for each in the sequence do the following actions
    while let Some(operator) = tokens.next() {
        match operator {
            //operators: solve if precedent has higher priority and is not OpType::LP then push cur
            ProdToken::NOT
            | ProdToken::OR
            | ProdToken::AND
            | ProdToken::KLEENE
            | ProdToken::QM
            | ProdToken::PL => {
                while !operators.is_empty()
                    && operators.last().unwrap() != &ProdToken::LP
                    && operators.last().unwrap().priority() >= operator.priority()
                {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.push(operator);
            }
            // left parenthesis: push. Will be resolved by a right parenthesis.
            ProdToken::LP => operators.push(operator),
            // right parenthesis: combine all the nodes until left parenthesis is found.
            ProdToken::RP => {
                // corner case: avoid *)? ?)? +)? to become *? ?? +?, these are different ops
                if let Some(next) = tokens.peek() {
                    if let Some(top) = operators.last() {
                        if let Some(update) = avoid_unwanted_nongreedy(*top, *next) {
                            *operators.last_mut().unwrap() = update;
                            tokens.next(); // skip the next ?
                        }
                    }
                }
                // normal case
                while !operators.is_empty() && operators.last().unwrap() != &ProdToken::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
            // id: push to stack
            ProdToken::Id(_) => {
                let leaf = Tree::new_leaf(operator);
                operands.push(leaf);
            }
        }
    }
    //solve all remaining operators
    while !operators.is_empty() {
        combine_nodes(&mut operands, &mut operators);
    }
    operands.pop().unwrap()
}

/// Complementary to the function gen_precedence_tree, combines two nodes with the last operator
/// in the stack.
fn combine_nodes<'a>(operands: &mut Vec<PrecedenceTree<'a>>, operators: &mut Vec<ProdToken<'a>>) {
    let operator = operators.pop().unwrap();
    let children = if operator.required_operands() == 2 {
        //binary operator
        let right = operands.pop().unwrap();
        let left = operands.pop().unwrap();
        vec![left, right]
    } else {
        // unary operator
        debug_assert!(operator.required_operands() == 1);
        vec![operands.pop().unwrap()]
    };
    let node = Tree::new_node(operator, children);
    operands.push(node);
}

/// Used in the gen_precedence_tree function.
/// If last is */?/+ next is ? and current is ), the parenthesis removal will introduce an unwanted
/// non-greedy operation. This function simplifies the `last` and `next` operators so the meaning
/// is the intended one
fn avoid_unwanted_nongreedy<'a, 'b>(
    last: ProdToken<'a>,
    next: ProdToken<'b>,
) -> Option<ProdToken<'a>> {
    if next == ProdToken::QM {
        match last {
            ProdToken::KLEENE => Some(last),
            ProdToken::QM => Some(last),
            ProdToken::PL => Some(ProdToken::KLEENE),
            _ => None,
        }
    } else {
        None
    }
}

/// Transforms a precedence parse tree in a precedence parse tree where the groups like `[a-z]`
/// are expanded in `a | b | c ... | y | z`
fn expand_literals(node: PrecedenceTree) -> Tree<LexerRuleElement<char>> {
    match node.value {
        ProdToken::Id(lit) => expand_literal_node(lit),
        n => {
            let children = node.into_children().map(expand_literals).collect();
            let operation = match n {
                ProdToken::KLEENE => LexerOp::Kleene,
                ProdToken::QM => LexerOp::Qm,
                ProdToken::NOT => LexerOp::Not,
                ProdToken::OR => LexerOp::Or,
                ProdToken::AND => LexerOp::And,
                _ => panic!(), // ( and ) already removed by the precedende function
            };
            Tree::new_node(LexerRuleElement::Operation(operation), children)
        }
    }
}

/// Expands a single node containing sets like `[a-z]` in a set with all the simbols like
/// `{a, b, c, d....}`. Replace also the . symbol with the special placeholder to represent any
/// value and any eventual set with .
fn expand_literal_node(literal: &str) -> Tree<LexerRuleElement<char>> {
    if literal == "." {
        Tree::new_leaf(LexerRuleElement::AnyValue)
    } else {
        let mut charz = Vec::new();
        let mut iter = literal.chars();
        let start = iter.next().unwrap();
        let end;
        let mut last = '\x00';
        let mut is_set = false;
        if start == '[' {
            end = ']';
            is_set = true;
        } else {
            end = '\'';
        };
        // process all character between quotes or braces
        while let Some(char) = iter.next() {
            let mut pushme = char;
            if char == '\\' {
                //escaped char
                pushme = unescape_character(iter.next().unwrap(), &mut iter);
                last = pushme;
                charz.push(pushme);
            } else if start == '[' && char == '-' {
                //set in form a-z, A-Z, 0-9, etc..
                let from = last as u32 + 1;
                let until = iter.next().unwrap() as u32 + 1; //included
                for i in from..until {
                    pushme = std::char::from_u32(i).unwrap();
                    charz.push(pushme);
                }
            } else if char == end {
                //end of sequence
                break;
            } else {
                //normal char
                last = pushme;
                charz.push(pushme);
            }
        }
        //check possible range in form 'a'..'z', at this point I ASSUME this can be a literal only
        //and the syntax has already been checked.
        if let Some(_c @ '.') = iter.next() {
            iter.next();
            iter.next();
            is_set = true;
            let mut until_char = iter.next().unwrap();
            if until_char == '\\' {
                until_char = unescape_character(iter.next().unwrap(), &mut iter);
            }
            let from = last as u32 + 1;
            let until = until_char as u32 + 1;
            for i in from..until {
                charz.push(std::char::from_u32(i).unwrap());
            }
        }
        // set of characters will be transformed into an "or" by the Symbol table later
        if is_set || charz.is_empty() {
            Tree::new_leaf(LexerRuleElement::CharSet(charz.into_iter().collect()))
        } else {
            // concatenation instead must be addressed here
            let mut done = charz
                .into_iter()
                .map(|x| Tree::new_leaf(LexerRuleElement::CharSet(btreeset! {x})))
                .collect::<Vec<_>>();
            while done.len() > 1 {
                let right = done.pop().unwrap();
                let left = done.pop().unwrap();
                let new =
                    Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![left, right]);
                done.push(new);
            }
            done.pop().unwrap()
        }
    }
}

/// Transforms escaped strings in the form "\\n" to the single character they represent '\n'.
/// Works also for unicode in the form "\\UXXXX".
///
/// **NOTE**: it does NOT work for every unicode character (as they can be up to \uXXXXXX) because
/// ANTLR grammars support only up to \u{FFFF}.
fn unescape_character<T: Iterator<Item = char>>(letter: char, iter: &mut T) -> char {
    match letter {
        'n' => '\n',
        'r' => '\r',
        'b' => '\x08',
        't' => '\t',
        'f' => '\x0C',
        'u' | 'U' => {
            //NOTE: ANTLR grammars support only the BMP(0th) plane. Max is \uXXXX
            //      So no escaped emojis :'(
            let digit3 = iter.next().unwrap().to_digit(16).unwrap();
            let digit2 = iter.next().unwrap().to_digit(16).unwrap();
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit3 << 12 | digit2 << 8 | digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        'x' | 'X' => {
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        _ => letter,
    }
}

/// Transforms a regexp in a sequence of operands or operators.
fn regex_to_operands(regex: &str) -> Vec<ProdToken> {
    let mut tokenz = Vec::<ProdToken>::new();
    let mut iter = regex.chars().peekable();
    let mut index = 0;
    while let Some(char) = iter.next() {
        let op = match char {
            '*' => ProdToken::KLEENE,
            '|' => ProdToken::OR,
            '?' => ProdToken::QM,
            '+' => ProdToken::PL,
            '~' => ProdToken::NOT,
            '(' => ProdToken::LP,
            ')' => ProdToken::RP,
            _ => {
                let val = read_token(&regex[index..], char, &mut iter);
                index += val.len() - 1;
                ProdToken::Id(val)
            }
        };
        if !tokenz.is_empty() && implicit_concatenation(tokenz.last().unwrap(), &op) {
            tokenz.push(ProdToken::AND);
        }
        index += 1;
        tokenz.push(op);
    }
    tokenz
}

//No clippy, this is not more readable.
#[allow(clippy::useless_let_if_seq)]
/// Returns the slice of the regexp representing the token as 'a' or 'a'..'b' or '[a-z]'.
fn read_token<'a>(input: &'a str, first: char, it: &mut Peekable<Chars>) -> &'a str {
    match first {
        '.' => &input[..1],
        '[' => {
            let counted = consume_counting_until(it, ']');
            &input[..counted + 2] //2 is to match [], plus all the bytes counted inside
        }
        '\'' => {
            let mut counted = consume_counting_until(it, '\'');
            let mut id = &input[..counted + 2]; //this is a valid literal. check for range ''..''
            if input.len() > counted + 5 && &input[(counted + 2)..(counted + 5)] == "..\'" {
                //this is actually a range ''..'' so first advance the iterator by 3 pos
                it.next();
                it.next();
                it.next();
                //then update the counter for the literal length by accounting also the new lit.
                counted += consume_counting_until(it, '\'');
                id = &input[..counted + 6];
            }
            id
        }
        _ => panic!("Unsupported literal {}", input),
    }
}

/// Given two tokens, `a` and `b` returns true if a concatenation is implied between the two.
/// For example if the tokens are `'a'` and `'b'` the result will be true, because the word `ab`
/// is the concatenation of the two letters.
/// If the tokens are `'a'` and `*` the result will be false as there is no implicit operator.
///
/// Refer to the following table for each rule:
/// - `/` = Not allowed.
/// - `✗` = Concatenation not required.
/// - `✓` = Concatenation required.
/// - Rows: parameter `last`
/// - Columns: parameter `current`
/// Operator `*` is valid also for `?` and `+`.
///
/// |last|`|`|`~`|`*`|`(`|`)`| ID|
///  |---|---|---|---|---|---|---|
///  |`|`|`/`|`/`|`/`|`/`|`/`|`✗`|
///  |`*`|`/`|`✓`|`/`|`✓`|`✗`|`✓`|
///  |`)`|`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
///  |`~`|`/`|`✗`|`/`|`✗`|`/`|`✗`|
///  |`(`|`/`|`✗`|`/`|`✗`|`✗`|`✗`|
///  |ID |`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
fn implicit_concatenation(last: &ProdToken, current: &ProdToken) -> bool {
    let last_is_kleene_family =
        *last == ProdToken::KLEENE || *last == ProdToken::PL || *last == ProdToken::QM;
    let not_lp_id = *current == ProdToken::LP || current.is_id() || *current == ProdToken::NOT;
    (last.is_id() || *last == ProdToken::RP || last_is_kleene_family) && not_lp_id
}

/// Consumes the input (accounting for escaped chars) until the `until` character is found.
/// Returns the number of bytes (u8) consumed.
fn consume_counting_until(it: &mut Peekable<Chars>, until: char) -> usize {
    let mut escapes = 0;
    let mut skipped = 0;
    for skip in it {
        if skip == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if skip == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
        skipped += skip.len_utf8();
    }
    skipped
}

#[cfg(test)]
mod tests {
    use super::topological_sort;
    use crate::error::ParseError;
    use crate::grammar::{Action, Grammar};
    use std::collections::BTreeSet;

    const COMMENT_RICH_GRAMMAR: &str = include_str!("../../resources/comment_rich_grammar.txt");
    const FRAGMENTS_CASE_ERR: &str = include_str!("../../resources/fragments_case_err.txt");
    const FRAGMENTS_CONTAINS_NT: &str = include_str!("../../resources/fragments_contains_nt.txt");
    const FRAGMENTS_GRAMMAR: &str = include_str!("../../resources/fragments_grammar.txt");
    const LEXER_ACTIONS_HARDER: &str = include_str!("../../resources/lexer_actions_harder.txt");
    const LEXER_ACTIONS_SIMPLER: &str = include_str!("../../resources/lexer_actions_simpler.txt");
    const LEXER_CYCLIC: &str = include_str!("../../resources/lexer_cyclic.txt");
    const LEXER_INVALID_ACTION: &str = include_str!("../../resources/lexer_invalid_action.txt");
    const SIMPLE_GRAMMAR: &str = include_str!("../../resources/simple_grammar.txt");

    #[test]
    //Asserts that a DAG return correctly a topological sort
    fn topological_sort_dag() {
        let n0: BTreeSet<usize> = vec![1, 4].into_iter().collect();
        let n1: BTreeSet<usize> = vec![2].into_iter().collect();
        let n2: BTreeSet<usize> = vec![3].into_iter().collect();
        let n3: BTreeSet<usize> = vec![].into_iter().collect();
        let n4: BTreeSet<usize> = vec![3].into_iter().collect();
        let n5: BTreeSet<usize> = vec![].into_iter().collect();

        let graph = vec![n0, n1, n2, n3, n4, n5];
        match topological_sort(&graph) {
            Some(order) => {
                let expected = vec![3, 4, 2, 1, 0, 5];
                assert_eq!(order, expected, "Wrong topological order");
            }
            None => panic!("A DAG should return a topological sort"),
        }
    }

    #[test]
    //Asserts that a graph with cycles cannot have a topological sort
    fn topological_sort_cycles() {
        let n0: BTreeSet<usize> = vec![1, 4].into_iter().collect();
        let n1: BTreeSet<usize> = vec![2].into_iter().collect();
        let n2: BTreeSet<usize> = vec![3].into_iter().collect();
        let n3: BTreeSet<usize> = vec![3].into_iter().collect();
        let n4: BTreeSet<usize> = vec![3].into_iter().collect();
        let n5: BTreeSet<usize> = vec![].into_iter().collect();

        let graph = vec![n0, n1, n2, n3, n4, n5];
        if topological_sort(&graph).is_some() {
            panic!("A graph with cycles should not have a topological order");
        }
    }

    #[test]
    //Asserts the method len() returns the sum of terminal and non terminals
    fn grammar_len() {
        let g = Grammar::new(Vec::new().as_slice(), Vec::new().as_slice());
        assert_eq!(g.len(), 0);
        let terminals = vec![("LETTER_LO", "[a-z]").into(), ("LETTER_UP", "[A-Z]").into()];
        let non_terminals = vec![
            ("letter", "LETTER_UP | LETTER_LO").into(),
            ("word", "word letter | letter").into(),
        ];
        let g = Grammar::new(&terminals, &non_terminals);
        assert_eq!(g.len(), 4);
    }

    #[test]
    //Asserts the method len() returns the sum of terminal and non terminals
    fn grammar_len_term() {
        let g = Grammar::new(Vec::new().as_slice(), Vec::new().as_slice());
        assert_eq!(g.len(), 0);
        let terminals = vec![("LETTER_LO", "[a-z]").into(), ("LETTER_UP", "[A-Z]").into()];
        let non_terminals = vec![
            ("letter", "LETTER_UP | LETTER_LO").into(),
            ("word", "word letter | letter").into(),
        ];
        let g = Grammar::new(&terminals, &non_terminals);
        assert_eq!(g.len_term(), 2);
    }

    #[test]
    //Asserts the method len() returns the sum of terminal and non terminals
    fn grammar_len_nonterm() {
        let g = Grammar::new(Vec::new().as_slice(), Vec::new().as_slice());
        assert_eq!(g.len(), 0);
        let terminals = vec![("LETTER_LO", "[a-z]").into(), ("LETTER_UP", "[A-Z]").into()];
        let non_terminals = vec![("letter", "LETTER_UP | LETTER_LO").into()];
        let g = Grammar::new(&terminals, &non_terminals);
        assert_eq!(g.len_nonterm(), 1);
    }

    #[test]
    //Asserts the method is_empty() works as expected
    fn grammar_is_empty() {
        let g = Grammar::empty();
        assert!(g.is_empty());
        let terminals = vec![("LETTER_LO", "[a-z]").into(), ("LETTER_UP", "[A-Z]").into()];
        let non_terminals = vec![
            ("letter", "LETTER_UP | LETTER_LO").into(),
            ("word", "word letter | letter").into(),
        ];
        let g = Grammar::new(&terminals, &non_terminals);
        assert!(!g.is_empty());
    }

    #[test]
    //Asserts order and production correctness in a hand-crafted grammar
    fn grammar_crafted() {
        let terminals = vec![("LETTER_LO", "[a-z]").into(), ("LETTER_UP", "[A-Z]").into()];
        let non_terminals = vec![
            ("letter", "LETTER_UP | LETTER_LO").into(),
            ("word", "word letter | letter").into(),
        ];
        let g = Grammar::new(&terminals, &non_terminals);
        assert_eq!(
            g.iter_term().collect::<Vec<_>>(),
            terminals.iter().collect::<Vec<_>>(),
        );
        assert_eq!(
            g.iter_nonterm().collect::<Vec<_>>(),
            non_terminals.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    //Asserts that a grammar content can be parsed (without invoking the file)
    fn grammar_parse_string() {
        let g = "grammar g;
    letter: LETTER_UP | LETTER_LO;
    word: word letter | letter;
    LETTER_UP: [A-Z];
    LETTER_LO: [a-z];";
        match Grammar::parse_antlr(g) {
            Ok(g) => {
                assert_eq!(g.len(), 4);
                assert_eq!(g.len_term(), 2);
                assert_eq!(g.len_nonterm(), 2);
            }
            Err(_) => panic!("Failed to parse grammar string"),
        }
    }

    #[test]
    fn extract_literal() {
        let grammar = "LP: '(';\nRP: ')';\nrandom: '(' '(' Џɯɷ幫Ѩ䷘ ')' ')';\nfunction: fn '{' '}';";
        match Grammar::parse_antlr(grammar) {
            Ok(g) => {
                assert_eq!(g.len(), 6);
            }
            Err(_) => panic!("Failed to parse grammar"),
        }
    }

    #[test]
    //Asserts that non-existent files returns error
    fn parse_grammar_non_existent() {
        match Grammar::parse_grammar("./resources/ada_grammar.txt") {
            Ok(_) => panic!("Expected the file to not exist!"),
            Err(e) => assert_eq!(
                e.to_string(),
                "IOError: No such file or directory (os error 2)"
            ),
        }
    }

    #[test]
    //Asserts that the file is parsed correctly even with high number of escape chars
    fn parse_highly_escaped() {
        match Grammar::parse_antlr(COMMENT_RICH_GRAMMAR) {
            Ok(g) => {
                assert_eq!(g.len(), 5);
            }
            Err(e) => panic!("{}", e.to_string()),
        }
    }

    #[test]
    //Asserts that the fragment using non-terminals generates syntax error
    fn parse_fragments_nonterminal() {
        match Grammar::parse_antlr(FRAGMENTS_CONTAINS_NT) {
            Ok(_) => panic!("Expected a syntax error!"),
            Err(e) => assert_eq!(
                e.to_string(),
                "SyntaxError: Lexer rule DIGIT cannot reference Parser non-terminal digit"
            ),
        }
    }

    #[test]
    //Asserts that the fragment using wrong naming generates syntax error
    fn parse_fragments_lowercase_naming() {
        match Grammar::parse_antlr(FRAGMENTS_CASE_ERR) {
            Ok(_) => panic!("Expected a syntax error!"),
            Err(e) => assert_eq!(
                e.to_string(),
                "SyntaxError: Fragments should be uppercase: fragment digit: [0-9]+;"
            ),
        }
    }

    #[test]
    //Asserts that the fragments are replaced correctly
    fn parse_recursive_fragments() {
        match Grammar::parse_antlr(FRAGMENTS_GRAMMAR) {
            Ok(g) => {
                assert_eq!(g.len(), 2);
            }
            Err(_) => panic!(),
        }
    }

    #[test]
    //Asserts that a simple grammar is parsed correctly.
    fn parse_simple_grammar_correctly() {
        match Grammar::parse_antlr(SIMPLE_GRAMMAR) {
            Ok(g) => assert_eq!(
                g.len(),
                9,
                "Grammar was parsed correctly, but a different number of production was expected"
            ),
            Err(_) => panic!("Simple grammar failed to parse"),
        }
    }

    #[test]
    //Asserts that ALL the productions are parsed correctly
    fn get_production() {
        let g = Grammar::parse_antlr(SIMPLE_GRAMMAR).unwrap();
        let nterm = g.iter_nonterm().map(|prod| &prod.body).collect::<Vec<_>>();
        let term = g.iter_term().map(|prod| &prod.body).collect::<Vec<_>>();
        let expected_term = [
            "~[,\\n\\r\"]+",
            "'\"'('\"\"'|~'\"')*'\"'",
            "','",
            "'\\r'",
            "'\\n'",
        ];
        let expected_nterm = [
            "hdr row+ ",
            "row ",
            "field (COMMA field)* CR? LF ",
            "TEXT| STRING|",
        ];
        assert_eq!(term, expected_term);
        assert_eq!(nterm, expected_nterm);
    }

    #[test]
    //Asserts that the order of the production is kept unchanged
    fn order_unchanged() {
        let g = Grammar::parse_antlr(SIMPLE_GRAMMAR).unwrap();
        let nterm = g.iter_nonterm().map(|prod| &prod.body).collect::<Vec<_>>();
        let term = g.iter_term().map(|prod| &prod.body).collect::<Vec<_>>();
        assert_eq!(term[0], "~[,\\n\\r\"]+");
        assert_eq!(term[1], "'\"'('\"\"'|~'\"')*'\"'");
        assert_eq!(term[2], "','");
        assert_eq!(term[3], "'\\r'");
        assert_eq!(term[4], "'\\n'");
        assert_eq!(nterm[0], "hdr row+ ");
        assert_eq!(nterm[1], "row ");
        assert_eq!(nterm[2], "field (COMMA field)* CR? LF ");
        assert_eq!(nterm[3], "TEXT| STRING|");
    }

    #[test]
    //Asserts that the order of the production is kept unchanged by iterating terminals
    fn order_iter_term() {
        match Grammar::parse_antlr(SIMPLE_GRAMMAR) {
            Ok(g) => {
                let vec = g.iter_term().map(|prod| &prod.body).collect::<Vec<_>>();
                //0, 1, 2 are replaced literals
                assert_eq!(vec[0], "~[,\\n\\r\"]+");
                assert_eq!(vec[1], "'\"'('\"\"'|~'\"')*'\"'");
            }
            Err(_) => panic!("Simple grammar failed to parse"),
        }
    }

    #[test]
    //Asserts that the order of the production is kept unchanged by iterating non-terminals
    fn order_unchanged_iter_nonterm() {
        match Grammar::parse_antlr(SIMPLE_GRAMMAR) {
            Ok(g) => {
                let vec = g.iter_nonterm().map(|prod| &prod.body).collect::<Vec<_>>();
                assert_eq!(vec[0], "hdr row+ ");
                assert_eq!(vec[1], "row ");
                assert_eq!(vec[2], "field (COMMA field)* CR? LF ");
                assert_eq!(vec[3], "TEXT| STRING|");
            }
            Err(_) => panic!("Simple grammar failed to parse"),
        }
    }

    #[test]
    //Asserts that a grammar is parsed and the actions extracted correctly.
    //Simpler version with trivial body and single action.
    fn parse_actions_simpler() -> Result<(), ParseError> {
        let grammar = Grammar::parse_antlr(LEXER_ACTIONS_SIMPLER)?;
        let terminals = grammar.iter_term().collect::<Vec<_>>();
        let index_empty = *grammar.modes_index.get("").unwrap();
        let index_channel_mode = *grammar.modes_index.get("ChannelName").unwrap();
        let expected = vec![
            Action::Skip,
            Action::More,
            Action::Type,
            Action::Type,
            Action::Channel,
            Action::Channel,
            Action::Mode(index_channel_mode),
            Action::Mode(index_empty),
            Action::PushMode(index_channel_mode),
            Action::PushMode(index_empty),
            Action::PopMode,
        ];
        for (terminal, expected) in terminals.iter().zip(expected) {
            assert_eq!(
                *terminal.actions.as_ref().unwrap().iter().next().unwrap(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    //Asserts that a grammar is parsed and the actions extracted correctly.
    //Harder version with multiple actions and tricky -> productions
    fn parse_actions_harder() -> Result<(), ParseError> {
        let grammar = Grammar::parse_antlr(LEXER_ACTIONS_HARDER)?;
        let terminals = grammar.iter_term().collect::<Vec<_>>();
        assert_eq!(terminals[1].actions.as_ref().unwrap().len(), 2); // whitespace action
        let mut ws_iter = terminals[1].actions.as_ref().unwrap().iter();
        assert_eq!(*ws_iter.next().unwrap(), Action::More);
        assert_eq!(*ws_iter.next().unwrap(), Action::Channel,);
        assert_eq!(
            *terminals[2]
                .actions
                .as_ref()
                .unwrap()
                .iter()
                .next()
                .unwrap(),
            Action::Skip
        );
        assert_eq!(
            *terminals[3]
                .actions
                .as_ref()
                .unwrap()
                .iter()
                .next()
                .unwrap(),
            Action::More
        );
        Ok(())
    }

    #[test]
    //Asserts that terminal productions are cleaned up of spaces and embedded productions
    //this should be done also in recursive replacement of terminals
    fn terminal_cleaned() {
        match Grammar::parse_antlr(LEXER_ACTIONS_HARDER) {
            Ok(g) => {
                assert_eq!(
                    g.iter_term().map(|prod| &prod.body).next().unwrap(),
                    "[a->b\\-\\]]+'->'|([ ]+)"
                );
                assert_eq!(
                    g.iter_term().map(|prod| &prod.body).nth(2).unwrap(),
                    "('\\r''\\n'?|'\\n')"
                );
            }
            Err(_) => panic!("grammar failed to parse"),
        }
    }

    #[test]
    //Asserts that invalid lexer actions are reported as errors
    fn invalid_lexer_actions() {
        match Grammar::parse_antlr(LEXER_INVALID_ACTION) {
            Ok(_) => panic!("Invalid lexer actions should not be able to parse correctly"),
            Err(e) => assert_eq!(e.to_string(), "SyntaxError: invalid action `channel(name`"),
        }
    }

    #[test]
    //Asserts that cyclic rules like S->S; cannot be solved in the lexer
    fn lexer_rules_cycles_err() {
        match Grammar::parse_antlr(LEXER_CYCLIC) {
            Ok(_) => panic!("expected a failure"),
            Err(e) => assert_eq!(
                e.to_string(),
                "SyntaxError: Lexer contains cyclic productions!"
            ),
        }
    }

    #[test]
    fn identify_literal_empty() {
        let char_literal = "''";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_basic_concat() {
        let char_literal = "'aaa'";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),OP(&)[VALUE([a]),VALUE([a])]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped() {
        let char_literal = "'a\\x24'";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),VALUE([$])]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_seq() {
        let unicode_seq = "'დოლორ'";
        let tree = Tree {
            value: ProdToken::Id(unicode_seq),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected =
            "OP(&)[VALUE([დ]),OP(&)[VALUE([ო]),OP(&)[VALUE([ლ]),OP(&)[VALUE([ო]),VALUE([რ])]]]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped_range() {
        let square = "[\\-a-d\\]]";
        let tree = Tree {
            value: ProdToken::Id(square),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([-]abcd])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_range() {
        let range = "'\\U16C3'..'\\u16C5'";
        let tree = Tree {
            value: ProdToken::Id(range),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([ᛃᛄᛅ])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_nongreedy_kleene() {
        let expr_greedy = "'a'.*'a'";
        let expr_nongreedy = "'a'.*?'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn identify_nongreedy_qm() {
        let expr_greedy = "'a'.?'a'";
        let expr_nongreedy = "'a'.??'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn identify_nongreedy_plus() {
        let expr_greedy = "'a'.+'a'";
        let expr_nongreedy = "'a'.+?'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn regex_with_unicode_literals() {
        // ANTLR does not support unicode literals in the grammar,
        // but this library does for convenience.
        let regex = "[あいうえお]|[アイウエオ]";
        let prec_tree = gen_precedence_tree(regex);
        let expected = "|[[あいうえお],[アイウエオ]]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_or_klenee() {
        let expr = "'a'|'b'*'c'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "|['a',&[*['b'],'c']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_klenee_par() {
        let expr = "'a'*('b'|'c')*'d'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[*['a'],*[|['b','c']]],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation_par() {
        let expr = "('a')~'b'('c')('d')'e'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[&[&['a',~['b']],'c'],'d'],'e']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation() {
        let expr = "'a'~'b''c'('d')";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[&['a',~['b']],'c'],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_qm_plus_klenee_par() {
        let expr = "'a'?('b'*|'c'*)+'d'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[?['a'],+[|[*['b'],*['c']]]],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_negation_klenee() {
        let expr = "~'a'*";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*[~['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_negation_klenee_or() {
        let expr = "~'a'*|'b'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "|[*[~['a']],'b']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_literal_negation_literal() {
        let expr = "'a'~'a'*'a'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&['a',*[~['a']]],'a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_nongreedy_negation() {
        let expr = "'a'~'a'*?'a'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&['a',?[*[~['a']]]],'a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_kleene_qm() {
        let expr = "(('a')*)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_not_kleene_qm() {
        let expr = "(~('a')*)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*[~['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_qm_qm() {
        let expr = "(('a')?)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "?['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_plus_qm() {
        let expr = "(('a')+)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_dont_simplify_nongreedy() {
        let expr = "('a')+?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "?[+['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regression_disappearing_literal() {
        let expr = "'a'*~'b'*";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[*['a'],*[~['b']]]";
        assert_eq!(as_str(&prec_tree), expected);
    }
}
