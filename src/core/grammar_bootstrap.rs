use crate::error::ParseError;
use crate::grammar::{Action, Grammar};
use regex::Regex;
use std::collections::{BTreeSet, HashMap, VecDeque};

const SINGLE_QUOTE_AS_U8: u8 = b'\'';

/// Parse a .g4 grammar with a manually written, ad-hoc parser.
pub(super) fn bootstrap_parse_string(content: &str) -> Result<Grammar, ParseError> {
    let productions = retrieve_productions(&content);
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
///
/// Return a tuple containing:
/// * The body without the action. For example `a -> skip` will return `a`.
/// * A Set containing every Action for the current production. For example `a -> skip, popMode`
/// will return a set containing `Action::SKIP` and `Action::POPMODE`
///
/// A SyntaxError is returned if the action is illegal.
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

/// Matches a specific action into an enum. Just a wrapper to split longer functions.
///
/// Takes as input the text representing a SINGLE action and returns the action expressed with the
/// Action enum.
///
/// Returns SyntaxError if the action is illegal
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

#[cfg(test)]
#[path = "tests/grammar.rs"]
mod tests;
