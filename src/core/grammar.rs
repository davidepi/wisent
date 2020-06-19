use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::{iter::Peekable, str::Chars};

use crate::error::ParseError;

#[derive(Debug)]
pub struct Grammar {
    pub terminals: HashMap<String, String>,
    pub non_terminals: HashMap<String, String>,
}

impl Grammar {
    /// Returns the total number of productions. This includes terminals and
    /// non-terminals but not fragments.
    /// ## Returns
    /// A number representing the sum of terminals and non-terminals productions
    pub fn len(self) -> usize {
        return self.terminals.len() + self.non_terminals.len();
    }
}

pub fn parse_grammar(path: &str) -> Result<Grammar, ParseError> {
    let grammar_content = std::fs::read_to_string(path)?;
    let productions = retrieve_productions(&grammar_content);
    let frags = build_grammar(productions)?;
    // validate productions and categorize them into terminal or non-terminal
    // based on their uppercase or lowercase letter
    let grammar = solve_terminals_dependencies(frags.0, frags.1)?;
    Ok(grammar)
}

/// Categorizes various productions into terminal, non-terminal and
/// fragments.
/// ## Arguments
/// * `productions` - A vector of String where each String is a production
/// ending with `;`. This vector may contain fragments, but in this case the
/// fragment keyword must be passed as well.
/// ## Returns
/// A Result object with the following types:
/// * `Ok(Grammar, HashMap<String, String>)` - A tuple containing the following
/// items:
///     1. a Grammar, comprised of terminals and non-terminals
///     1. an HashMap comprised of fragments with the left side of the fragment
///     as key and the right side as value
/// * `Err(ParseError)` - A ParseError object containing a description of the
/// error
///
fn build_grammar(
    productions: Vec<String>,
) -> Result<(Grammar, HashMap<String, String>), ParseError> {
    let mut terminals = HashMap::new();
    let mut non_terminals = HashMap::new();
    let mut fragments = HashMap::new();
    //the capt.group of this regex will be passed to p_re so I need to include ;
    let f_re = r"\s*fragment\s+((?:.|\n)+;)";
    let p_re = r"\s*(\w+)\s*:\s*((?:.|\n)+);";
    let re_fr = Regex::new(f_re).unwrap(); //fragment detection
    let re_pd = Regex::new(p_re).unwrap(); //production detection
    for production in &productions {
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
                        non_terminals.insert(name, rule);
                    } else {
                        return Err(ParseError::SyntaxError {
                            message: format!("Fragments should be lowercase {}", production),
                        });
                    }
                } else {
                    if !is_fragment {
                        terminals.insert(name, rule);
                    } else {
                        fragments.insert(name, rule);
                    }
                }
            }
            None => {
                return Err(ParseError::SyntaxError {
                    message: format!("Unknown production: {}", prod),
                });
            }
        }
    }
    Ok((
        Grammar {
            terminals,
            non_terminals,
        },
        fragments,
    ))
}

/// Removes the fragments from lexer rules by effectively replacing them with
/// their production body. Additionally, solves the recursion in Lexer rules
/// (this should not be a thing at all but it's allowed by ANTLR...)
/// ## Arguments
/// * `grammar` - A Grammar containing a set of terminals and non-terminals
/// * `fragments` - An HashMap containing the fragments. The key is the head of
/// the production and the value is the body.
/// ## Returns
/// * `Ok(Grammar)` - The grammar with every fragment and lexer rule recursion
/// replaced with its actual body
/// * `Err(ParseError)` - A syntax error in case the lexer or the fragments
/// reference parser rules (forbidden by the specification)
fn solve_terminals_dependencies(
    grammar: Grammar,
    fragments: HashMap<String, String>,
) -> Result<Grammar, ParseError> {
    let mut merge = fragments
        .iter()
        .map(|(k, v)| (&k[..], &v[..]))
        .collect::<HashMap<&str, &str>>();
    merge.extend(grammar.terminals.iter().map(|(k, v)| (&k[..], &v[..])));
    //create a map assigning an index to each production head. Also creates the
    //adiacency list containing the other productions used recursively in the body
    let heads_no = merge.len();
    let mut head2id = HashMap::new();
    let mut id2head = vec![""; heads_no];
    //I don't expect to have many dependencies so TreeSet > HashSet
    let mut graph = vec![BTreeSet::<usize>::new(); heads_no];
    let mut transpose = vec![BTreeSet::<usize>::new(); heads_no];
    //this array contains the index of every terminal referenced in a body
    //will be used to split the body and remove the terminals
    let mut split_here = vec![BTreeSet::<usize>::new(); heads_no];
    let mut idx = 0_usize;
    for terminal in &merge {
        id2head[idx] = *terminal.0;
        head2id.insert(*terminal.0, idx);
        idx = idx + 1;
        split_here[idx].insert(0);
        split_here[idx].insert((*terminal.0).len());
    }
    let map2ids = head2id;

    //find all the dependencies in the bodies and build the DAG (hopefully it's a DAG)
    let re = Regex::new(r"\w+").unwrap();
    for terminal in &merge {
        let term_head = *terminal.0;
        let term_body = *terminal.1;
        let term_id = *map2ids.get(term_head).unwrap(); //this DEFINITELY exists
        for mat in re.find_iter(term_body) {
            // the terminal referenced (depdendency to be satisfied)
            let dep = &term_body[mat.start()..mat.end()];
            //this is NOT guaranteed to exist! the regex can match literals!
            match map2ids.get(dep) {
                Some(dep_id) => {
                    //build the graph
                    graph[term_id].insert(*dep_id);
                    transpose[*dep_id].insert(term_id);
                    //and record the position in the string of the terminal
                    split_here[term_id].insert(mat.start());
                    split_here[term_id].insert(mat.end());
                }
                None => {
                    if grammar.non_terminals.contains_key(dep) {
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

    //if the dependencies can be satisfied, replace them
    let mut new_terminals = HashMap::<String, String>::new();
    if let Some(order) = topological_sort(&graph) {
        for node in order {
            let body = *merge.get(id2head[node]).unwrap();
            let mut last_split = 0_usize;
            //
            let new_body = split_here[node]
                .iter()
                .map(|idx| {
                    let mut ret = String::with_capacity(*idx - last_split + 2);
                    //get the slice
                    let cur_slice = &body[last_split..*idx];
                    last_split = *idx;
                    //if is a head replace it with body
                    match merge.get(cur_slice) {
                        Some(prod) => {
                            ret.push('(');
                            ret.push_str(*prod);
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
            new_terminals.insert(id2head[node].to_string(), new_body);
        }
    } else {
        return Err(ParseError::SyntaxError {
            message: format!("Lexer contains cyclic productions!"),
        });
    }
    //remove the fragments as I don't need them anymore
    for fragment in fragments {
        new_terminals.remove(&fragment.0);
    }
    Ok(Grammar {
        terminals: new_terminals,
        non_terminals: grammar.non_terminals,
    })
}

/// Performs a topological sort using an iterative DFS.
/// ## Arguments
/// * `graph` - A graph represented as adjacency list.
/// ## Returns
/// * `Some(value)` - a Vec containing the ordered indices if graph was a DAG.
/// * `None` - if the graph was not acyclic.
pub(super) fn topological_sort(graph: &Vec<BTreeSet<usize>>) -> Option<Vec<usize>> {
    //The idea is is the one described by Cormen et al. (2001), Mark record
    //if the DFS can reach node of the current branch and thus there is a cycle
    //In addition, being this function iterative, the `toprocess` array is used
    //to defer the node into post-order.
    #[derive(Clone, PartialEq)]
    enum Mark {
        NONE,      //Node untouched
        TEMPORARY, //Current node being processed
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
/// This effectively works by removing every comment and then splitting over ;
/// tokens that are not quoted, although in this functions is implemented as a
/// single pass.
/// The comments removed are the multiline `/*`-`*/` and single line `//`, `#`.
/// ## Arguments
/// * `content` - A string containing the original `.g4` grammar content.
/// * `filename` - The name of the original file parsed.
/// ## Returns
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
                    consume_line(&mut it);
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
    //remove productions without colon. This should remove the grammar XX; stmt.
    //very naive as a proper check for escaped char will be performed later.
    //FIXME: this sucks. I wrote a grammar as S -> A; and didn't realise the
    //       syntax error because it was stripped away :angery:
    productions
        .into_iter()
        .filter(|s| s.contains(':'))
        .collect()
}

/// Advances the iterator until the next `\n` character.
/// Also the last `\n` is discarded.
/// ## Arguments
/// * `it` The iterator that will be advanced
/// * `ret` The string where the final \n will be appended
fn consume_line(it: &mut Peekable<Chars>) {
    while let Some(skip) = it.next() {
        if skip == '\n' {
            break;
        }
    }
}

/// Advances the iterator until the given character and appends all the
/// encountered characters. This function takes into account also escape
/// character, so if the given charcater is ', this won't  stop in case a \' is
/// encountered.
/// ## Arguments
/// * `it` - The iterator that will be advanced
/// * `ret` - The string where the various character will be appended
/// * `until` - The character that will stop the method. Escaped versions of
/// this character won't be considered
fn append_until(it: &mut Peekable<Chars>, ret: &mut String, until: char) {
    let mut escapes = 0;
    while let Some(push) = it.next() {
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
