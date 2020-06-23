use std::collections::{BTreeSet, HashMap};

use grammar::Grammar;

use crate::grammar;

#[test]
//Asserts the method len() returns the sum of terminal and non terminals
fn grammar_len() {
    let mut g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert_eq!(g.len(), 0);
    let mut terminals = vec!["[a-z]".to_owned(), "[A-Z]".to_owned()];
    let mut non_terminals = vec![
        "LETTER_UP | LETTER_LO".to_owned(),
        "word letter | letter".to_owned(),
    ];
    let mut names = vec![
        "LETTER_LO".to_owned(),
        "LETTER_UP".to_owned(),
        "letter".to_owned(),
        "word".to_owned(),
    ];
    let mut g = Grammar::new(&terminals, &non_terminals, &names);
    assert_eq!(g.len(), 4);
}

#[test]
//Asserts the method is_empty() works as expected
fn grammar_is_empty() {
    let mut g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert!(g.is_empty());
    let mut terminals = vec!["[a-z]".to_owned(), "[A-Z]".to_owned()];
    let mut non_terminals = vec![
        "LETTER_UP | LETTER_LO".to_owned(),
        "word letter | letter".to_owned(),
    ];
    let mut names = vec![
        "LETTER_LO".to_owned(),
        "LETTER_UP".to_owned(),
        "letter".to_owned(),
        "word".to_owned(),
    ];
    let mut g = Grammar::new(&terminals, &non_terminals, &names);
    assert!(!g.is_empty());
}

#[test]
//Asserts that non-existent files returns error
fn parse_grammar_non_existent() {
    match grammar::parse_grammar("./resources/java_grammar.txt") {
        Ok(_) => assert!(false, "Expected the file to not exist!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "IOError: No such file or directory (os error 2)"
        ),
    }
}

#[test]
//Asserts that the file is parsed correctly even with high number of escape chars
fn parse_highly_escaped() {
    match grammar::parse_grammar("./resources/comment_rich_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 5);
        }
        Err(e) => assert!(false, e.to_string()),
    }
}

#[test]
//Asserts that the fragment using non-terminals generates syntax error
fn parse_fragments_nonterminal() {
    match grammar::parse_grammar("./resources/fragments_contains_nt.txt") {
        Ok(_) => assert!(false, "Expected a syntax error!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Lexer rule DIGIT cannot reference Parser non-terminal digit"
        ),
    }
}

#[test]
//Asserts that the fragment using wrong naming generates syntax error
fn parse_fragments_lowercase_naming() {
    match grammar::parse_grammar("./resources/fragments_case_err.txt") {
        Ok(_) => assert!(false, "Expected a syntax error!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Fragments should be lowercase: fragment digit: [0-9]+;"
        ),
    }
}

#[test]
//Asserts that the fragments are replaced correctly
fn parse_recursive_fragments() {
    match grammar::parse_grammar("./resources/fragments_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 2);
        }
        Err(_) => assert!(false),
    }
}

#[test]
//Asserts that a simple grammar is parsed correctly.
fn parse_simple_grammar_correctly() {
    match grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => assert_eq!(
            g.len(),
            6,
            "Grammar was parsed correctly, but a different number of production was expected"
        ),
        Err(_) => assert!(false, "Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that the order of the production is kept unchanged (between terminals and non-terminals)
fn order_unchanged() {
    match grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.names[0], "TEXT");
            assert_eq!(g.terminals[0], "~[,\\n\\r\"]+ ");
            assert_eq!(g.names[1], "STRING");
            assert_eq!(g.terminals[1], "'\"' ('\"\"'|~'\"')* '\"' ");
            assert_eq!(g.names[2], "csvFile");
            assert_eq!(g.names[3], "hdr");
            assert_eq!(g.names[4], "row");
            assert_eq!(g.non_terminals[2], "field (',' field)* '\\r'? '\\n' ");
            assert_eq!(g.names[5], "field");
        }
        Err(_) => assert!(false, "Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that the C ANTLR grammar is parsed correctly. This grammar is longer than the CSV one.
fn parse_c_grammar_correctly() {
    match grammar::parse_grammar("./resources/c_grammar.txt") {
        Ok(g) => assert_eq!(
            g.len(),
            191,
            "Grammar was parsed correctly, but a different number of production was expected"
        ),
        Err(_) => assert!(false, "C grammar failed to parse"),
    }
}

#[test]
//Asserts that cyclic rules like S->S; cannot be solved in the lexer
fn lexer_rules_cycles_err() {
    match grammar::parse_grammar("./resources/lexer_cyclic.txt") {
        Ok(_) => assert!(false, "expected a failure"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Lexer contains cyclic productions!"
        ),
    }
}

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
    match grammar::topological_sort(&graph) {
        Some(order) => {
            let expected = vec![3, 4, 2, 1, 0, 5];
            assert_eq!(order, expected, "Wrong topological order");
        }
        None => assert!(false, "A DAG should return a topological sort"),
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
    match grammar::topological_sort(&graph) {
        Some(_) => assert!(
            false,
            "A graph with cycles should not have a topological order"
        ),
        None => assert!(true, "A DAG should return a topological sort"),
    }
}
