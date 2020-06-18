use crate::grammar;

#[test]
//Assert that non-existent files returns error
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
//Assert that the file is parsed correctly even with high number of escape chars
fn parse_highly_escaped() {
    match grammar::parse_grammar("./resources/comment_rich_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 5);
        }
        Err(e) => assert!(false, e.to_string()),
    }
}

#[test]
//Assert that the fragment using non-terminals generates syntax error
fn parse_fragments_nonterminal() {
    match grammar::parse_grammar("./resources/fragments_contains_nt.txt") {
        Ok(_) => assert!(false, "Expected a syntax error!"),
        Err(e) => assert_eq!(
            &e.to_string()[..],
            "SyntaxError: Lexer rule DIGIT cannot reference Parser non-terminal digit"
        ),
    }
}

#[test]
//Assert that the fragment using wrong naming generates syntax error
fn parse_fragments_lowercase_naming() {
    match grammar::parse_grammar("./resources/fragments_case_err.txt") {
        Ok(_) => assert!(false, "Expected a syntax error!"),
        Err(e) => assert_eq!(
            &e.to_string()[..],
            "SyntaxError: Fragments should be lowercase"
        ),
    }
}

#[test]
//Assert that the fragments are replaced correctly
fn parse_recursive_fragments() {
    match grammar::parse_grammar("./resources/fragments_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 2);
        }
        Err(_) => assert!(false),
    }
}

#[test]
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
