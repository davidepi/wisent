use crate::grammar;

#[test]
//Assert that non-existent files returns error
fn parse_grammar_non_existent() {
    match grammar::parse_grammar("./resources/java_grammar.txt") {
        Ok(_) => assert!(false, "Expected the file to not exist!"),
        Err(_) => assert!(true),
    }
}

#[test]
//Assert that the file is parsed
fn parse_grammar_existent() {
    match grammar::parse_grammar("./resources/comment_rich_grammar.txt") {
        Ok(g) => {
            let productions = g.productions;
            assert_eq!(productions.len(), 5);
        }
        Err(_) => assert!(false),
    }
}
