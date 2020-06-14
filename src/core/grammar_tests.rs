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
        Ok(s) => {
            let res: String = s.content.chars().filter(|c| !c.is_whitespace()).collect();
            assert_eq!(
                &res,
                "s:sABCD;A:[\\]\'\"/*#//]B:\'\\\'\'C:[\\\\\\]/*]D:\'\\\\\'"
            )
        }
        Err(_) => assert!(false),
    }
}
