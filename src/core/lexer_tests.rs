use crate::error::ParseError;
use crate::lexer;
use crate::lexer::{gen_parse_tree, tree_json};

#[test]
fn identify_literal() {
    let char_literal = "'a'";
    let unicode_literal = "'駒ツ'";
    let square = "[a-z\\]\\']";
    let range = "'a'..'z'";
    let any = ".";
    let negated = "~[a-z]";
    let stop = "~[a-z]*";
    let multiple = "[a-z][A-Z]";
    let error = "'a'...'b'";
}

#[test]
//Asserts an error is thrown in case parentheses are unmatched
fn unmatched_parentheses() {
    let expr = "('a'|'b'))";
    match gen_parse_tree(expr) {
        Ok(_) => assert!(false, "The regexp should be invalid"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Unmatched parentheses in ('a'|'b'))"
        ),
    }
}

#[test]
//Asserts correctness in precedence evaluation when parentheses are not present
fn regex_correct_precedence() {
    let mut expr;
    let mut tree;
    let mut str;

    expr = "'a'|'b'*'c'";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str, "{\"val\":\"|\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"'c'\"}}}");

    expr = "'a'*('b'|'c')*'d'";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'a'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"'b'\"},\"right\":{\"val\":\"'c'\"}}},\"right\":{\"val\":\"'d'\"}}}");

    expr = "('a')~'b'('c')('d')'e'";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'d'\"},\"right\":{\"val\":\"'e'\"}}}}}");

    expr = "'a'~'b''c'('d')";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\"right\":{\"val\":\"'d'\"}}}}");

    expr = "'a'?('b'*|'c'*)+'d'";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str,"{\"val\":\"&\",\"left\":{\"val\":\"?\",\"left\":{\"val\":\"'a'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"+\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"*\",\"left\":{\"val\":\"'c'\"}}}},\"right\":{\"val\":\"'d'\"}}}");
}
