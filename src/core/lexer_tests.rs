use crate::error::ParseError;
use crate::lexer;
use crate::lexer::{collect_alphabet, expand_literal_node, gen_parse_tree, replace_dot_wildcard};
use std::collections::HashSet;

#[test]
fn replace_dot() {
    // let mut alphabet = HashSet::new();
    let expr = "'abc'|.*";
    let mut tree = gen_parse_tree(expr);
    let expanded = collect_alphabet(tree);
    let replaced = replace_dot_wildcard(expanded.0, expanded.1);
    println!("{}", replaced);
}

#[test]
fn identify_literal() {
    let mut tree;
    let mut char_set = HashSet::new();
    let char_literal = "'a\\x24'";
    tree = expand_literal_node(char_literal, &mut char_set);
    assert_eq!(char_set.len(), 2);
    assert!(char_set.contains(&'a'));
    assert!(char_set.contains(&'$'));
    char_set.clear();

    let unicode_literal = "'დოლორ'";
    tree = expand_literal_node(unicode_literal, &mut char_set);
    assert_eq!(char_set.len(), 4);
    assert!(char_set.contains(&'დ'));
    assert!(char_set.contains(&'ო'));
    assert!(char_set.contains(&'ლ'));
    assert!(char_set.contains(&'ო'));
    assert!(char_set.contains(&'რ'));
    char_set.clear();
    assert_eq!(format!("{}",tree), "{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(დ)\"},\"right\":{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(ო)\"},\"right\":{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(ლ)\"},\"right\":{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(ო)\"},\"right\":{\"val\":\"VALUE(რ)\"}}}}}");

    let square = "[\\-a-z\\]]";
    tree = expand_literal_node(square, &mut char_set);
    assert_eq!(char_set.len(), 28);
    assert!(char_set.contains(&'c'));
    assert!(!char_set.contains(&'9'));
    assert!(char_set.contains(&']'));
    assert!(char_set.contains(&'-'));
    char_set.clear();

    let range = "'\\U16C3'..'\\u16C5'";
    tree = expand_literal_node(range, &mut char_set);
    assert_eq!(char_set.len(), 3);
    assert!(char_set.contains(&'ᛃ'));
    assert!(char_set.contains(&'ᛄ'));
    assert!(char_set.contains(&'ᛅ'));
    char_set.clear();
}

#[test]
//Asserts correctness in precedence evaluation when parentheses are not present
fn regex_correct_precedence() {
    let mut expr;
    let mut tree;
    let mut str;

    expr = "'a'|'b'*'c'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(str, "{\"val\":\"|\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"'c'\"}}}");

    expr = "'a'*('b'|'c')*'d'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'a'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"'b'\"},\"right\":{\"val\":\"'c'\"}}},\"right\":{\"val\":\"'d'\"}}}");

    expr = "('a')~'b'('c')('d')'e'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'d'\"},\"right\":{\"val\":\"'e'\"}}}}}");

    expr = "'a'~'b''c'('d')";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(str, "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\"right\":{\"val\":\"'d'\"}}}}");

    expr = "'a'?('b'*|'c'*)+'d'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(str,"{\"val\":\"&\",\"left\":{\"val\":\"?\",\"left\":{\"val\":\"'a'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"+\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"*\",\"left\":{\"val\":\"'c'\"}}}},\"right\":{\"val\":\"'d'\"}}}");
}
