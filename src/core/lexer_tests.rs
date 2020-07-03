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
fn regex_parse_tree() {
    let mut expr;
    let mut tree;
    let mut str;
    expr = "~[a-z]*";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(
        str,
        "{\"id\":\"*\",\"val\":[{\"id\":\"~\",\"val\":[{\"id\":\"ID\",\"val\":\"[a-z]\"},{}]},{}]}"
    );

    expr = "'a'('b'|'c')*'d'";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str, "{\"id\":\"&\",\"val\":[{\"id\":\"ID\",\"val\":\"'a'\"},{\"id\":\"&\",\"val\":[{\"id\":\"*\",\"val\":[{\"id\":\"|\",\"val\":[{\"id\":\"ID\",\"val\":\"'b'\"},{\"id\":\"ID\",\"val\":\"'c'\"}]},{}]},{\"id\":\"ID\",\"val\":\"'d'\"}]}]}");

    expr = "'#'([ \\t]+)?'define'~[#]*";
    tree = gen_parse_tree(expr).unwrap();
    str = String::new();
    tree_json(&tree, &mut str);
    assert_eq!(str,"{\"id\":\"&\",\"val\":[{\"id\":\"ID\",\"val\":\"'#'\"},{\"id\":\"&\",\"val\":[{\"id\":\"?\",\"val\":[{\"id\":\"+\",\"val\":[{\"id\":\"ID\",\"val\":\"[ \\t]\"},{}]},{}]},{\"id\":\"&\",\"val\":[{\"id\":\"ID\",\"val\":\"'define'\"},{\"id\":\"*\",\"val\":[{\"id\":\"~\",\"val\":[{\"id\":\"ID\",\"val\":\"[#]\"},{}]},{}]}]}]}]}");
}
