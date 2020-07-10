use crate::error::ParseError;
use crate::lexer;
use crate::lexer::{expand_literals, gen_parse_tree, BSTree, OpType, RegexOp};

#[test]
fn identify_literal() {
    let tree_sample = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    let mut tree;
    let mut ret_tree;
    let mut str;

    let char_literal = "'a\\x24'";
    tree = tree_sample.clone();
    tree.value.value = char_literal;
    ret_tree = expand_literals(tree);
    str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        "{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(a)\"},\"right\":{\"val\":\"VALUE($)\"}}"
    );

    let unicode_seq = "'დოლორ'";
    tree = tree_sample.clone();
    tree.value.value = unicode_seq;
    ret_tree = expand_literals(tree);
    str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        "{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(დ)\"},\"right\":{\"val\":\"OP(&)\
    \",\"left\":{\"val\":\"VALUE(ო)\"},\"right\":{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(ლ)\"}\
    ,\"right\":{\"val\":\"OP(&)\",\"left\":{\"val\":\"VALUE(ო)\"},\"right\":{\"val\":\"VALUE(რ)\"}}\
    }}}"
    );

    let square = "[\\-a-d\\]]";
    tree = tree_sample.clone();
    tree.value.value = square;
    ret_tree = expand_literals(tree);
    str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        "{\"val\":\"OP(|)\",\"left\":{\"val\":\"VALUE(-)\"},\"right\":{\"val\":\"OP(|)\
    \",\"left\":{\"val\":\"VALUE(a)\"},\"right\":{\"val\":\"OP(|)\",\"left\":{\"val\":\"VALUE(b)\"}\
    ,\"right\":{\"val\":\"OP(|)\",\"left\":{\"val\":\"VALUE(c)\"},\"right\":{\"val\":\"OP(|)\",\"le\
    ft\":{\"val\":\"VALUE(d)\"},\"right\":{\"val\":\"VALUE(])\"}}}}}}"
    );

    let range = "'\\U16C3'..'\\u16C5'";
    tree = tree_sample.clone();
    tree.value.value = range;
    ret_tree = expand_literals(tree);
    str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        "{\"val\":\"OP(|)\",\"left\":{\"val\":\"VALUE(ᛃ)\"},\"right\":{\"val\":\"OP(|)\
    \",\"left\":{\"val\":\"VALUE(ᛄ)\"},\"right\":{\"val\":\"VALUE(ᛅ)\"}}}"
    );
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
    assert_eq!(
        str,
        "{\"val\":\"|\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"'c'\"}}}"
    );

    expr = "'a'*('b'|'c')*'d'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'a'\"}},\"right\":\
    {\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"'b'\"},\"rig\
    ht\":{\"val\":\"'c'\"}}},\"right\":{\"val\":\"'d'\"}}}"
    );

    expr = "('a')~'b'('c')('d')'e'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\
    \"right\":{\"val\":\"&\",\"left\":{\"val\":\"'d'\"},\"right\":{\"val\":\"'e'\"}}}}}"
    );

    expr = "'a'~'b''c'('d')";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\
    \"right\":{\"val\":\"'d'\"}}}}"
    );

    expr = "'a'?('b'*|'c'*)+'d'";
    tree = gen_parse_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"?\",\"left\":{\"val\":\"'a'\"}},\"right\":{\
    \"val\":\"&\",\"left\":{\"val\":\"+\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"*\",\"left\":\
    {\"val\":\"'b'\"}},\"right\":{\"val\":\"*\",\"left\":{\"val\":\"'c'\"}}}},\"right\":{\"val\":\"\
    'd'\"}}}"
    );
}
