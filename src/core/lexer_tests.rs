use crate::grammar::Grammar;
use crate::lexer::{
    canonicalise, expand_literals, gen_parse_tree, get_alphabet, subset_construction,
    transition_table_dfa, transition_table_nfa, BSTree, OpType, RegexOp,
};
use std::fs;

#[test]
fn lex_c_grammar() {
    let g = Grammar::parse_grammar("./resources/c_grammar.txt").unwrap();
    let dfa = transition_table_dfa(&g);
    let str = format!("{}", dfa);
    fs::write("/home/davide/Desktop/prova.dot", str).expect("Unable to write file");
}

#[test]
fn dfa_construction_conflicts() {
    //they should be different: the second accept abb as a*b+ (appearing first in the productions)
    let grammar1 = Grammar::new(
        &["'a'", "'abb'", "'a'*'b'+"],
        &[],
        &["A", "ABB", "ASTARBPLUS"],
    );
    let dfa1 = transition_table_dfa(&grammar1);
    let grammar2 = Grammar::new(
        &["'a'*'b'+", "'abb'", "'a'"],
        &[],
        &["ASTARBPLUS", "ABB", "A"],
    );
    let dfa2 = transition_table_dfa(&grammar2);
    println!("{}", dfa1);
    println!("{}", dfa2);
}

#[test]
fn dfa_construction_single_acc() {
    let terminal = "('a'|'b')*'abb'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let dfa = transition_table_dfa(&grammar);
    println!("{}", dfa);
}

#[test]
fn dfa_construction_multi_production() {
    let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    let dfa = transition_table_dfa(&grammar);
    assert!(!dfa.is_empty());
    println!("{}", dfa);
}

#[test]
fn nfa_construction_single_production() {
    let terminal = "(('a'*'b')|'c')?'c'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let nfa = transition_table_nfa(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 16);
    assert_eq!(nfa.edges(), 19);
}

#[test]
fn nfa_construction_multi_production() {
    let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    let nfa = transition_table_nfa(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 7);
    assert_eq!(nfa.edges(), 8);
}

#[test]
fn canonical_tree() {
    let mut expr;
    let mut tree;
    let mut new_tree;
    let mut alphabet;
    let mut str;

    expr = "('a'*.)*'a'";
    tree = expand_literals(gen_parse_tree(expr));
    alphabet = get_alphabet(&tree);
    new_tree = canonicalise(tree, &alphabet);
    str = format!("{}", new_tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"l\
        eft\":{\"val\":\"a\"}},\"right\":{\"val\":\"|\",\"left\":{\"val\":\"\u{10a261}\"},\"right\"\
        :{\"val\":\"a\"}}}},\"right\":{\"val\":\"a\"}}"
    );

    expr = "('a'*'b')+'a'";
    tree = expand_literals(gen_parse_tree(expr));
    alphabet = get_alphabet(&tree);
    new_tree = canonicalise(tree, &alphabet);
    str = format!("{}", new_tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"&\",\"left\":{\"val\":\"&\",\"left\":{\"val\
    \":\"*\",\"left\":{\"val\":\"a\"}},\"right\":{\"val\":\"b\"}},\"right\":{\"val\":\"*\",\"left\"\
    :{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"a\"}},\"right\":{\"val\":\"b\"}}}},\
    \"right\":{\"val\":\"a\"}}"
    );

    expr = "('a'*'b')?'a'";
    tree = expand_literals(gen_parse_tree(expr));
    alphabet = get_alphabet(&tree);
    new_tree = canonicalise(tree, &alphabet);
    str = format!("{}", new_tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"\u{107fe1}\"},\"rig\
    ht\":{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"a\"}},\"right\":{\"val\":\"b\"}\
    }},\"right\":{\"val\":\"a\"}}"
    );

    expr = "~[ab]('a'|'c')";
    tree = expand_literals(gen_parse_tree(expr));
    alphabet = get_alphabet(&tree);
    new_tree = canonicalise(tree, &alphabet);
    str = format!("{}", new_tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"c\"},\"right\":{\"val\":\"\
        \u{10A261}\"}},\"right\":{\"val\":\"|\",\"left\":{\"val\":\"a\"},\"right\":{\"val\":\"c\
    \"}}}"
    );
}

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
