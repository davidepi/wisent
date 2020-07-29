use crate::grammar::Grammar;
use crate::lexer::{
    canonicalise, expand_literals, gen_precedence_tree, get_alphabet, Automaton, BSTree, OpType,
    RegexOp, DFA, NFA,
};

#[test]
fn canonical_tree() {
    let mut expr;
    let mut tree;
    let mut new_tree;
    let mut alphabet;
    let mut str;

    expr = "('a'*.)*'a'";
    tree = expand_literals(gen_precedence_tree(expr));
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
    tree = expand_literals(gen_precedence_tree(expr));
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
    tree = expand_literals(gen_precedence_tree(expr));
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
    tree = expand_literals(gen_precedence_tree(expr));
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
    tree = gen_precedence_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"|\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"*\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"'c'\"}}}"
    );

    expr = "'a'*('b'|'c')*'d'";
    tree = gen_precedence_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"'a'\"}},\"right\":\
    {\"val\":\"&\",\"left\":{\"val\":\"*\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"'b'\"},\"rig\
    ht\":{\"val\":\"'c'\"}}},\"right\":{\"val\":\"'d'\"}}}"
    );

    expr = "('a')~'b'('c')('d')'e'";
    tree = gen_precedence_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\
    \"right\":{\"val\":\"&\",\"left\":{\"val\":\"'d'\"},\"right\":{\"val\":\"'e'\"}}}}}"
    );

    expr = "'a'~'b''c'('d')";
    tree = gen_precedence_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"'a'\"},\"right\":{\"val\":\"&\",\"left\":{\
    \"val\":\"~\",\"left\":{\"val\":\"'b'\"}},\"right\":{\"val\":\"&\",\"left\":{\"val\":\"'c'\"},\
    \"right\":{\"val\":\"'d'\"}}}}"
    );

    expr = "'a'?('b'*|'c'*)+'d'";
    tree = gen_precedence_tree(expr);
    str = format!("{}", &tree);
    assert_eq!(
        str,
        "{\"val\":\"&\",\"left\":{\"val\":\"?\",\"left\":{\"val\":\"'a'\"}},\"right\":{\
    \"val\":\"&\",\"left\":{\"val\":\"+\",\"left\":{\"val\":\"|\",\"left\":{\"val\":\"*\",\"left\":\
    {\"val\":\"'b'\"}},\"right\":{\"val\":\"*\",\"left\":{\"val\":\"'c'\"}}}},\"right\":{\"val\":\"\
    'd'\"}}}"
    );
}

#[test]
fn dfa_conflicts_resolution() {
    //they should be different: the second accept abb as a*b+ (appearing first in the productions)
    let grammar1 = Grammar::new(
        &["'a'", "'abb'", "'a'*'b'+"],
        &[],
        &["A", "ABB", "ASTARBPLUS"],
    );
    let dfa1 = DFA::new(&grammar1);
    let grammar2 = Grammar::new(
        &["'a'*'b'+", "'abb'", "'a'"],
        &[],
        &["ASTARBPLUS", "ABB", "A"],
    );
    let dfa2 = DFA::new(&grammar2);
    assert!(!dfa1.is_empty());
    assert_eq!(dfa1.nodes(), 6);
    assert_eq!(dfa1.edges(), 9);
    assert!(!dfa2.is_empty());
    assert_eq!(dfa2.nodes(), 4);
    assert_eq!(dfa2.edges(), 7);
}

#[test]
fn dfa_direct_construction_no_sink() {
    let terminal = "('a'|'b')*'abb'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let dfa = DFA::new(&grammar);
    assert!(!dfa.is_empty());
    assert_eq!(dfa.nodes(), 4);
    assert_eq!(dfa.edges(), 8);
}

#[test]
fn dfa_direct_construction_sink_accepting() {
    let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
    let dfa = DFA::new(&grammar);
    assert!(!dfa.is_empty());
    assert_eq!(dfa.nodes(), 3);
    assert_eq!(dfa.edges(), 30);
}

#[test]
fn dfa_subset_construction_no_sink() {
    let terminal = "('a'|'b')*'abb'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let nfa = NFA::new(&grammar);
    let dfa_direct = DFA::new(&grammar);
    let dfa_subset = nfa.to_dfa();
    assert_eq!(dfa_subset.nodes(), 4);
    assert_eq!(dfa_subset.edges(), 8);
    assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
    assert_eq!(dfa_subset.edges(), dfa_direct.edges());
}

#[test]
fn dfa_subset_construction_sink_accepting() {
    let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
    let nfa = NFA::new(&grammar);
    let dfa_direct = DFA::new(&grammar);
    let dfa_subset = nfa.to_dfa();
    assert_eq!(dfa_subset.nodes(), 3);
    assert_eq!(dfa_subset.edges(), 30);
    assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
    assert_eq!(dfa_subset.edges(), dfa_direct.edges());
}

#[test]
fn dfa_direct_construction_single_acc() {
    let terminal = "(('a'*'b')|'c')?'c'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let dfa = DFA::new(&grammar);
    assert!(!dfa.is_empty());
    assert_eq!(dfa.nodes(), 5);
    assert_eq!(dfa.edges(), 7);
}

#[test]
fn dfa_direct_construction_multi_production() {
    let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    let dfa = DFA::new(&grammar);
    assert!(!dfa.is_empty());
    assert_eq!(dfa.nodes(), 3);
    assert_eq!(dfa.edges(), 3);
}

#[test]
fn dfa_direct_construction_empty() {
    let grammar = Grammar::new(&[], &[], &[]);
    let dfa = DFA::new(&grammar);
    assert!(dfa.is_empty());
}

#[test]
fn dfa_subset_construction_multi_production() {
    let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    let nfa = NFA::new(&grammar);
    let dfa_direct = DFA::new(&grammar);
    let dfa_subset = nfa.to_dfa();
    assert_eq!(dfa_subset.nodes(), 3);
    assert_eq!(dfa_subset.edges(), 3);
    assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
    assert_eq!(dfa_subset.edges(), dfa_direct.edges());
}

#[test]
fn dfa_subset_construction_single_production() {
    let terminal = "(('a'*'b')|'c')?'c'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let nfa = NFA::new(&grammar);
    let dfa_direct = DFA::new(&grammar);
    let dfa_subset = nfa.to_dfa();
    assert_eq!(dfa_subset.nodes(), 5);
    assert_eq!(dfa_subset.edges(), 7);
    assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
    assert_eq!(dfa_subset.edges(), dfa_direct.edges());
}

#[test]
fn dfa_subset_construction_empty() {
    let grammar = Grammar::new(&[], &[], &[]);
    let nfa = NFA::new(&grammar);
    let dfa = nfa.to_dfa();
    assert!(nfa.is_empty());
    assert!(dfa.is_empty());
}

#[test]
fn nfa_construction_single_production() {
    let terminal = "(('a'*'b')|'c')?'c'";
    let names = "PROD1";
    let grammar = Grammar::new(&[terminal], &[], &[names]);
    let nfa = NFA::new(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 16);
    assert_eq!(nfa.edges(), 19);
}

#[test]
fn nfa_construction_multi_production() {
    let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    let nfa = NFA::new(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 7);
    assert_eq!(nfa.edges(), 8);
}

#[test]
fn nfa_empty() {
    let grammar = Grammar::new(&[], &[], &[]);
    let nfa = NFA::new(&grammar);
    assert!(nfa.is_empty());
}
