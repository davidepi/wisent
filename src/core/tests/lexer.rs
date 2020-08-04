use crate::grammar::Grammar;
use crate::lexer::{
    canonicalise, expand_literals, gen_precedence_tree, get_set_of_symbols, Automaton, BSTree,
    OpType, RegexOp, SymbolTable, NFA,
};
use std::collections::BTreeSet;

#[test]
fn canonical_tree_any() {
    let expr = "('a'*.)*'a'";
    let tree = expand_literals(gen_precedence_tree(expr));
    let alphabet = get_set_of_symbols(&tree);
    let symtable = SymbolTable::new(alphabet);
    let new_tree = canonicalise(tree, &symtable);
    let str = format!("{}", new_tree);
    assert_eq!(
        str,
        r#"{"val":"&","left":{"val":"*","left":{"val":"&","left":{"val":"*","left":{"val":"1"}},"right":{"val":"|","left":{"val":"1"},"right":{"val":"2"}}}},"right":{"val":"1"}}"#
    );
}

#[test]
fn canonical_tree_plus() {
    let expr = "('a'*'b')+'a'";
    let tree = expand_literals(gen_precedence_tree(expr));
    let alphabet = get_set_of_symbols(&tree);
    let symtable = SymbolTable::new(alphabet);
    let new_tree = canonicalise(tree, &symtable);
    let str = format!("{}", new_tree);
    assert_eq!(
        str,
        r#"{"val":"&","left":{"val":"&","left":{"val":"&","left":{"val":"*","left":{"val":"1"}},"right":{"val":"2"}},"right":{"val":"*","left":{"val":"&","left":{"val":"*","left":{"val":"1"}},"right":{"val":"2"}}}},"right":{"val":"1"}}"#
    );
}

#[test]
fn canonical_tree_qm() {
    let expr = "('a'*'b')?'a'";
    let tree = expand_literals(gen_precedence_tree(expr));
    let alphabet = get_set_of_symbols(&tree);
    let symtable = SymbolTable::new(alphabet);
    let new_tree = canonicalise(tree, &symtable);
    let str = format!("{}", new_tree);
    assert_eq!(
        str,
        r#"{"val":"&","left":{"val":"|","left":{"val":"ϵ"},"right":{"val":"&","left":{"val":"*","left":{"val":"1"}},"right":{"val":"2"}}},"right":{"val":"1"}}"#
    );
}

#[test]
fn canonical_tree_neg() {
    let expr = "~[ab]('a'|'c')";
    let tree = expand_literals(gen_precedence_tree(expr));
    let alphabet = get_set_of_symbols(&tree);
    let symtable = SymbolTable::new(alphabet);
    let new_tree = canonicalise(tree, &symtable);
    let str = format!("{}", new_tree);
    assert_eq!(
        str,
        r#"{"val":"&","left":{"val":"|","left":{"val":"3"},"right":{"val":"4"}},"right":{"val":"|","left":{"val":"1"},"right":{"val":"3"}}}"#
    );
}

#[test]
fn identify_literal_empty() {
    let char_literal = "''";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = char_literal;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(str, r#"{"val":"VALUE([])"}"#);
}

#[test]
fn identify_literal_basic_concat() {
    let char_literal = "'aaa'";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = char_literal;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        r#"{"val":"OP(&)","left":{"val":"VALUE([a])"},"right":{"val":"OP(&)","left":{"val":"VALUE([a])"},"right":{"val":"VALUE([a])"}}}"#
    );
}

#[test]
fn identify_literal_escaped() {
    let char_literal = "'a\\x24'";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = char_literal;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        r#"{"val":"OP(&)","left":{"val":"VALUE([a])"},"right":{"val":"VALUE([$])"}}"#
    );
}

#[test]
fn identify_literal_unicode_seq() {
    let unicode_seq = "'დოლორ'";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = unicode_seq;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(
        str,
        r#"{"val":"OP(&)","left":{"val":"VALUE([დ])"},"right":{"val":"OP(&)","left":{"val":"VALUE([ო])"},"right":{"val":"OP(&)","left":{"val":"VALUE([ლ])"},"right":{"val":"OP(&)","left":{"val":"VALUE([ო])"},"right":{"val":"VALUE([რ])"}}}}}"#
    );
}

#[test]
fn identify_literal_escaped_range() {
    let square = "[\\-a-d\\]]";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = square;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(str, r#"{"val":"VALUE([-]abcd])"}"#);
}

#[test]
fn identify_literal_unicode_range() {
    let range = "'\\U16C3'..'\\u16C5'";
    let mut tree = BSTree {
        value: RegexOp {
            r#type: OpType::ID,
            value: "",
            priority: 0,
        },
        left: None,
        right: None,
    };
    tree.value.value = range;
    let ret_tree = expand_literals(tree);
    let str = format!("{}", &ret_tree);
    assert_eq!(str, r#"{"val":"VALUE([ᛃᛄᛅ])"}"#);
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

// #[test]
// fn dfa_conflicts_resolution() {
//     //they should be different: the second accept abb as a*b+ (appearing first in the productions)
//     let grammar1 = Grammar::new(
//         &["'a'", "'abb'", "'a'*'b'+"],
//         &[],
//         &["A", "ABB", "ASTARBPLUS"],
//     );
//     let dfa1 = DFA::new(&grammar1);
//     let grammar2 = Grammar::new(
//         &["'a'*'b'+", "'abb'", "'a'"],
//         &[],
//         &["ASTARBPLUS", "ABB", "A"],
//     );
//     let dfa2 = DFA::new(&grammar2);
//     assert!(!dfa1.is_empty());
//     assert_eq!(dfa1.nodes(), 6);
//     assert_eq!(dfa1.edges(), 9);
//     assert!(!dfa2.is_empty());
//     assert_eq!(dfa2.nodes(), 4);
//     assert_eq!(dfa2.edges(), 7);
// }
//
// #[test]
// fn dfa_direct_construction_no_sink() {
//     let terminal = "('a'|'b')*'abb'";
//     let names = "PROD1";
//     let grammar = Grammar::new(&[terminal], &[], &[names]);
//     let dfa = DFA::new(&grammar);
//     assert!(!dfa.is_empty());
//     assert_eq!(dfa.nodes(), 4);
//     assert_eq!(dfa.edges(), 8);
// }
//
// #[test]
// fn dfa_direct_construction_sink_accepting() {
//     let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
//     let dfa = DFA::new(&grammar);
//     assert!(!dfa.is_empty());
//     assert_eq!(dfa.nodes(), 3);
//     assert_eq!(dfa.edges(), 30);
// }
//
// #[test]
// fn dfa_subset_construction_no_sink() {
//     let terminal = "('a'|'b')*'abb'";
//     let names = "PROD1";
//     let grammar = Grammar::new(&[terminal], &[], &[names]);
//     let nfa = NFA::new(&grammar);
//     let dfa_direct = DFA::new(&grammar);
//     let dfa_subset = nfa.to_dfa();
//     assert_eq!(dfa_subset.nodes(), 4);
//     assert_eq!(dfa_subset.edges(), 8);
//     assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
//     assert_eq!(dfa_subset.edges(), dfa_direct.edges());
// }
//
// #[test]
// fn dfa_subset_construction_sink_accepting() {
//     let grammar = Grammar::new(&["[0-9]", "[0-9]+"], &[], &["digits", "more_digits"]);
//     let nfa = NFA::new(&grammar);
//     let dfa_direct = DFA::new(&grammar);
//     let dfa_subset = nfa.to_dfa();
//     assert_eq!(dfa_subset.nodes(), 3);
//     assert_eq!(dfa_subset.edges(), 30);
//     assert_eq!(dfa_subset.nodes(), dfa_direct.nodes());
//     assert_eq!(dfa_subset.edges(), dfa_direct.edges());
// }
//
// #[test]
// fn dfa_direct_construction_start_accepting() {
//     let grammar = Grammar::new(&["'a'*"], &[], &["ASTAR"]);
//     let dfa_direct = DFA::new(&grammar);
//     assert!(!dfa_direct.is_empty());
//     assert_eq!(dfa_direct.nodes(), 1);
//     assert_eq!(dfa_direct.edges(), 1);
// }
//
// #[test]
// fn dfa_direct_construction_single_acc() {
//     let terminal = "(('a'*'b')|'c')?'c'";
//     let names = "PROD1";
//     let grammar = Grammar::new(&[terminal], &[], &[names]);
//     let dfa = DFA::new(&grammar);
//     assert!(!dfa.is_empty());
//     assert_eq!(dfa.nodes(), 5);
//     assert_eq!(dfa.edges(), 7);
// }
//
// #[test]
// fn dfa_direct_construction_multi_production() {
//     let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
//     let dfa = DFA::new(&grammar);
//     assert!(!dfa.is_empty());
//     assert_eq!(dfa.nodes(), 3);
//     assert_eq!(dfa.edges(), 3);
// }
//
// #[test]
// fn dfa_direct_construction_empty() {
//     let grammar = Grammar::new(&[], &[], &[]);
//     let dfa = DFA::new(&grammar);
//     assert!(dfa.is_empty());
// }
//
// #[test]
// fn dfa_subset_construction_multi_production() {
//     let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
//     let nfa = NFA::new(&grammar);
//     let dfa_subset = nfa.to_dfa();
//     assert_eq!(dfa_subset.nodes(), 3);
//     assert_eq!(dfa_subset.edges(), 3);
// }
//
// #[test]
// fn dfa_subset_construction_start_accepting() {
//     let grammar = Grammar::new(&["'a'*"], &[], &["ASTAR"]);
//     let nfa = NFA::new(&grammar);
//     let dfa_subset = nfa.to_dfa();
//     assert!(!dfa_subset.is_empty());
//     assert_eq!(dfa_subset.nodes(), 1);
//     assert_eq!(dfa_subset.edges(), 1);
// }
//
// #[test]
// fn dfa_subset_construction_single_production() {
//     let terminal = "(('a'*'b')|'c')?'c'";
//     let names = "PROD1";
//     let grammar = Grammar::new(&[terminal], &[], &[names]);
//     let nfa = NFA::new(&grammar);
//     let dfa_subset = nfa.to_dfa();
//     assert_eq!(dfa_subset.nodes(), 5);
//     assert_eq!(dfa_subset.edges(), 7);
// }
//
// #[test]
// fn dfa_subset_construction_empty() {
//     let grammar = Grammar::new(&[], &[], &[]);
//     let nfa = NFA::new(&grammar);
//     let dfa = nfa.to_dfa();
//     assert!(nfa.is_empty());
//     assert!(dfa.is_empty());
// }

#[test]
fn nfa_set_productions() {
    let grammar = Grammar::new(&["[a-c]([b-d]?[e-g])*", "[fg]+"], &[], &["LONG1", "LONG2"]);
    let nfa = NFA::new(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 31);
    assert_eq!(nfa.edges(), 38);
}

#[test]
fn nfa_start_accepting() {
    let grammar = Grammar::new(&["'ab'*"], &[], &["ABSTAR"]);
    let nfa = NFA::new(&grammar);
    assert!(!nfa.is_empty());
    assert_eq!(nfa.nodes(), 6);
    assert_eq!(nfa.edges(), 7);
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

#[test]
fn symbol_table_single_set() {
    let set1 = btreeset! {'a'};
    let set = btreeset! {set1};
    let symbol = SymbolTable::new(set);
    assert_eq!(*symbol.table.get(&'a').unwrap(), 1 as usize);
    assert_eq!(symbol.ids(), 3);
}

#[test]
fn symbol_table_construction() {
    let set1 = btreeset! {'a', 'b', 'c'};
    let set2 = btreeset! {'b', 'c', 'd'};
    let set3 = btreeset! {'d', 'e'};
    let set4 = btreeset! {'e', 'f', 'g', 'h'};
    let set5 = btreeset! {'h', 'a'};
    let set6 = btreeset! {'h', 'c'};
    let set = btreeset! {set1, set2, set3, set4, set5, set6};
    let symbol = SymbolTable::new(set);
    assert_eq!(symbol.get('f'), symbol.get('g'));
    assert_ne!(symbol.get('a'), symbol.get('b'));
}

#[test]
fn symbol_table_empty() {
    let symbol = SymbolTable::empty();
    assert_eq!(symbol.ids(), 2);
}

#[test]
fn symbol_table_character_outside_alphabet() {
    let set1 = btreeset! {'a', 'b', 'c'};
    let set2 = btreeset! {'b', 'c', 'd'};
    let symbol = SymbolTable::new(btreeset! {set1, set2});
    assert_eq!(symbol.ids(), 5);
    assert_eq!(symbol.get('e'), symbol.ids() - 1);
}

#[test]
fn symbol_table_get_set() {
    let set1 = btreeset! {'a', 'b', 'c', 'd', 'e', 'f', 'g',};
    let set2 = btreeset! {'d', 'e', 'f'};
    let set3 = btreeset! {'f','g','h'};
    let set = btreeset! {set1, set2, set3};
    let symbol = SymbolTable::new(set);
    assert_ne!(symbol.get('f'), symbol.get('g'));
    assert_eq!(symbol.get('d'), symbol.get('e'));

    let retrieve1 = symbol.get_set(&btreeset! {'a', 'b', 'c'});
    assert_eq!(retrieve1.len(), 1); //[a, b, c] have the same value
    let retrieve2 = symbol.get_set(&btreeset! {'d', 'e', 'f'});
    assert_eq!(retrieve2.len(), 2); //[d, e] [f] are the sets
}

#[test]
fn symbol_table_get_negated() {
    let set1 = btreeset! {'a', 'b', 'c'};
    let set2 = btreeset! {'b', 'c', 'd'};
    let symbol = SymbolTable::new(btreeset! {set1, set2});

    let negate_me = btreeset! {'b','c'};
    let negated = symbol.get_negated(&negate_me);
    assert!(negated.contains(&symbol.get('a')));
    assert!(negated.contains(&symbol.get('d')));
    assert!(negated.contains(&symbol.get('㊈')));
    assert!(!negated.contains(&symbol.get('b')));
    assert!(!negated.contains(&symbol.get('c')));
}
