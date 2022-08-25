use crate::grammar::{Action, Grammar};
use crate::grammar_bootstrap::topological_sort;
use std::collections::BTreeSet;

#[test]
//Asserts that a DAG return correctly a topological sort
fn topological_sort_dag() {
    let n0: BTreeSet<usize> = vec![1, 4].into_iter().collect();
    let n1: BTreeSet<usize> = vec![2].into_iter().collect();
    let n2: BTreeSet<usize> = vec![3].into_iter().collect();
    let n3: BTreeSet<usize> = vec![].into_iter().collect();
    let n4: BTreeSet<usize> = vec![3].into_iter().collect();
    let n5: BTreeSet<usize> = vec![].into_iter().collect();

    let graph = vec![n0, n1, n2, n3, n4, n5];
    match topological_sort(&graph) {
        Some(order) => {
            let expected = vec![3, 4, 2, 1, 0, 5];
            assert_eq!(order, expected, "Wrong topological order");
        }
        None => panic!("A DAG should return a topological sort"),
    }
}

#[test]
//Asserts that a graph with cycles cannot have a topological sort
fn topological_sort_cycles() {
    let n0: BTreeSet<usize> = vec![1, 4].into_iter().collect();
    let n1: BTreeSet<usize> = vec![2].into_iter().collect();
    let n2: BTreeSet<usize> = vec![3].into_iter().collect();
    let n3: BTreeSet<usize> = vec![3].into_iter().collect();
    let n4: BTreeSet<usize> = vec![3].into_iter().collect();
    let n5: BTreeSet<usize> = vec![].into_iter().collect();

    let graph = vec![n0, n1, n2, n3, n4, n5];
    match topological_sort(&graph) {
        Some(_) => panic!("A graph with cycles should not have a topological order"),
        None => (), // everything ok!
    }
}

#[test]
//Asserts the method len() returns the sum of terminal and non terminals
fn grammar_len() {
    let g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert_eq!(g.len(), 0);
    let terminals = vec!["[a-z]", "[A-Z]"];
    let non_terminals = vec!["LETTER_UP | LETTER_LO", "word letter | letter"];
    let names = vec!["LETTER_LO", "LETTER_UP", "letter", "word"];
    let g = Grammar::new(&terminals, &non_terminals, &names);
    assert_eq!(g.len(), 4);
}

#[test]
//Asserts the method len() returns the sum of terminal and non terminals
fn grammar_len_term() {
    let g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert_eq!(g.len(), 0);
    let terminals = vec!["[a-z]", "[A-Z]"];
    let non_terminals = vec!["LETTER_UP | LETTER_LO", "word letter | letter"];
    let names = vec!["LETTER_LO", "LETTER_UP", "letter", "word"];
    let g = Grammar::new(&terminals, &non_terminals, &names);
    assert_eq!(g.len_term(), 2);
}

#[test]
//Asserts the method len() returns the sum of terminal and non terminals
fn grammar_len_nonterm() {
    let g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert_eq!(g.len(), 0);
    let terminals = vec!["[a-z]", "[A-Z]"];
    let non_terminals = vec!["LETTER_UP | LETTER_LO"];
    let names = vec!["LETTER_LO", "LETTER_UP", "letter"];
    let g = Grammar::new(&terminals, &non_terminals, &names);
    assert_eq!(g.len_nonterm(), 1);
}

#[test]
//Asserts the method is_empty() works as expected
fn grammar_is_empty() {
    let g = Grammar::new(
        Vec::new().as_slice(),
        Vec::new().as_slice(),
        Vec::new().as_slice(),
    );
    assert!(g.is_empty());
    let terminals = vec!["[a-z]", "[A-Z]"];
    let non_terminals = vec!["LETTER_UP | LETTER_LO", "word letter | letter"];
    let names = vec!["LETTER_LO", "LETTER_UP", "letter", "word"];
    let g = Grammar::new(&terminals, &non_terminals, &names);
    assert!(!g.is_empty());
}

#[test]
//Asserts order and production correctness in a hand-crafted grammar
fn grammar_crafted() {
    let terminals = vec!["[a-z]", "[A-Z]"];
    let non_terminals = vec!["LETTER_UP | LETTER_LO", "word letter | letter"];
    let names = vec!["LETTER_LO", "LETTER_UP", "letter", "word"];
    let g = Grammar::new(&terminals, &non_terminals, &names);
    assert_eq!(g[0], "[a-z]");
    assert_eq!(g[1], "[A-Z]");
    assert_eq!(g[2], "LETTER_UP | LETTER_LO");
    assert_eq!(g[3], "word letter | letter");
}

#[test]
//Asserts that a grammar content can be parsed (without invoking the file)
fn grammar_parse_string() {
    let g = "grammar g;
    letter: LETTER_UP | LETTER_LO;
    word: word letter | letter;
    LETTER_UP: [A-Z];
    LETTER_LO: [a-z];";
    match Grammar::parse_string(g) {
        Ok(g) => {
            assert_eq!(g.len(), 4);
            assert_eq!(g.len_term(), 2);
            assert_eq!(g.len_nonterm(), 2);
        }
        Err(_) => panic!("Failed to parse grammar string"),
    }
}

#[test]
fn extract_literal() {
    let grammar = "LP: '(';\nRP: ')';\nrandom: '(' '(' Џɯɷ幫Ѩ䷘ ')' ')';\nfunction: fn '{' '}';";
    match Grammar::parse_string(grammar) {
        Ok(g) => {
            assert_eq!(g.len(), 6);
        }
        Err(_) => panic!("Failed to parse grammar"),
    }
}

#[test]
//Asserts that non-existent files returns error
fn parse_grammar_non_existent() {
    match Grammar::parse_grammar("./resources/ada_grammar.txt") {
        Ok(_) => panic!("Expected the file to not exist!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "IOError: No such file or directory (os error 2)"
        ),
    }
}

#[test]
//Asserts that the file is parsed correctly even with high number of escape chars
fn parse_highly_escaped() {
    match Grammar::parse_grammar("./resources/comment_rich_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 5);
        }
        Err(e) => panic!("{}", e.to_string()),
    }
}

#[test]
//Asserts that the fragment using non-terminals generates syntax error
fn parse_fragments_nonterminal() {
    match Grammar::parse_grammar("./resources/fragments_contains_nt.txt") {
        Ok(_) => panic!("Expected a syntax error!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Lexer rule DIGIT cannot reference Parser non-terminal digit"
        ),
    }
}

#[test]
//Asserts that the fragment using wrong naming generates syntax error
fn parse_fragments_lowercase_naming() {
    match Grammar::parse_grammar("./resources/fragments_case_err.txt") {
        Ok(_) => panic!("Expected a syntax error!"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Fragments should be uppercase: fragment digit: [0-9]+;"
        ),
    }
}

#[test]
//Asserts that the fragments are replaced correctly
fn parse_recursive_fragments() {
    match Grammar::parse_grammar("./resources/fragments_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.len(), 2);
        }
        Err(_) => panic!(),
    }
}

#[test]
//Asserts that a simple grammar is parsed correctly.
fn parse_simple_grammar_correctly() {
    match Grammar::parse_grammar(
        "./resources/simple_grammar\
    .txt",
    ) {
        Ok(g) => assert_eq!(
            g.len(),
            9,
            "Grammar was parsed correctly, but a different number of production was expected"
        ),
        Err(_) => panic!("Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that ALL the productions are parsed correctly
fn get_production() {
    match Grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => {
            assert_eq!(g.get("TEXT").unwrap(), "~[,\\n\\r\"]+");
            assert_eq!(g.get("STRING").unwrap(), "'\"'('\"\"'|~'\"')*'\"'");
            assert_eq!(g.get("csvFile").unwrap(), "hdr row+ ");
            assert_eq!(g.get("hdr").unwrap(), "row ");
            assert_eq!(g.get("row").unwrap(), "field (COMMA field)* CR? LF ");
            assert_eq!(g.get("field").unwrap(), "TEXT| STRING|");
        }
        Err(_) => panic!("Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that the order of the production is kept unchanged (between terminals and non-terminals)
//using the `at()` method
fn order_unchanged_at() {
    match Grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => {
            assert_eq!(g[0], "~[,\\n\\r\"]+");
            assert_eq!(g[1], "'\"'('\"\"'|~'\"')*'\"'");
            assert_eq!(g[2], "','");
            assert_eq!(g[3], "'\\r'");
            assert_eq!(g[4], "'\\n'");
            assert_eq!(g[5], "hdr row+ ");
            assert_eq!(g[6], "row ");
            assert_eq!(g[7], "field (COMMA field)* CR? LF ");
            assert_eq!(g[8], "TEXT| STRING|");
        }
        Err(_) => panic!("Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that the order of the production is kept unchanged by iterating terminals
fn order_iter_term() {
    match Grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => {
            let vec = g.iter_term().as_slice();
            //0, 1, 2 are replaced literals
            assert_eq!(vec[0], "~[,\\n\\r\"]+");
            assert_eq!(vec[1], "'\"'('\"\"'|~'\"')*'\"'");
        }
        Err(_) => panic!("Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that the order of the production is kept unchanged by iterating non-terminals
fn order_unchanged_iter_nonterm() {
    match Grammar::parse_grammar("./resources/simple_grammar.txt") {
        Ok(g) => {
            let vec = g.iter_nonterm().as_slice();
            assert_eq!(vec[0], "hdr row+ ");
            assert_eq!(vec[1], "row ");
            assert_eq!(vec[2], "field (COMMA field)* CR? LF ");
            assert_eq!(vec[3], "TEXT| STRING|");
        }
        Err(_) => panic!("Simple grammar failed to parse"),
    }
}

#[test]
//Asserts that a grammar is parsed and the actions extracted correctly.
//Simpler version with trivial body and single action.
fn parse_actions_simpler() {
    match Grammar::parse_grammar("./resources/lexer_actions_simpler.txt") {
        Ok(g) => {
            assert_eq!(
                *g.action("Skip").unwrap().iter().next().unwrap(),
                Action::SKIP
            );
            assert_eq!(
                *g.action("More").unwrap().iter().next().unwrap(),
                Action::MORE
            );
            assert_eq!(
                *g.action("PopMode").unwrap().iter().next().unwrap(),
                Action::POPMODE
            );
            assert_eq!(
                *g.action("TypeEmpty").unwrap().iter().next().unwrap(),
                Action::TYPE("".to_owned())
            );
            assert_eq!(
                *g.action("TypeFull").unwrap().iter().next().unwrap(),
                Action::TYPE("TypeName".to_owned())
            );
            assert_eq!(
                *g.action("ChannelEmpty").unwrap().iter().next().unwrap(),
                Action::CHANNEL("".to_owned())
            );
            assert_eq!(
                *g.action("ChannelFull").unwrap().iter().next().unwrap(),
                Action::CHANNEL("ChannelName".to_owned())
            );
            assert_eq!(
                *g.action("ModeEmpty").unwrap().iter().next().unwrap(),
                Action::MODE("".to_owned())
            );
            assert_eq!(
                *g.action("ModeFull").unwrap().iter().next().unwrap(),
                Action::MODE("ChannelName".to_owned())
            );
            assert_eq!(
                *g.action("PushModeEmpty").unwrap().iter().next().unwrap(),
                Action::PUSHMODE("".to_owned())
            );
            assert_eq!(
                *g.action("PushModeFull").unwrap().iter().next().unwrap(),
                Action::PUSHMODE("ChannelName".to_owned())
            );
        }
        Err(_) => panic!("grammar failed to parse"),
    }
}

#[test]
//Asserts that a grammar is parsed and the actions extracted correctly.
//Harder version with multiple actions and tricky -> productions
fn parse_actions_harder() {
    match Grammar::parse_grammar("./resources/lexer_actions_harder.txt") {
        Ok(g) => {
            assert!(g.action("Dashbrack").unwrap().is_empty());
            assert_eq!(g.action("Whitespace").unwrap().len(), 2);
            let mut ws_iter = g.action("Whitespace").unwrap().iter();
            assert_eq!(*ws_iter.next().unwrap(), Action::MORE);
            assert_eq!(
                *ws_iter.next().unwrap(),
                Action::CHANNEL("CHANNEL_NAME".to_owned())
            );
            assert_eq!(
                *g.action("Newline").unwrap().iter().next().unwrap(),
                Action::SKIP
            );
            assert_eq!(
                *g.action("Text").unwrap().iter().next().unwrap(),
                Action::MORE
            );
            assert_eq!(g.action_nth(0), g.action("Dashbrack").unwrap());
            assert_eq!(g.action_nth(1), g.action("Whitespace").unwrap());
            assert_eq!(g.action_nth(2), g.action("Newline").unwrap());
            assert_eq!(g.action_nth(3), g.action("Text").unwrap());
            assert_eq!(g.action("NONEXISTENT"), None);
        }
        Err(_) => panic!("grammar failed to parse"),
    }
}

#[test]
//Asserts that terminal productions are cleaned up of spaces and embedded productions
//this should be done also in recursive replacement of terminals
fn terminal_cleaned() {
    match Grammar::parse_grammar("./resources/lexer_actions_harder.txt") {
        Ok(g) => {
            assert_eq!(g.get("Dashbrack").unwrap(), "[a->b\\-\\]]+'->'|([ ]+)");
            assert_eq!(g.get("Newline").unwrap(), "('\\r''\\n'?|'\\n')");
        }
        Err(_) => panic!("grammar failed to parse"),
    }
}

#[test]
//Asserts that invalid lexer actions are reported as errors
fn invalid_lexer_actions() {
    match Grammar::parse_grammar("./resources/lexer_invalid_action.txt") {
        Ok(_) => panic!("Invalid lexer actions should not be able to parse correctly"),
        Err(e) => assert_eq!(e.to_string(), "SyntaxError: invalid action `channel(name`"),
    }
}

#[test]
//Asserts that the C ANTLR grammar is parsed correctly. This grammar is longer than the CSV one.
fn parse_c_grammar_correctly() {
    match Grammar::parse_grammar("./resources/c_grammar.txt") {
        Ok(g) => assert_eq!(
            g.len(),
            205,
            "Grammar was parsed correctly, but a different number of production was expected"
        ),
        Err(_) => panic!("C grammar failed to parse"),
    }
}

#[test]
//Asserts that cyclic rules like S->S; cannot be solved in the lexer
fn lexer_rules_cycles_err() {
    match Grammar::parse_grammar("./resources/lexer_cyclic.txt") {
        Ok(_) => panic!("expected a failure"),
        Err(e) => assert_eq!(
            e.to_string(),
            "SyntaxError: Lexer contains cyclic productions!"
        ),
    }
}
