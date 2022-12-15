use crate::grammar::{LexerOp, LexerRuleElement, Tree};
use crate::lexer::SymbolTable;
use rustc_hash::FxHashSet;
use std::collections::BTreeSet;

/// Two operands (Symbol and Accepting state) and a limited set of operators (*, AND, OR).
/// Used to build the canonical parse tree.
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum Literal {
    /// The input symbol (a single byte).
    Symbol(u32),
    /// The given accepting state.
    Acc(u32),
    /// Kleenee star unary operator `*`.
    KLEENE,
    /// Concatenation operator.
    AND,
    /// Alternation operator.
    OR,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Symbol(i) => write!(f, "{}", i),
            Literal::Acc(i) => write!(f, "ACC({})", i),
            Literal::KLEENE => write!(f, "*"),
            Literal::AND => write!(f, "&"),
            Literal::OR => write!(f, "|"),
        }
    }
}

/// Parse tree for the regex with only *, AND, OR. Thus removing + or ? or ^.
pub(super) type CanonicalTree = Tree<Literal>;

/// Checks if a parsing tree contains non-greedy productions
pub(super) fn is_nongreedy(node: &Tree<LexerRuleElement<char>>) -> bool {
    let mut nodes = vec![node];
    while let Some(node) = nodes.pop() {
        if *node.value() == LexerRuleElement::Operation(LexerOp::Qm) {
            let child = node.children().next().expect("? node must have a child");
            match child.value() {
                LexerRuleElement::Operation(LexerOp::Kleene)
                | LexerRuleElement::Operation(LexerOp::Qm)
                | LexerRuleElement::Operation(LexerOp::Pl) => return true,
                _ => (),
            }
        }
        nodes.extend(node.children());
    }
    false
}

/// Transform a regex extended parse tree to a canonical parse tree (i.e. a tree with only symbols,
/// the *any symbol* placeholder, concatenation, alternation, kleene star).
/// **note** that non-greediness of a rule is removed by this function, so it must be recorded
/// somewhere else beforehand.
pub(super) fn canonicalise(
    node: &Tree<LexerRuleElement<char>>,
    symtable: &SymbolTable,
) -> CanonicalTree {
    match node.value() {
        LexerRuleElement::CharSet(i) => {
            create_literal_node(symtable.symbols_ids(i), symtable.epsilon_id())
        }
        LexerRuleElement::AnyValue => {
            create_literal_node(symtable.any_value_id(), symtable.epsilon_id())
        }
        LexerRuleElement::Operation(op) => {
            match op {
                LexerOp::Not => {
                    create_literal_node(solve_negated(node, symtable), symtable.epsilon_id())
                }
                LexerOp::Or => {
                    let children = node.children().map(|c| canonicalise(c, symtable)).collect();
                    Tree::new_node(Literal::OR, children)
                }
                LexerOp::And => {
                    let children = node.children().map(|c| canonicalise(c, symtable)).collect();
                    Tree::new_node(Literal::AND, children)
                }
                LexerOp::Kleene => {
                    let child = node
                        .children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("* node must have a child node");
                    Tree::new_node(Literal::KLEENE, vec![child])
                }
                LexerOp::Qm => {
                    let child = node
                        .children()
                        .next()
                        .expect("? node must have a child node");
                    if *child.value() == LexerRuleElement::Operation(LexerOp::Kleene)
                        || *child.value() == LexerRuleElement::Operation(LexerOp::Qm)
                        || *child.value() == LexerRuleElement::Operation(LexerOp::Pl)
                    {
                        // non-greedy rule, just remove the ?
                        canonicalise(child, symtable)
                    } else {
                        let canonical_child = canonicalise(child, symtable);
                        let epsilon = create_literal_node(BTreeSet::new(), symtable.epsilon_id());
                        Tree::new_node(Literal::OR, vec![epsilon, canonical_child])
                    }
                }
                LexerOp::Pl => {
                    let child = node
                        .children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("+ node must have a child node");
                    let right = Tree::new_node(Literal::KLEENE, vec![child.clone()]);
                    Tree::new_node(Literal::AND, vec![child, right])
                }
            }
        }
    }
}

/// converts a set of IDs into a several nodes concatenated with the | operators.
/// (e.g. from `[a,b,c]` to `'a'|'b'|'c'`.
/// Returns epsilon if the set is empty.
fn create_literal_node(set: BTreeSet<u32>, epsilon_id: u32) -> CanonicalTree {
    if set.is_empty() {
        Tree::new_leaf(Literal::Symbol(epsilon_id))
    } else if set.len() == 1 {
        Tree::new_leaf(Literal::Symbol(set.into_iter().next().unwrap()))
    } else {
        let children = set
            .into_iter()
            .map(|val| Tree::new_leaf(Literal::Symbol(val)))
            .collect();
        Tree::new_node(Literal::OR, children)
    }
}

/// Returns the entire set of symbols used in a given tree.
pub(super) fn alphabet_from_node(root: &Tree<LexerRuleElement<char>>) -> FxHashSet<BTreeSet<char>> {
    let mut ret = FxHashSet::default();
    let mut todo_nodes = vec![root];
    while let Some(node) = todo_nodes.pop() {
        match node.value() {
            LexerRuleElement::CharSet(i) => {
                ret.insert(i.clone());
            }
            LexerRuleElement::AnyValue => {}
            LexerRuleElement::Operation(_) => {
                todo_nodes.extend(node.children());
            }
        }
    }
    ret
}

// Solve a node with a negated set, by returning the allowed set of literals.
// panics in case an operator different from OR or NOT is encountered.
fn solve_negated(node: &Tree<LexerRuleElement<char>>, symtable: &SymbolTable) -> BTreeSet<u32> {
    debug_assert!(*node.value() == LexerRuleElement::Operation(LexerOp::Not));
    let entire_alphabet = symtable.any_value_id();
    let mut descendant_alphabet = BTreeSet::new();
    let mut todo = node.children().collect::<Vec<_>>();
    while let Some(child) = todo.pop() {
        match child.value() {
            LexerRuleElement::CharSet(v) => {
                descendant_alphabet.extend(symtable.symbols_ids(v));
            }
            LexerRuleElement::AnyValue => {
                descendant_alphabet.extend(symtable.any_value_id());
            }
            LexerRuleElement::Operation(LexerOp::Or) => {
                todo.extend(child.children());
            }
            LexerRuleElement::Operation(LexerOp::Not) => {
                descendant_alphabet.extend(solve_negated(child, symtable));
            }
            _ => panic!("Operation not supported in a negated set"),
        }
    }
    entire_alphabet
        .difference(&descendant_alphabet)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::lexer::grammar_conversion::{
        alphabet_from_node, canonicalise, expand_literals, gen_precedence_tree, is_nongreedy,
        ProdToken,
    };
    use crate::lexer::{SymbolTable, Tree};
    use std::fmt::Write;

    /// encoded representation of a tree in form of string
    /// otherwise the formatted version takes a lot of space (macros too, given the tree generics)
    fn as_str<T: std::fmt::Display>(node: &Tree<T>) -> String {
        let mut string = String::new();
        let children = node.children().map(as_str).collect::<Vec<_>>();
        write!(&mut string, "{}", node.value()).unwrap();
        if !children.is_empty() {
            write!(&mut string, "[{}]", children.join(",")).unwrap();
        }
        string
    }

    #[test]
    fn canonical_tree_any() {
        let expr = "('a'*.)*'a'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[*[&[*[0],|[0,1]]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_plus() {
        let expr = "('a'*'b')+'a'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[&[&[*[0],1],*[&[*[0],1]]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_qm() {
        let expr = "('a'*'b')?'a'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[|[3,&[*[0],1]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_neg() {
        let expr = "~[ab]('a'|'c')";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[|[2,3],|[0,2]]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_double_negation() {
        let expr = "~(~[a-c])|'d'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "|[0,1]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_kleene() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.*'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.*?'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_plus() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.+'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.+?'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_qm() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.?'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.??'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn identify_literal_empty() {
        let char_literal = "''";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_basic_concat() {
        let char_literal = "'aaa'";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),OP(&)[VALUE([a]),VALUE([a])]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped() {
        let char_literal = "'a\\x24'";
        let tree = Tree {
            value: ProdToken::Id(char_literal),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),VALUE([$])]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_seq() {
        let unicode_seq = "'დოლორ'";
        let tree = Tree {
            value: ProdToken::Id(unicode_seq),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected =
            "OP(&)[VALUE([დ]),OP(&)[VALUE([ო]),OP(&)[VALUE([ლ]),OP(&)[VALUE([ო]),VALUE([რ])]]]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped_range() {
        let square = "[\\-a-d\\]]";
        let tree = Tree {
            value: ProdToken::Id(square),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([-]abcd])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_range() {
        let range = "'\\U16C3'..'\\u16C5'";
        let tree = Tree {
            value: ProdToken::Id(range),
            children: vec![],
        };
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([ᛃᛄᛅ])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_nongreedy_kleene() {
        let expr_greedy = "'a'.*'a'";
        let expr_nongreedy = "'a'.*?'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn identify_nongreedy_qm() {
        let expr_greedy = "'a'.?'a'";
        let expr_nongreedy = "'a'.??'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn identify_nongreedy_plus() {
        let expr_greedy = "'a'.+'a'";
        let expr_nongreedy = "'a'.+?'a'";
        let prec_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let prec_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        assert!(!is_nongreedy(&prec_greedy));
        assert!(is_nongreedy(&prec_nongreedy));
    }

    #[test]
    fn regex_with_unicode_literals() {
        // ANTLR does not support unicode literals in the grammar,
        // but this library does for convenience.
        let regex = "[あいうえお]|[アイウエオ]";
        let prec_tree = gen_precedence_tree(regex);
        let expected = "|[[あいうえお],[アイウエオ]]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_or_klenee() {
        let expr = "'a'|'b'*'c'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "|['a',&[*['b'],'c']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_klenee_par() {
        let expr = "'a'*('b'|'c')*'d'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[*['a'],*[|['b','c']]],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation_par() {
        let expr = "('a')~'b'('c')('d')'e'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[&[&['a',~['b']],'c'],'d'],'e']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation() {
        let expr = "'a'~'b''c'('d')";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[&['a',~['b']],'c'],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_qm_plus_klenee_par() {
        let expr = "'a'?('b'*|'c'*)+'d'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&[?['a'],+[|[*['b'],*['c']]]],'d']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_negation_klenee() {
        let expr = "~'a'*";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*[~['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_negation_klenee_or() {
        let expr = "~'a'*|'b'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "|[*[~['a']],'b']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_precedence_literal_negation_literal() {
        let expr = "'a'~'a'*'a'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&['a',*[~['a']]],'a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_nongreedy_negation() {
        let expr = "'a'~'a'*?'a'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[&['a',?[*[~['a']]]],'a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_kleene_qm() {
        let expr = "(('a')*)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_not_kleene_qm() {
        let expr = "(~('a')*)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*[~['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_qm_qm() {
        let expr = "(('a')?)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "?['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_simplify_plus_qm() {
        let expr = "(('a')+)?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "*['a']";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_dont_simplify_nongreedy() {
        let expr = "('a')+?";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "?[+['a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regression_disappearing_literal() {
        let expr = "'a'*~'b'*";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[*['a'],*[~['b']]]";
        assert_eq!(as_str(&prec_tree), expected);
    }
}
