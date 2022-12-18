use crate::grammar::{LexerOp, LexerRuleElement, Tree};
use crate::lexer::SymbolTable;
use rustc_hash::FxHashSet;
use std::collections::BTreeSet;

/// Two operands (Symbol and Accepting state) and a limited set of operators (*, AND, OR).
/// Used to build the canonical parse tree.
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum CanonicalLexerRuleElement {
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

impl std::fmt::Display for CanonicalLexerRuleElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalLexerRuleElement::Symbol(i) => write!(f, "{}", i),
            CanonicalLexerRuleElement::Acc(i) => write!(f, "ACC({})", i),
            CanonicalLexerRuleElement::KLEENE => write!(f, "*"),
            CanonicalLexerRuleElement::AND => write!(f, "&"),
            CanonicalLexerRuleElement::OR => write!(f, "|"),
        }
    }
}

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
) -> Tree<CanonicalLexerRuleElement> {
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
                    Tree::new_node(CanonicalLexerRuleElement::OR, children)
                }
                LexerOp::And => {
                    let children = node.children().map(|c| canonicalise(c, symtable)).collect();
                    Tree::new_node(CanonicalLexerRuleElement::AND, children)
                }
                LexerOp::Kleene => {
                    let child = node
                        .children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("* node must have a child node");
                    Tree::new_node(CanonicalLexerRuleElement::KLEENE, vec![child])
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
                        Tree::new_node(
                            CanonicalLexerRuleElement::OR,
                            vec![epsilon, canonical_child],
                        )
                    }
                }
                LexerOp::Pl => {
                    let child = node
                        .children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("+ node must have a child node");
                    let right =
                        Tree::new_node(CanonicalLexerRuleElement::KLEENE, vec![child.clone()]);
                    Tree::new_node(CanonicalLexerRuleElement::AND, vec![child, right])
                }
            }
        }
    }
}

/// converts a set of IDs into a several nodes concatenated with the | operators.
/// (e.g. from `[a,b,c]` to `'a'|'b'|'c'`.
/// Returns epsilon if the set is empty.
fn create_literal_node(set: BTreeSet<u32>, epsilon_id: u32) -> Tree<CanonicalLexerRuleElement> {
    if set.is_empty() {
        Tree::new_leaf(CanonicalLexerRuleElement::Symbol(epsilon_id))
    } else if set.len() == 1 {
        Tree::new_leaf(CanonicalLexerRuleElement::Symbol(
            set.into_iter().next().unwrap(),
        ))
    } else {
        let children = set
            .into_iter()
            .map(|val| Tree::new_leaf(CanonicalLexerRuleElement::Symbol(val)))
            .collect();
        Tree::new_node(CanonicalLexerRuleElement::OR, children)
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
    use super::{alphabet_from_node, canonicalise};
    use crate::grammar::{LexerOp, LexerRuleElement, Tree};
    use crate::lexer::SymbolTable;
    use maplit::btreeset;
    use std::char;
    use std::collections::BTreeSet;
    use std::fmt::Write;

    /// encoded representation of a tree in form of string
    /// otherwise the formatted version takes a lot of space (macros too, given the tree generics)
    fn as_str<T: std::fmt::Display>(node: &Tree<T>) -> String {
        let mut string = String::new();
        let children = node.children().map(as_str).collect::<Vec<_>>();
        write!(&mut string, "{}", node.value()).unwrap();
        if !children.is_empty() {
            write!(&mut string, "({})", children.join(",")).unwrap();
        }
        string
    }

    // fast way of creating a tree. For tests only.
    fn from_str<T: Iterator<Item = char>>(chars: &mut T) -> Tree<LexerRuleElement<char>> {
        let op = chars.next().unwrap();
        match op {
            '&' => {
                assert_eq!(chars.next().unwrap(), '(');
                let left = from_str(chars);
                assert_eq!(chars.next().unwrap(), ',');
                let right = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![left, right])
            }
            '|' => {
                assert_eq!(chars.next().unwrap(), '(');
                let left = from_str(chars);
                assert_eq!(chars.next().unwrap(), ',');
                let right = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::Or), vec![left, right])
            }
            '*' => {
                assert_eq!(chars.next().unwrap(), '(');
                let child = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::Kleene), vec![child])
            }
            '?' => {
                assert_eq!(chars.next().unwrap(), '(');
                let child = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::Qm), vec![child])
            }
            '+' => {
                assert_eq!(chars.next().unwrap(), '(');
                let child = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::Pl), vec![child])
            }
            '~' => {
                assert_eq!(chars.next().unwrap(), '(');
                let child = from_str(chars);
                assert_eq!(chars.next().unwrap(), ')');
                Tree::new_node(LexerRuleElement::Operation(LexerOp::Not), vec![child])
            }
            '.' => Tree::new_leaf(LexerRuleElement::AnyValue),
            '[' => {
                let mut set = BTreeSet::new();
                for next in chars.by_ref() {
                    if next != ']' {
                        set.insert(next);
                    } else {
                        break;
                    }
                }
                Tree::new_leaf(LexerRuleElement::CharSet(set))
            }
            '\'' => {
                let mut seq = Vec::new();
                for next in chars.by_ref() {
                    if next != '\'' {
                        seq.push(next);
                    } else {
                        break;
                    }
                }
                if seq.len() <= 1 {
                    Tree::new_leaf(LexerRuleElement::CharSet(seq.into_iter().collect()))
                } else {
                    seq.reverse();
                    let left =
                        Tree::new_leaf(LexerRuleElement::CharSet(btreeset! {seq.pop().unwrap()}));
                    let right =
                        Tree::new_leaf(LexerRuleElement::CharSet(btreeset! {seq.pop().unwrap()}));
                    let op = LexerRuleElement::Operation(LexerOp::And);
                    let mut root = Tree::new_node(op, vec![left, right]);
                    while let Some(next) = seq.pop() {
                        let new = Tree::new_leaf(LexerRuleElement::CharSet(btreeset! {next}));
                        root = Tree::new_node(LexerRuleElement::Operation(LexerOp::And), vec![new]);
                    }
                    root
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn canonical_tree_any() {
        let tree = from_str(&mut "&(*(&(*('a'),.)),'a')".chars()); // ('a'*.)*'a'
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(&tree, &symtable);
        let expected = "&(*(&(*(0),|(0,1))),0)";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_plus() {
        let tree = from_str(&mut "&(+(&(*('a'),'b')),'a')".chars()); // ('a'*'b')+'a'
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(&tree, &symtable);
        let expected = "&(&(&(*(0),1),*(&(*(0),1))),0)";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_qm() {
        let tree = from_str(&mut "&(?(&(*('a'),'b')),'a')".chars()); // ('a'*'b')?'a'
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(&tree, &symtable);
        let expected = "&(|(3,&(*(0),1)),0)";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_neg() {
        let tree = from_str(&mut "&(~([ab]),|('a','c'))".chars()); // ~[ab]('a'|'c')
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(&tree, &symtable);
        let expected = "&(|(2,3),|(0,2))";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_double_negation() {
        let tree = from_str(&mut "|(~(~([abc])),'d')".chars()); // ~(~[a-c])|'d'
        let alphabet = alphabet_from_node(&tree).into_iter().collect::<Vec<_>>();
        let symtable = SymbolTable::new(&alphabet);
        let canonical_tree = canonicalise(&tree, &symtable);
        let expected = "|(0,1)";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_kleene() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let tree_greedy = from_str(&mut "&(&('a',*(.)),'a')".chars()); // 'a'.*'a'
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(&tree_greedy, &symtable_greedy);
        let tree_nongreedy = from_str(&mut "&(&('a',?(*(.))),'a')".chars()); // 'a'.*?'a'
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(&tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_plus() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let tree_greedy = from_str(&mut "&(&('a',+(.)),'a')".chars()); // 'a'.+'a'
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(&tree_greedy, &symtable_greedy);
        let tree_nongreedy = from_str(&mut "&(&('a',?(+(.))),'a')".chars()); // 'a'.+?'a'
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(&tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_qm() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let tree_greedy = from_str(&mut "&(&('a',?(.)),'a')".chars()); // 'a'.?'a'
        let alphabet_greedy = alphabet_from_node(&tree_greedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_greedy = SymbolTable::new(&alphabet_greedy);
        let canonical_tree_greedy = canonicalise(&tree_greedy, &symtable_greedy);
        let tree_nongreedy = from_str(&mut "&(&('a',?(?(.))),'a')".chars()); // 'a'.??'a'
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy)
            .into_iter()
            .collect::<Vec<_>>();
        let symtable_nongreedy = SymbolTable::new(&alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(&tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }
}
