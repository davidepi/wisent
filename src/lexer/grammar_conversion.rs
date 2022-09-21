use super::{SymbolTable, Tree};
use crate::grammar::Grammar;
use maplit::btreeset;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::iter::Peekable;
use std::str::Chars;

#[derive(Copy, Clone)]
/// Operands/Operators that can be found in a Regex along with their value and priority.
/// This struct is used to build the regex parse tree with the correct priority.
struct RegexOp<'a> {
    /// Type of operand/operator for the regex (for example concatenation, ?, *, (, id...).
    optype: OpType,
    /// Value of the operand as string slice. Used mostly for the various ID.
    value: &'a str,
    /// priority of the operator.
    priority: u8,
}

///Operators for a regex (and an operand, ID).
#[derive(PartialEq, Debug, Copy, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum OpType {
    /// Kleenee star `*`.
    KLEENE,
    /// Question mark `?`.
    QM,
    /// Plus sign `+`.
    PL,
    /// Left parenthesis `(`.
    LP,
    /// Right parenthesis `)`.
    RP,
    /// Not `~`.
    NOT,
    /// Alternation of two operands.
    OR,
    /// Concatenation of two operands.
    AND,
    /// An Operand.
    ID,
}

impl std::fmt::Display for RegexOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::KLEENE => write!(f, "*"),
            OpType::QM => write!(f, "?"),
            OpType::PL => write!(f, "+"),
            OpType::LP => write!(f, "("),
            OpType::RP => write!(f, ")"),
            OpType::NOT => write!(f, "~"),
            OpType::OR => write!(f, "|"),
            OpType::AND => write!(f, "&"),
            OpType::ID => write!(f, "ID"),
        }
    }
}

/// Extended Literal: exacltly like RegexOp but does not depend on the string slice
/// (because every set has been expanded to a single letter).
#[derive(PartialEq, Debug, Clone)]
enum ExLiteral<T> {
    Value(BTreeSet<T>),
    AnyValue,
    Operation(OpType),
}

impl<T: std::fmt::Display> std::fmt::Display for ExLiteral<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExLiteral::Value(i) => {
                let mut string = String::from("[");
                for charz in i {
                    write!(string, "{}", charz)?;
                }
                string.push(']');
                write!(f, "VALUE({})", string)
            }
            ExLiteral::AnyValue => write!(f, "ANY"),
            ExLiteral::Operation(tp) => write!(f, "OP({})", tp),
        }
    }
}

/// Two operands (Symbol and Accepting state) and a limited set of operators (*, AND, OR).
/// Used to build the canonical parse tree.
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum Literal {
    /// The input symbol (a single byte).
    Symbol(u32),
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

/// Parse tree for the regex operands, accounting for precedence.
type PrecedenceTree<'a> = Tree<RegexOp<'a>>;
/// Parse tree for the regex with only *, AND, OR. Thus removing + or ? or ^.
pub(super) type CanonicalTree = Tree<Literal>;

/// Generates a canonical tree from the lexer productions of a grammar.
///
/// A canonical tree is a tree with only * & and | operations. The SymbolTable contains the
/// alphabet of the grammar.
///
/// Returns also if each production is non-greedy
pub(super) fn canonical_trees(grammar: &Grammar) -> (Vec<CanonicalTree>, SymbolTable, Vec<bool>) {
    // Convert a grammar into a series of parse trees, then expand the sets
    let parse_trees = grammar
        .iter_term()
        .map(String::as_ref)
        .map(gen_precedence_tree)
        .map(expand_literals)
        .collect::<Vec<_>>();
    let nongreedy = parse_trees.iter().map(is_nongreedy).collect();
    // collect the alphabet for DFA
    let alphabet = parse_trees
        .iter()
        .flat_map(alphabet_from_node)
        .collect::<BTreeSet<_>>();
    let symtable = SymbolTable::new(alphabet);
    // convert the parse tree into a canonical one (not ? or +, only *)
    let canonical_trees = parse_trees
        .into_iter()
        .map(|x| canonicalise(x, &symtable))
        .collect::<Vec<_>>();
    (canonical_trees, symtable, nongreedy)
}

/// Creates a parse tree with correct precedence given the input regex.
///
/// This works similarly to the conversion to Reverse-Polish Notation: two stacks where to push
/// or pop (operators and operands) based on the encountered operators.
fn gen_precedence_tree(regex: &str) -> PrecedenceTree {
    let mut operands = Vec::new();
    let mut operators: Vec<RegexOp> = Vec::new();
    // first get a sequence of operands and operators
    let tokens = regex_to_operands(regex);
    // for each in the sequence do the following actions
    for operator in tokens {
        match operator.optype {
            //operators after operand with highest priority -> solve immediately
            //unless previous one was OpType::NOT
            OpType::KLEENE | OpType::QM | OpType::PL => {
                if let Some(top) = operators.last() {
                    if top.optype == OpType::NOT {
                        combine_nodes(&mut operands, &mut operators);
                    }
                }
                operators.push(operator);
                combine_nodes(&mut operands, &mut operators);
            }
            //operators before operand solve if precedent has higher priority and is not OpType::LP
            //then push current
            OpType::NOT | OpType::OR | OpType::AND => {
                if let Some(top) = operators.last() {
                    if top.priority > operator.priority && top.optype != OpType::LP {
                        combine_nodes(&mut operands, &mut operators);
                    }
                }
                operators.push(operator);
            }
            // left parenthesis: push. Will be resolved by a right parenthesis.
            OpType::LP => operators.push(operator),
            // right parenthesis: combine all the nodes until left parenthesis is found.
            OpType::RP => {
                while !operators.is_empty() && operators.last().unwrap().optype != OpType::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
            // id: push to stack
            OpType::ID => {
                let leaf = Tree::new_leaf(operator);
                operands.push(leaf);
            }
        }
    }
    //solve all remaining operators
    while !operators.is_empty() {
        combine_nodes(&mut operands, &mut operators);
    }
    operands.pop().unwrap()
}

/// Complementary to the function gen_precedence_tree, combines two nodes with the last operator
/// in the stack.
fn combine_nodes<'a>(operands: &mut Vec<PrecedenceTree<'a>>, operators: &mut Vec<RegexOp<'a>>) {
    let operator = operators.pop().unwrap();
    let children = if operator.optype == OpType::OR || operator.optype == OpType::AND {
        //binary operator
        let right = operands.pop().unwrap();
        let left = operands.pop().unwrap();
        vec![left, right]
    } else {
        // unary operator
        vec![operands.pop().unwrap()]
    };
    let node = Tree::new_node(operator, children);
    operands.push(node);
}

/// Transforms a precedence parse tree in a precedence parse tree where the groups like `[a-z]`
/// are expanded in `a | b | c ... | y | z`
fn expand_literals(node: PrecedenceTree) -> Tree<ExLiteral<char>> {
    match node.value.optype {
        OpType::ID => expand_literal_node(node.value.value),
        n => {
            let children = node.into_children().map(expand_literals).collect();
            Tree::new_node(ExLiteral::Operation(n), children)
        }
    }
}

/// Expands a single node containing sets like `[a-z]` in a set with all the simbols like
/// `{a, b, c, d....}`. Replace also the . symbol with the special placeholder to represent any
/// value and any eventual set with .
fn expand_literal_node(literal: &str) -> Tree<ExLiteral<char>> {
    if literal == "." {
        Tree::new_leaf(ExLiteral::AnyValue)
    } else {
        let mut charz = Vec::new();
        let mut iter = literal.chars();
        let start = iter.next().unwrap();
        let end;
        let mut last = '\x00';
        let mut is_set = false;
        if start == '[' {
            end = ']';
            is_set = true;
        } else {
            end = '\'';
        };
        // process all character between quotes or braces
        while let Some(char) = iter.next() {
            let mut pushme = char;
            if char == '\\' {
                //escaped char
                pushme = unescape_character(iter.next().unwrap(), &mut iter);
                last = pushme;
                charz.push(pushme);
            } else if start == '[' && char == '-' {
                //set in form a-z, A-Z, 0-9, etc..
                let from = last as u32 + 1;
                let until = iter.next().unwrap() as u32 + 1; //included
                for i in from..until {
                    pushme = std::char::from_u32(i).unwrap();
                    charz.push(pushme);
                }
            } else if char == end {
                //end of sequence
                break;
            } else {
                //normal char
                last = pushme;
                charz.push(pushme);
            }
        }
        //check possible range in form 'a'..'z', at this point I ASSUME this can be a literal only
        //and the syntax has already been checked.
        if let Some(_c @ '.') = iter.next() {
            iter.next();
            iter.next();
            is_set = true;
            let mut until_char = iter.next().unwrap();
            if until_char == '\\' {
                until_char = unescape_character(iter.next().unwrap(), &mut iter);
            }
            let from = last as u32 + 1;
            let until = until_char as u32 + 1;
            for i in from..until {
                charz.push(std::char::from_u32(i).unwrap());
            }
        }
        // set of characters will be transformed into an "or" by the Symbol table later
        if is_set || charz.is_empty() {
            Tree::new_leaf(ExLiteral::Value(charz.into_iter().collect()))
        } else {
            // concatenation instead must be addressed here
            let mut done = charz
                .into_iter()
                .map(|x| Tree::new_leaf(ExLiteral::Value(btreeset! {x})))
                .collect::<Vec<_>>();
            while done.len() > 1 {
                let right = done.pop().unwrap();
                let left = done.pop().unwrap();
                let new = Tree::new_node(ExLiteral::Operation(OpType::AND), vec![left, right]);
                done.push(new);
            }
            done.pop().unwrap()
        }
    }
}

/// Transforms escaped strings in the form "\\n" to the single character they represent '\n'.
/// Works also for unicode in the form "\\UXXXX".
///
/// **NOTE**: it does NOT work for every unicode character (as they can be up to \uXXXXXX) because
/// ANTLR grammars support only up to \u{FFFF}.
fn unescape_character<T: Iterator<Item = char>>(letter: char, iter: &mut T) -> char {
    match letter {
        'n' => '\n',
        'r' => '\r',
        'b' => '\x08',
        't' => '\t',
        'f' => '\x0C',
        'u' | 'U' => {
            //NOTE: ANTLR grammars support only the BMP(0th) plane. Max is \uXXXX
            //      So no escaped emojis :'(
            let digit3 = iter.next().unwrap().to_digit(16).unwrap();
            let digit2 = iter.next().unwrap().to_digit(16).unwrap();
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit3 << 12 | digit2 << 8 | digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        'x' | 'X' => {
            let digit1 = iter.next().unwrap().to_digit(16).unwrap();
            let digit0 = iter.next().unwrap().to_digit(16).unwrap();
            let code = digit1 << 4 | digit0;
            std::char::from_u32(code).unwrap()
        }
        _ => letter,
    }
}

/// Transforms a regexp in a sequence of operands or operators.
fn regex_to_operands(regex: &str) -> Vec<RegexOp> {
    let mut tokenz = Vec::<RegexOp>::new();
    let mut iter = regex.chars().peekable();
    let mut index = 0;
    while let Some(char) = iter.next() {
        let tp;
        let val;
        let priority;
        match char {
            '*' => {
                tp = OpType::KLEENE;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '|' => {
                tp = OpType::OR;
                val = &regex[index..index + 1];
                priority = 1;
            }
            '?' => {
                tp = OpType::QM;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '+' => {
                tp = OpType::PL;
                val = &regex[index..index + 1];
                priority = 4;
            }
            '~' => {
                tp = OpType::NOT;
                val = &regex[index..index + 1];
                priority = 3;
            }
            '(' => {
                tp = OpType::LP;
                val = &regex[index..index + 1];
                priority = 5;
            }
            ')' => {
                tp = OpType::RP;
                val = &regex[index..index + 1];
                priority = 5;
            }
            _ => {
                tp = OpType::ID;
                priority = 0;
                val = read_token(&regex[index..], char, &mut iter);
            }
        };
        if !tokenz.is_empty() && implicit_concatenation(&tokenz.last().unwrap().optype, &tp) {
            tokenz.push(RegexOp {
                optype: OpType::AND,
                value: "&",
                priority: 2,
            })
        }
        tokenz.push(RegexOp {
            optype: tp,
            value: val,
            priority,
        });
        index += val.len();
    }
    tokenz
}

//No clippy, this is not more readable.
#[allow(clippy::useless_let_if_seq)]
/// Returns the slice of the regexp representing the token as 'a' or 'a'..'b' or '[a-z]'.
fn read_token<'a>(input: &'a str, first: char, it: &mut Peekable<Chars>) -> &'a str {
    match first {
        '.' => &input[..1],
        '[' => {
            let counted = consume_counting_until(it, ']');
            &input[..counted + 2] //2 is to match [], plus all the bytes counted inside
        }
        '\'' => {
            let mut counted = consume_counting_until(it, '\'');
            let mut id = &input[..counted + 2]; //this is a valid literal. check for range ''..''
            if input.len() > counted + 5 && &input[(counted + 2)..(counted + 5)] == "..\'" {
                //this is actually a range ''..'' so first advance the iterator by 3 pos
                it.next();
                it.next();
                it.next();
                //then update the counter for the literal length by accounting also the new lit.
                counted += consume_counting_until(it, '\'');
                id = &input[..counted + 6];
            }
            id
        }
        _ => panic!("Unsupported literal {}", input),
    }
}

/// Given two tokens, `a` and `b` returns true if a concatenation is implied between the two.
/// For example if the tokens are `'a'` and `'b'` the result will be true, because the word `ab`
/// is the concatenation of the two letters.
/// If the tokens are `'a'` and `*` the result will be false as there is no implicit operator.
///
/// Refer to the following table for each rule:
/// - `/` = Not allowed.
/// - `✗` = Concatenation not required.
/// - `✓` = Concatenation required.
/// - Rows: parameter `last`
/// - Columns: parameter `current`
/// Operator `*` is valid also for `?` and `+`.
///
/// |last|`|`|`~`|`*`|`(`|`)`| ID|
///  |---|---|---|---|---|---|---|
///  |`|`|`/`|`/`|`/`|`/`|`/`|`✗`|
///  |`*`|`/`|`✓`|`/`|`✓`|`✗`|`✓`|
///  |`)`|`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
///  |`~`|`/`|`✗`|`/`|`✗`|`/`|`✗`|
///  |`(`|`/`|`✗`|`/`|`✗`|`✗`|`✗`|
///  |ID |`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
fn implicit_concatenation(last: &OpType, current: &OpType) -> bool {
    let last_is_kleene_family =
        *last == OpType::KLEENE || *last == OpType::PL || *last == OpType::QM;
    let not_lp_id = *current == OpType::LP || *current == OpType::ID || *current == OpType::NOT;
    (*last == OpType::ID || *last == OpType::RP || last_is_kleene_family) && not_lp_id
}

/// Consumes the input (accounting for escaped chars) until the `until` character is found.
/// Returns the number of bytes (u8) consumed.
fn consume_counting_until(it: &mut Peekable<Chars>, until: char) -> usize {
    let mut escapes = 0;
    let mut skipped = 0;
    for skip in it {
        if skip == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if skip == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
        skipped += skip.len_utf8();
    }
    skipped
}

/// Checks if a parsing tree contains non-greedy productions
fn is_nongreedy(node: &Tree<ExLiteral<char>>) -> bool {
    let mut nodes = vec![node];
    while let Some(node) = nodes.pop() {
        if node.value == ExLiteral::Operation(OpType::QM) {
            let child = node.children().next().expect("? node must have a child");
            match child.value {
                ExLiteral::Operation(OpType::KLEENE)
                | ExLiteral::Operation(OpType::QM)
                | ExLiteral::Operation(OpType::PL) => return true,
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
fn canonicalise(node: Tree<ExLiteral<char>>, symtable: &SymbolTable) -> CanonicalTree {
    match node.value {
        ExLiteral::Value(i) => create_literal_node(symtable.symbols_ids(&i), symtable.epsilon_id()),
        ExLiteral::AnyValue => create_literal_node(symtable.any_value_id(), symtable.epsilon_id()),
        ExLiteral::Operation(op) => {
            match op {
                OpType::NOT => {
                    create_literal_node(solve_negated(&node, symtable), symtable.epsilon_id())
                }
                OpType::OR => {
                    let children = node
                        .into_children()
                        .map(|c| canonicalise(c, symtable))
                        .collect();
                    Tree::new_node(Literal::OR, children)
                }
                OpType::AND => {
                    let children = node
                        .into_children()
                        .map(|c| canonicalise(c, symtable))
                        .collect();
                    Tree::new_node(Literal::AND, children)
                }
                OpType::KLEENE => {
                    let child = node
                        .into_children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("* node must have a child node");
                    Tree::new_node(Literal::KLEENE, vec![child])
                }
                OpType::QM => {
                    let child = node
                        .into_children()
                        .next()
                        .expect("? node must have a child node");
                    if child.value == ExLiteral::Operation(OpType::KLEENE)
                        || child.value == ExLiteral::Operation(OpType::QM)
                        || child.value == ExLiteral::Operation(OpType::PL)
                    {
                        // non-greedy rule, just remove the ?
                        canonicalise(child, symtable)
                    } else {
                        let canonical_child = canonicalise(child, symtable);
                        let epsilon = create_literal_node(BTreeSet::new(), symtable.epsilon_id());
                        Tree::new_node(Literal::OR, vec![epsilon, canonical_child])
                    }
                }
                OpType::PL => {
                    let child = node
                        .into_children()
                        .map(|c| canonicalise(c, symtable))
                        .next()
                        .expect("+ node must have a child node");
                    let right = Tree::new_node(Literal::KLEENE, vec![child.clone()]);
                    Tree::new_node(Literal::AND, vec![child, right])
                }
                n => panic!("Unexpected operation {}", n),
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

/// Returns the set of symbols used in a given tree.
fn alphabet_from_node(root: &Tree<ExLiteral<char>>) -> BTreeSet<BTreeSet<char>> {
    let mut ret = BTreeSet::new();
    let mut todo_nodes = vec![root];
    while let Some(node) = todo_nodes.pop() {
        match &node.value {
            ExLiteral::Value(i) => {
                ret.insert(i.clone());
            }
            ExLiteral::AnyValue => {}
            ExLiteral::Operation(_) => {
                todo_nodes.extend(node.children());
            }
        }
    }
    ret
}

// Solve a node with a negated set, by returning the allowed set of literals.
// panics in case an operator different from OR or NOT is encountered.
fn solve_negated(node: &Tree<ExLiteral<char>>, symtable: &SymbolTable) -> BTreeSet<u32> {
    debug_assert!(node.value == ExLiteral::Operation(OpType::NOT));
    let entire_alphabet = symtable.any_value_id();
    let mut descendant_alphabet = BTreeSet::new();
    let mut todo = node.children().collect::<Vec<_>>();
    while let Some(child) = todo.pop() {
        match &child.value {
            ExLiteral::Value(v) => {
                descendant_alphabet.extend(symtable.symbols_ids(v));
            }
            ExLiteral::AnyValue => {
                descendant_alphabet.extend(symtable.any_value_id());
            }
            ExLiteral::Operation(OpType::OR) => {
                todo.extend(child.children());
            }
            ExLiteral::Operation(OpType::NOT) => {
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
        alphabet_from_node, canonicalise, expand_literals, gen_precedence_tree, OpType, RegexOp,
    };
    use crate::lexer::{SymbolTable, Tree};
    use std::fmt::Write;

    use super::is_nongreedy;

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
        let alphabet = alphabet_from_node(&tree);
        let symtable = SymbolTable::new(alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[*[&[*[0],|[0,1]]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_plus() {
        let expr = "('a'*'b')+'a'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree);
        let symtable = SymbolTable::new(alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[&[&[*[0],1],*[&[*[0],1]]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_qm() {
        let expr = "('a'*'b')?'a'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree);
        let symtable = SymbolTable::new(alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[|[3,&[*[0],1]],0]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_neg() {
        let expr = "~[ab]('a'|'c')";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree);
        let symtable = SymbolTable::new(alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "&[|[2,3],|[0,2]]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_double_negation() {
        let expr = "~(~[a-c])|'d'";
        let tree = expand_literals(gen_precedence_tree(expr));
        let alphabet = alphabet_from_node(&tree);
        let symtable = SymbolTable::new(alphabet);
        let canonical_tree = canonicalise(tree, &symtable);
        let expected = "|[0,1]";
        assert_eq!(as_str(&canonical_tree), expected);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_kleene() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.*'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy);
        let symtable_greedy = SymbolTable::new(alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.*?'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy);
        let symtable_nongreedy = SymbolTable::new(alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_plus() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.+'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy);
        let symtable_greedy = SymbolTable::new(alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.+?'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy);
        let symtable_nongreedy = SymbolTable::new(alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn canonical_tree_ignores_nongreedy_qm() {
        // nongreedines is handled by the DFA and DFA simulator, not the grammar
        let expr_greedy = "'a'.?'a'";
        let tree_greedy = expand_literals(gen_precedence_tree(expr_greedy));
        let alphabet_greedy = alphabet_from_node(&tree_greedy);
        let symtable_greedy = SymbolTable::new(alphabet_greedy);
        let canonical_tree_greedy = canonicalise(tree_greedy, &symtable_greedy);
        let expr_nongreedy = "'a'.??'a'";
        let tree_nongreedy = expand_literals(gen_precedence_tree(expr_nongreedy));
        let alphabet_nongreedy = alphabet_from_node(&tree_nongreedy);
        let symtable_nongreedy = SymbolTable::new(alphabet_nongreedy);
        let canonical_tree_nongreedy = canonicalise(tree_nongreedy, &symtable_nongreedy);
        assert_eq!(canonical_tree_nongreedy, canonical_tree_greedy);
    }

    #[test]
    fn identify_literal_empty() {
        let char_literal = "''";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = char_literal;
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_basic_concat() {
        let char_literal = "'aaa'";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = char_literal;
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),OP(&)[VALUE([a]),VALUE([a])]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped() {
        let char_literal = "'a\\x24'";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = char_literal;
        let expanded_tree = expand_literals(tree);
        let expected = "OP(&)[VALUE([a]),VALUE([$])]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_seq() {
        let unicode_seq = "'დოლორ'";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = unicode_seq;
        let expanded_tree = expand_literals(tree);
        let expected =
            "OP(&)[VALUE([დ]),OP(&)[VALUE([ო]),OP(&)[VALUE([ლ]),OP(&)[VALUE([ო]),VALUE([რ])]]]]";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_escaped_range() {
        let square = "[\\-a-d\\]]";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = square;
        let expanded_tree = expand_literals(tree);
        let expected = "VALUE([-]abcd])";
        assert_eq!(as_str(&expanded_tree), expected);
    }

    #[test]
    fn identify_literal_unicode_range() {
        let range = "'\\U16C3'..'\\u16C5'";
        let mut tree = Tree {
            value: RegexOp {
                optype: OpType::ID,
                value: "",
                priority: 0,
            },
            children: vec![],
        };
        tree.value.value = range;
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
        let expected = "&[*['a'],&[*[|['b','c']],'d']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation_par() {
        let expr = "('a')~'b'('c')('d')'e'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&['a',&[~['b'],&['c',&['d','e']]]]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_negation() {
        let expr = "'a'~'b''c'('d')";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&['a',&[~['b'],&['c','d']]]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_correct_precedence_qm_plus_klenee_par() {
        let expr = "'a'?('b'*|'c'*)+'d'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&[?['a'],&[+[|[*['b'],*['c']]],'d']]";
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
        let expected = "&['a',&[*[~['a']],'a']]";
        assert_eq!(as_str(&prec_tree), expected);
    }

    #[test]
    fn regex_nongreedy_negation() {
        let expr = "'a'~'a'*?'a'";
        let prec_tree = gen_precedence_tree(expr);
        let expected = "&['a',&[?[*[~['a']]],'a']]";
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
