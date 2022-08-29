use maplit::btreeset;

use super::{BSTree, SymbolTable};
use crate::grammar::Grammar;
use std::collections::BTreeSet;
use std::iter::{Enumerate, Peekable};
use std::str::Chars;

#[derive(Copy, Clone)]
/// Operands/Operators that can be found in a Regex along with their value and priority.
/// This struct is used to build the regex parse tree with the correct priority.
struct RegexOp<'a> {
    /// Type of operand/operator for the regex (for example concatenation, ?, *, (, id...).
    r#type: OpType,
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
enum ExLiteral {
    Value(BTreeSet<char>),
    AnyValue,
    Operation(OpType),
}

impl std::fmt::Display for ExLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExLiteral::Value(i) => {
                let mut string = String::from("[");
                for charz in i {
                    string.push(*charz);
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
    /// The input symbol (a single letter. The value is the number in the symbol table).
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
type PrecedenceTree<'a> = BSTree<RegexOp<'a>>;
/// Parse tree for the regex without sets (only single letters).
type ExpandedPrecedenceTree = BSTree<ExLiteral>;
/// Parse tree for the regex with only *, AND, OR. Thus removing + or ? or ^.
pub(super) type CanonicalTree = BSTree<Literal>;

/// Generates a canonical tree from the lexer productions of a grammar.
///
/// A canonical tree is a tree with only * & and | operations. The SymbolTable contains the
/// alphabet of the grammar.
pub(super) fn canonical_trees(grammar: &Grammar) -> (Vec<CanonicalTree>, SymbolTable) {
    // Convert a grammar into a series of parse trees, then expand the sets
    let parse_trees = grammar
        .iter_term()
        .map(String::as_ref)
        .map(gen_precedence_tree)
        .map(expand_literals)
        .collect::<Vec<_>>();
    // collect the alphabet for DFA
    let alphabet = parse_trees
        .iter()
        .flat_map(get_set_of_symbols)
        .collect::<BTreeSet<_>>();
    let symtable = SymbolTable::new(alphabet);
    // convert the parse tree into a canonical one (not ? or +, only *)
    let canonical_trees = parse_trees
        .into_iter()
        .map(|x| canonicalise(x, &symtable))
        .collect::<Vec<_>>();
    (canonical_trees, symtable)
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
        match operator.r#type {
            //operators after operand with highest priority -> solve immediately
            OpType::KLEENE | OpType::QM | OpType::PL => {
                operators.push(operator);
                combine_nodes(&mut operands, &mut operators);
            }
            //operators before operand solve if precedent has higher priority and not (
            //then push current
            OpType::NOT | OpType::OR | OpType::AND => {
                if !operators.is_empty() {
                    let top = operators.last().unwrap();
                    if top.priority > operator.priority && top.r#type != OpType::LP {
                        combine_nodes(&mut operands, &mut operators);
                    }
                }
                operators.push(operator);
            }
            // left parenthesis: push. Will be resolved by a right parenthesis.
            OpType::LP => operators.push(operator),
            // right parenthesis: combine all the nodes until left parenthesis is found.
            OpType::RP => {
                while !operators.is_empty() && operators.last().unwrap().r#type != OpType::LP {
                    combine_nodes(&mut operands, &mut operators);
                }
                operators.pop();
            }
            // id: push to stack
            OpType::ID => {
                let leaf = BSTree {
                    value: operator,
                    left: None,
                    right: None,
                };
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
    let left;
    let right;
    if operator.r#type == OpType::OR || operator.r#type == OpType::AND {
        //binary operator
        right = Some(Box::new(operands.pop().unwrap()));
        left = Some(Box::new(operands.pop().unwrap()));
    } else {
        // unary operator
        left = Some(Box::new(operands.pop().unwrap()));
        right = None;
    };
    let ret = BSTree {
        value: operator,
        left,
        right,
    };
    operands.push(ret);
}

/// Transforms a precedence parse tree in a precedence parse tree where the groups like `[a-z]`
/// are expanded in `a | b | c ... | y | z`
fn expand_literals(node: PrecedenceTree) -> ExpandedPrecedenceTree {
    match node.value.r#type {
        OpType::ID => expand_literal_node(node.value.value),
        n => {
            let left = node.left.map(|l| Box::new(expand_literals(*l)));
            let right = node.right.map(|r| Box::new(expand_literals(*r)));
            BSTree {
                value: ExLiteral::Operation(n),
                left,
                right,
            }
        }
    }
}

/// Expands a single node containing sets like `[a-z]` in a set with all the simbols like
/// `{a, b, c, d....}`. Replace also the . symbol with the special placeholder to represent any
/// value and any eventual set with .
fn expand_literal_node(literal: &str) -> ExpandedPrecedenceTree {
    if literal == "." {
        BSTree {
            value: ExLiteral::AnyValue,
            left: None,
            right: None,
        }
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
            BSTree {
                value: ExLiteral::Value(charz.into_iter().collect::<BTreeSet<_>>()),
                left: None,
                right: None,
            }
        } else {
            // concatenation instead must be performed here
            let mut done = charz
                .into_iter()
                .map(|x| BSTree {
                    value: ExLiteral::Value(btreeset! {x}),
                    left: None,
                    right: None,
                })
                .collect::<Vec<_>>();
            while done.len() > 1 {
                let right = Some(Box::new(done.pop().unwrap()));
                let left = Some(Box::new(done.pop().unwrap()));
                let concat = BSTree {
                    value: ExLiteral::Operation(OpType::AND),
                    left,
                    right,
                };
                done.push(concat);
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
    let mut iter = regex.chars().peekable().enumerate();
    while let Some((index, char)) = iter.next() {
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
        if !tokenz.is_empty() && implicit_concatenation(&tokenz.last().unwrap().r#type, &tp) {
            tokenz.push(RegexOp {
                r#type: OpType::AND,
                value: "&",
                priority: 2,
            })
        }
        tokenz.push(RegexOp {
            r#type: tp,
            value: val,
            priority,
        });
    }
    tokenz
}

//No clippy, this is not more readable.
#[allow(clippy::useless_let_if_seq)]
/// Returns the slice of the regexp representing the token as 'a' or 'a'..'b' or '[a-z]'.
fn read_token<'a>(input: &'a str, first: char, it: &mut Enumerate<Peekable<Chars>>) -> &'a str {
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
///  |`~`|`/`|`✗`|`/`|`✗`|`/`|`✗`|
///  |`*`|`/`|`/`|`/`|`✓`|`✗`|`✓`|
///  |`(`|`/`|`✗`|`/`|`✗`|`✗`|`✗`|
///  |`)`|`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
///  |ID |`✗`|`✓`|`✗`|`✓`|`✗`|`✓`|
fn implicit_concatenation(last: &OpType, current: &OpType) -> bool {
    let last_is_kleene_family =
        *last == OpType::KLEENE || *last == OpType::PL || *last == OpType::QM;
    let cur_is_lp_or_id = *current == OpType::LP || *current == OpType::ID;
    (last_is_kleene_family && cur_is_lp_or_id)
        || (*last == OpType::RP && (*current == OpType::NOT || cur_is_lp_or_id))
        || (*last == OpType::ID && (*current == OpType::NOT || cur_is_lp_or_id))
}

/// Consumes the input (accounting for escaped chars) until the `until` character is found.
fn consume_counting_until(it: &mut Enumerate<Peekable<Chars>>, until: char) -> usize {
    let mut escapes = 0;
    let mut skipped = 0;
    for skip in it {
        if skip.1 == until {
            if escapes % 2 == 0 {
                break;
            }
            escapes = 0;
        } else if skip.1 == '\\' {
            escapes += 1;
        } else {
            escapes = 0;
        }
        skipped += skip.1.len_utf8();
    }
    skipped
}

/// Transform a regex extended parse tree to a canonical parse tree (i.e. a tree with only symbols,
/// the *any symbol* placeholder, concatenation, alternation, kleene star).
fn canonicalise(node: ExpandedPrecedenceTree, symtable: &SymbolTable) -> CanonicalTree {
    match node.value {
        ExLiteral::Value(i) => set_to_literal_node(symtable.symbols_ids(&i), symtable.epsilon_id()),
        ExLiteral::AnyValue => set_to_literal_node(
            symtable.symbols_ids_negated(&BTreeSet::new()),
            symtable.epsilon_id(),
        ),
        ExLiteral::Operation(op) => {
            match op {
                OpType::NOT => {
                    //get the entire used alphabet for the only existing node
                    let used_symbols = get_set_of_symbols(&node.left.unwrap())
                        .into_iter()
                        .flatten()
                        .collect::<BTreeSet<_>>();
                    let negated = symtable.symbols_ids_negated(&used_symbols);
                    set_to_literal_node(negated, symtable.epsilon_id())
                }
                OpType::OR => {
                    let left = node.left.map(|l| Box::new(canonicalise(*l, symtable)));
                    let right = node.right.map(|r| Box::new(canonicalise(*r, symtable)));
                    BSTree {
                        value: Literal::OR,
                        left,
                        right,
                    }
                }
                OpType::AND => {
                    let left = node.left.map(|l| Box::new(canonicalise(*l, symtable)));
                    let right = node.right.map(|r| Box::new(canonicalise(*r, symtable)));
                    BSTree {
                        value: Literal::AND,
                        left,
                        right,
                    }
                }
                OpType::KLEENE => {
                    let left = node.left.map(|l| Box::new(canonicalise(*l, symtable)));
                    let right = node.right.map(|r| Box::new(canonicalise(*r, symtable)));
                    BSTree {
                        value: Literal::KLEENE,
                        left,
                        right,
                    }
                }
                OpType::QM => {
                    let left = Some(Box::new(BSTree {
                        value: Literal::Symbol(symtable.epsilon_id()),
                        left: None,
                        right: None,
                    }));
                    //it the node has a ? DEFINITELY it has only a left children
                    let right = Some(Box::new(canonicalise(*node.left.unwrap(), symtable)));
                    BSTree {
                        value: Literal::OR,
                        left,
                        right,
                    }
                }
                OpType::PL => {
                    //it the node has a + DEFINITELY it has only a left children
                    let left = canonicalise(*node.left.unwrap(), symtable);
                    let right = BSTree {
                        value: Literal::KLEENE,
                        left: Some(Box::new(left.clone())),
                        right: None,
                    };
                    BSTree {
                        value: Literal::AND,
                        left: Some(Box::new(left)),
                        right: Some(Box::new(right)),
                    }
                }
                n => panic!("Unexpected operation {}", n),
            }
        }
    }
}

fn set_to_literal_node(set: BTreeSet<u32>, epsilon_id: u32) -> CanonicalTree {
    if set.is_empty() {
        BSTree {
            value: Literal::Symbol(epsilon_id),
            left: None,
            right: None,
        }
    } else {
        let mut nodes = Vec::new();
        for val in set {
            let node = BSTree {
                value: Literal::Symbol(val),
                left: None,
                right: None,
            };
            nodes.push(node);
        }
        while nodes.len() > 1 {
            let right = nodes.pop().unwrap();
            let left = nodes.pop().unwrap();
            let node = BSTree {
                value: Literal::OR,
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
            };
            nodes.push(node);
        }
        nodes.pop().unwrap()
    }
}

/// Returns the set of symbols for a given tree.
fn get_set_of_symbols(root: &ExpandedPrecedenceTree) -> BTreeSet<BTreeSet<char>> {
    let mut ret = BTreeSet::new();
    let mut todo_nodes = vec![root];
    while let Some(node) = todo_nodes.pop() {
        match &node.value {
            ExLiteral::Value(i) => {
                ret.insert(i.clone());
            }
            ExLiteral::AnyValue => {}
            ExLiteral::Operation(_) => {
                //nothing to do with the "operation" itself, but this is the only non-leaf type node
                if let Some(left) = &node.left {
                    todo_nodes.push(left);
                }
                if let Some(right) = &node.right {
                    todo_nodes.push(right);
                }
            }
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use crate::lexer::grammar_conversion::{
        canonicalise, expand_literals, gen_precedence_tree, get_set_of_symbols, OpType, RegexOp,
    };
    use crate::lexer::{BSTree, SymbolTable};

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
            r#"{"val":"&","left":{"val":"*","left":{"val":"&","left":{"val":"*","left":{"val":"0"}},"right":{"val":"|","left":{"val":"0"},"right":{"val":"1"}}}},"right":{"val":"0"}}"#
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
            r#"{"val":"&","left":{"val":"&","left":{"val":"&","left":{"val":"*","left":{"val":"0"}},"right":{"val":"1"}},"right":{"val":"*","left":{"val":"&","left":{"val":"*","left":{"val":"0"}},"right":{"val":"1"}}}},"right":{"val":"0"}}"#
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
            r#"{"val":"&","left":{"val":"|","left":{"val":"3"},"right":{"val":"&","left":{"val":"*","left":{"val":"0"}},"right":{"val":"1"}}},"right":{"val":"0"}}"#
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
            r#"{"val":"&","left":{"val":"|","left":{"val":"2"},"right":{"val":"3"}},"right":{"val":"|","left":{"val":"0"},"right":{"val":"2"}}}"#
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
}
