use fnv::FnvHashMap;
use maplit::btreeset;
use std::collections::hash_map::Iter;
use std::collections::BTreeSet;

// from ANTLR grammar to a lexer friendly-one
mod dfa;
mod grammar_conversion;
mod nfa;
mod simulator;

use crate::error::ParseError;

pub use self::dfa::Dfa;
pub use self::nfa::Nfa;

/// Value associated with the epsilon character
const EPSILON_VALUE: usize = 0;

/// A Binary Search Tree.
#[derive(Clone)]
struct BSTree<T> {
    /// The value contained in the tree.
    value: T,
    /// The left child of the tree.
    left: Option<Box<BSTree<T>>>,
    /// The right child of the tree.
    right: Option<Box<BSTree<T>>>,
}

impl<T: std::fmt::Display> std::fmt::Display for BSTree<T> {
    /// Prints the JSON representation of the tree.
    ///
    /// Each node is formatted as `{val:XXX,left:{...},right:{...}}`.
    /// The `left:{...}` or `right:{...}` parts are omitted if missing.
    /// This method requires the type of the tree to implement the Display trait.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\"val\":\"{}\"", &self.value)?;
        if let Some(left) = &self.left {
            write!(f, ",\"left\":{}", *left)?;
        }
        if let Some(right) = &self.right {
            write!(f, ",\"right\":{}", *right)?;
        }
        write!(f, "}}")
    }
}

/// Table assigning unique numerical values to symbols.
///
/// Symbols are grouped by productions: if two or more symbols are **always** found in the same
/// set (or subset) in every production, they are assigned the same index. This effectively reduces
/// the amount of edges required to build the NFA/DFA.
///
/// As an example, the productions `'a'` and `[a-z]` can be split into `'a'` and `[b-z]`
/// because the symbols from `b` to `z` will result in the same move in the NFA/DFA (as there are
/// no other productions). So, each letter `'a'` in the input can be converted to a number,
/// let's say `1`, and each letter from `'b'` to `'z'` can be converted to `2`, effectively reducing
/// the possible inputs to two single values instead of 26.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolTable {
    // table (char, assigned number). table.len() is the number for ANY char not in table.
    table: FnvHashMap<char, usize>,
    // table for reverse lookup, given an ID prints the transition set (useful only for debug)
    reverse: FnvHashMap<usize, BTreeSet<char>>,
}

impl SymbolTable {
    /// Builds an empty symbol table.
    pub fn empty() -> SymbolTable {
        let mut reverse = FnvHashMap::default();
        reverse.insert(0, btreeset!['\u{03F5}']);
        reverse.insert(1, btreeset!['\u{233A}']); //any char not in alphabet
        SymbolTable {
            table: FnvHashMap::default(),
            reverse,
        }
    }

    /// Builds the symbol table given a set of sets.
    ///
    /// This function takes as input a set of sets, `symbols`.
    /// Each set represents a possible input for a production, for example the production
    /// `[a-z]*[a-zA-Z0-9]` will have two input sets `[a-z]` and `[a-zA-Z0-9]` whereas the
    /// production `a`* will have only `[a]`.
    ///
    /// The construction works by refining the input sets: given two sets `A` and `B` the
    /// intersection `A∩B` is removed from them and added as extra set. This continues until every
    /// intersection between every pair yields ∅.
    ///
    /// In the above example, the resulting input sets after refining will be `[a]`, `[b-z]` and
    /// `[A-Z0-9]`. This means that the two productions can be converted to
    /// `([a]|[b-z])*([a]|[b-z]|[A-Z0-9])` and `[a]*`.
    /// An unique number can be assigned to each of these sets reducing the DFA moves for each state
    /// from 62 to just 3.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use std::collections::BTreeSet;
    /// use wisent::lexer::SymbolTable;
    ///
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let set = vec![abc, bcd].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(set);
    ///
    /// assert_ne!(symbol.get('a'), symbol.get('d'));
    /// assert_eq!(symbol.get('b'), symbol.get('c'));
    /// ```
    pub fn new(symbols: BTreeSet<BTreeSet<char>>) -> SymbolTable {
        // Refinement is done by taking a set `A` and comparing against all others (called `B`).
        // Three new sets are created: `A/(A∩B)`, `B/(A∩B)` and `A∩B`. If  `A/(A∩B)` = `A∩B`
        // (so the original is unmodified) only the new B set is added to the next processing list
        // and current processing continues unmodified. Otherwise also the intersection is added to
        // the next processing list BUT the current processing continues with the new A set.
        // when the current list is emptied, A is pushed to the done set, and the algorithm restarts
        // with the next processing list.
        let mut todo = symbols.into_iter().collect::<Vec<_>>();
        let mut done = BTreeSet::new();
        while !todo.is_empty() {
            let mut set_a = todo.pop().unwrap();
            let mut new_todo = Vec::new();
            while let Some(set_b) = todo.pop() {
                let intersection = set_a.intersection(&set_b).cloned().collect::<BTreeSet<_>>();
                let new_b = set_b
                    .difference(&intersection)
                    .cloned()
                    .collect::<BTreeSet<_>>();
                // always push `b` because it must be processed with other sets
                if !new_b.is_empty() {
                    new_todo.push(new_b);
                }
                if set_a != intersection && !intersection.is_empty() {
                    let new_a = set_a
                        .difference(&intersection)
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    // need to split the current set: I can do one comparison at time so I continue
                    // with the unique part and push the intersection inside the `to do`
                    set_a = new_a;
                    new_todo.push(intersection);
                }
            }
            done.insert(set_a);
            todo = new_todo;
        }
        // assign indices: unique to the same and increase only if something has been inserted
        let mut table = FnvHashMap::default();
        let mut uniques = 1; // 0 reserved for epsilon
        let mut reverse = FnvHashMap::default();
        for set in done.into_iter() {
            let mut inserted = false;
            for symbol in &set {
                inserted |= table.insert(*symbol, uniques).is_none();
            }
            reverse.insert(uniques, set);
            if inserted {
                uniques += 1;
            }
        }
        reverse.insert(0, btreeset!['\u{03F5}']);
        reverse.insert(uniques, btreeset!['\u{233A}']); //any char not in alphabet
        SymbolTable { table, reverse }
    }

    /// Returns the number of unique ids assigned to the symbols.
    ///
    /// IDs are progressive, so this is also the size required for the transition table of any
    /// NFA/DFA. This ALWAYS includes the two reserved ids: "epsilon" and "any char not in the
    /// table".
    /// # Examples
    /// Basic usage:
    /// ```
    /// use std::collections::BTreeSet;
    /// use wisent::lexer::SymbolTable;
    ///
    /// let a = vec!['a'].into_iter().collect::<BTreeSet<_>>();
    /// let set = vec![a].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(set);
    ///
    /// assert_eq!(symbol.ids(), 3);
    /// ```
    pub fn ids(&self) -> usize {
        self.reverse.len()
    }

    /// Returns the value associated for a specific char.
    ///
    /// If the char is not inside the symbol table, a value is returned anyway: this value
    /// represents any the transaction to be applied to any character not in the table.
    /// This value is also equal to the number of the symbols inside the table (hence the reason
    /// why the table is immutable after construction).
    ///
    /// The epsilon character IDs is always 0.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use std::collections::BTreeSet;
    /// use wisent::lexer::SymbolTable;
    ///
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let set = vec![abc, bcd].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(set);
    ///
    /// let index_a = symbol.get('a');
    /// let index_not_in_table = symbol.get('ダ');
    ///
    /// assert_eq!(index_a, 1);
    /// assert_eq!(index_not_in_table, 4);
    /// ```
    pub fn get(&self, symbol: char) -> usize {
        match self.table.get(&symbol) {
            Some(val) => *val,
            None => self.reverse.len() - 1,
        }
    }

    /// Returns the value(s) associated with a specific set.
    ///
    /// A set can be associated to a single val up to a number of vals equal to the input set
    /// size.
    ///
    /// Also in this case, a special number will be assigned to those symbols not in the symbol
    /// table.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use std::collections::BTreeSet;
    /// use wisent::lexer::SymbolTable;
    ///
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let set = vec![abc, bcd].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(set);
    ///
    /// let input_set = vec!['b', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let mut encoded = symbol.get_set(&input_set).into_iter();
    ///
    /// assert!(encoded.next().is_some()); // [b, c] (or [d])
    /// assert!(encoded.next().is_some()); // [d] (or [b, c])
    /// assert!(encoded.next().is_none());
    /// ```
    /// This example assigns three IDs to the symbols: `[a]`, `[b, c]` and `[d]`.
    /// A set of `[b, d]` will return two values: the one for `[b, c]` and the one for `[d]`.
    pub fn get_set(&self, symbols: &BTreeSet<char>) -> BTreeSet<usize> {
        let mut ret = BTreeSet::new();
        for symbol in symbols {
            match self.table.get(symbol) {
                Some(val) => ret.insert(*val),
                None => ret.insert(self.reverse.len() - 1),
            };
        }
        ret
    }

    /// Returns all the values NOT associated with a specific set.
    ///
    /// This method returns values associated for negated sets, simulating cases like `[^a-z]`, or,
    /// in ANTLR syntax, `~[a-z]`.
    ///
    /// **NOTE**: It is expected the input set to contain only symbols included in the table, so the
    /// special number for all the symbols not in the table will always be returned.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use std::collections::BTreeSet;
    /// use wisent::lexer::SymbolTable;
    ///
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let set = vec![abc, bcd].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(set);
    ///
    /// let input_set = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let mut negated = symbol.get_negated(&input_set).into_iter();
    ///
    /// assert!(negated.next().is_some()); //a
    /// assert!(negated.next().is_some()); // IDs of "any other char"
    /// assert!(negated.next().is_none());
    /// ```
    /// This example assigns three IDs to the symbols: `[a]`, `[b, c]` and `[d]`.
    /// A set of `[b, c, d]`, corresponding to `[^bcd]` in regexp syntax, will return two
    /// values, the one for `[a]` and the one for any other char not in table.
    pub fn get_negated(&self, symbols: &BTreeSet<char>) -> BTreeSet<usize> {
        let mut accept = BTreeSet::new();
        for symbol in &self.table {
            // this fails if negating only "partial sets". however in a normal execution should
            // NEVER happen (the same set passed as construction time should be passed here)
            if !symbols.contains(symbol.0) {
                accept.insert(*symbol.1);
            }
        }
        accept.insert(self.reverse.len() - 1);
        accept
    }

    /// Returns an iterator over the underlying table
    pub fn iter(&self) -> Iter<char, usize> {
        self.table.iter()
    }

    /// Converts the current symbol table into an array of bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut retval = Vec::new();
        for (index, chars) in &self.reverse {
            if *index == 0 || *index == self.table.len() {
                continue; //don't insert epsilon or "NOT_IN_ALPHABET"
            }
            retval.extend(u32::to_le_bytes(*index as u32));
            let all_symbols = chars.iter().collect::<String>();
            let all_symbols_bytes = all_symbols.as_bytes();
            retval.extend(u32::to_le_bytes(all_symbols_bytes.len() as u32));
            retval.extend_from_slice(all_symbols_bytes);
        }
        retval
    }

    /// Contructs a symbol table from an array of bytes previously generated with
    /// [SymbolTable::as_u8].
    pub fn from_bytes(v: &[u8]) -> Result<Self, ParseError> {
        let malformed_err = "malformed symbol_table";
        if !v.is_empty() {
            let mut i = 0;
            let mut table = FnvHashMap::default();
            let mut reverse = FnvHashMap::default();
            while i < v.len() {
                let symbol_index_bytes: [u8; 4] = v
                    .get(i..i + 4)
                    .ok_or_else(|| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?
                    .try_into()
                    .map_err(|_| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?;
                i += 4;
                let symbol_index = u32::from_le_bytes(symbol_index_bytes) as usize;
                let chars_len_bytes: [u8; 4] = v
                    .get(i..i + 4)
                    .ok_or_else(|| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?
                    .try_into()
                    .map_err(|_| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?;
                i += 4;
                let chars_len = u32::from_le_bytes(chars_len_bytes) as usize;
                let string_bytes = v
                    .get(i..i + chars_len)
                    .ok_or_else(|| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?
                    .to_vec();
                i += chars_len;
                let symbol_set = String::from_utf8(string_bytes)
                    .map_err(|_| ParseError::DeserializeError {
                        message: malformed_err.to_string(),
                    })?
                    .chars()
                    .collect::<BTreeSet<_>>();
                symbol_set.iter().copied().for_each(|char| {
                    table.insert(char, symbol_index);
                });
                reverse.insert(symbol_index, symbol_set);
            }
            reverse.insert(0, btreeset!['\u{03F5}']);
            reverse.insert(table.len(), btreeset!['\u{233A}']); //any char not in alphabet
            Ok(SymbolTable { table, reverse })
        } else {
            Err(ParseError::DeserializeError {
                message: "empty symbol table".to_string(),
            })
        }
    }
}

/// An Interface for a lexing Finite State Machine.
pub trait Automaton {
    /// Returns true if the automaton is empty.
    ///
    /// An automaton is empty if there are no transitions, and, as such, it halts in the starting
    /// state.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, Nfa};
    ///
    /// let grammar = Grammar::new(&[], &[], &[]);
    /// let nfa = Nfa::new(&grammar);
    ///
    /// assert!(nfa.is_empty());
    /// ```
    fn is_empty(&self) -> bool;

    /// Returns the number of nodes in the automaton.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, Nfa};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    ///
    /// assert_eq!(nfa.nodes(), 7);
    /// ```
    fn nodes(&self) -> usize;

    /// Returns the number of edges in the automaton.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, Nfa};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    ///
    /// assert_eq!(nfa.edges(), 8)
    /// ```
    fn edges(&self) -> usize;

    /// Returns a graphviz dot representation of the automaton as string.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, Nfa};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    /// nfa.to_dot();
    /// ```
    fn to_dot(&self) -> String;

    /// Saves the graphviz dot representation of the automaton to the given file.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Automaton, Nfa};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let nfa = Nfa::new(&grammar);
    /// nfa.save_dot("/home/user/nfa.dot");
    /// ```
    fn save_dot(&self, path: &str) {
        std::fs::write(path, self.to_dot())
            .unwrap_or_else(|_| panic!("Unable to write file {}", path));
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ParseError;
    use crate::lexer::SymbolTable;
    use maplit::btreeset;

    #[test]
    fn symbol_table_single_set() {
        let set1 = btreeset! {'a'};
        let set = btreeset! {set1};
        let symbol = SymbolTable::new(set);
        assert_eq!(*symbol.table.get(&'a').unwrap(), 1);
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

    #[test]
    fn symbol_table_serialize_deserialize() -> Result<(), ParseError> {
        let set1 = btreeset! {'a', 'b', 'c'};
        let set2 = btreeset! {'b', 'c', 'd'};
        let set3 = btreeset! {'d', 'e'};
        let set = btreeset! {set1, set2, set3};
        let symbol = SymbolTable::new(set);
        let serialized = symbol.as_bytes();
        let deserialized = SymbolTable::from_bytes(&serialized)?;
        assert_eq!(symbol, deserialized);
        Ok(())
    }
}
