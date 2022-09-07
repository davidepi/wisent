use rustc_hash::FxHashMap;
use std::collections::hash_map::Iter;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::path::Path;

// from ANTLR grammar to a lexer friendly-one
mod dfa;
mod grammar_conversion;
mod simulator;

use crate::error::ParseError;

pub use self::dfa::Dfa;
pub use self::simulator::{DfaSimulator, Utf8CharReader};

/// Trait used to represents various object in [Graphviz Dot notation](https://graphviz.org/).
pub trait GraphvizDot {
    /// Returns a graphviz dot representation of the object as string.
    /// # Examples
    /// Implementation of the [`Dfa`] class:
    /// ```
    /// use wisent::grammar::Grammar;
    /// use wisent::lexer::{Dfa, GraphvizDot};
    ///
    /// let grammar = Grammar::new(&["'a'", "'b'*"], &[], &["LETTER_A", "LETTER_B"]);
    /// let dfa = Dfa::new(&grammar);
    /// dfa.to_dot();
    /// ```
    fn to_dot(&self) -> String;

    /// Writes a graphviz dot representation to to file.
    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_dot())
    }
}

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

impl<T: std::fmt::Display> GraphvizDot for BSTree<T> {
    fn to_dot(&self) -> String {
        let mut retval = "digraph BST {\n".to_string();
        let mut next_id = 0;
        let mut nodes = vec![(self, next_id)];
        next_id += 1;
        while let Some((node, id)) = nodes.pop() {
            writeln!(retval, "    {}[label=\"{}\"];", id, node.value).unwrap();
            if let Some(left) = node.left.as_ref() {
                nodes.push((left.as_ref(), next_id));
                writeln!(retval, "    {}->{}", id, next_id).unwrap();
                next_id += 1;
            } else {
            }
            if let Some(right) = node.right.as_ref() {
                nodes.push((right.as_ref(), next_id));
                writeln!(retval, "    {}->{}", id, next_id).unwrap();
                next_id += 1;
            }
        }
        retval.push('}');
        retval
    }
}

/// Table assigning unique numerical values (IDs) to symbols.
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
    table: FxHashMap<char, u32>,
    // table for reverse lookup, given an ID prints the transition set (useful only for debug)
    reverse: FxHashMap<u32, BTreeSet<char>>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        let reverse = FxHashMap::default();
        SymbolTable {
            table: FxHashMap::default(),
            reverse,
        }
    }
}

impl SymbolTable {
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
    /// assert_ne!(symbol.symbol_id('a'), symbol.symbol_id('d'));
    /// assert_eq!(symbol.symbol_id('b'), symbol.symbol_id('c'));
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
        let mut table = FxHashMap::default();
        //let mut uniques = 1; // 0 reserved for epsilon transactions
        let mut uniques = 0;
        let mut reverse = FxHashMap::default();
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
        SymbolTable { table, reverse }
    }

    /// Returns the ID of a character not in the alphabet.
    ///
    /// This is a special ID, not contained in the symbol table, that represents any character not
    /// inside the symbol table.
    ///
    /// Its ID is the highest ID found in the symbol table + 1.
    pub fn not_in_alphabet_id(&self) -> u32 {
        self.reverse.len() as u32
    }

    /// Returns the ID of the epsilon symbol.
    ///
    /// The epsilon symbol represents a transaction without reading any other symbol. As such it
    /// cannot be inserted into the symbol table. However, it is needed for the DFA construction
    /// and the NFA construction and simulation. As such, this method provides the epsilon value
    /// for the current symbol table.
    ///
    /// Its ID is the highest ID found in the symbol table + 2.
    pub fn epsilon_id(&self) -> u32 {
        self.reverse.len() as u32 + 1
    }

    /// Returns the amount of unique IDs assigned to the symbols.
    ///
    /// IDs are progressive, so this is also the size required for the transition table of any
    /// NFA/DFA. This ALWAYS includes the ID reserved for any character not in the alphabet, but
    /// DOES NOT count the epsilon ID.
    /// The motivation of this discrepancy is the fact that in the transition table for a DFA the
    /// "not in alphabet" transition is needed, wheread the epsilon transition is not.
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
    /// assert_eq!(symbol.ids(), 2);
    /// ```
    pub fn ids(&self) -> u32 {
        self.reverse.len() as u32 + 1
    }

    /// Returns the ID associated for a specific char.
    ///
    /// If the char is not inside the symbol table, an ID is returned anyway: this ID
    /// represents the transaction to be applied to [any character not in the
    /// table](SymbolTable::not_in_alphabet_id).
    /// This value is also equal to the number of the symbols inside the table (hence the reason
    /// why the table is immutable after construction).
    ///
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
    /// let index_a = symbol.symbol_id('a');
    /// let index_not_in_table = symbol.symbol_id('ダ');
    ///
    /// assert_eq!(index_a, 0);
    /// assert_eq!(index_not_in_table, 3);
    /// ```
    pub fn symbol_id(&self, symbol: char) -> u32 {
        *self
            .table
            .get(&symbol)
            .unwrap_or(&self.not_in_alphabet_id())
    }

    /// Returns all the IDs associated with a specific set.
    ///
    /// A set can be associated to a single ID up to a number of IDs equal to the input set
    /// size.
    ///
    /// Also in this case, a [special number will be assigned to those symbols not in the symbol
    /// table](SymbolTable::not_in_alphabet_id).
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
    /// let mut encoded = symbol.symbols_ids(&input_set).into_iter();
    ///
    /// assert!(encoded.next().is_some()); // [b, c] (or [d])
    /// assert!(encoded.next().is_some()); // [d] (or [b, c])
    /// assert!(encoded.next().is_none());
    /// ```
    /// This example assigns three IDs to the symbols: `[a]`, `[b, c]` and `[d]`.
    /// A set of `[b, d]` will return two values: the one for `[b, c]` and the one for `[d]`.
    pub fn symbols_ids(&self, symbols: &BTreeSet<char>) -> BTreeSet<u32> {
        let mut ret = BTreeSet::new();
        for symbol in symbols {
            match self.table.get(symbol) {
                Some(val) => ret.insert(*val),
                None => ret.insert(self.not_in_alphabet_id()),
            };
        }
        ret
    }

    /// Returns all the IDs NOT associated with a specific set.
    ///
    /// This method returns values associated for negated sets, simulating cases like `[^a-z]`, or,
    /// in ANTLR syntax, `~[a-z]`.
    ///
    /// **NOTE**: It is expected the input set to contain only symbols included in the table, so
    /// [not in alphabet](SymbolTable::not_in_alphabet_id) will always be in the returned set.
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
    /// let mut negated = symbol.symbols_ids_negated(&input_set).into_iter();
    ///
    /// assert!(negated.next().is_some()); //a
    /// assert!(negated.next().is_some()); // IDs of "any other char"
    /// assert!(negated.next().is_none());
    /// ```
    /// This example assigns three IDs to the symbols: `[a]`, `[b, c]` and `[d]`.
    /// A set of `[b, c, d]`, corresponding to `[^bcd]` in regexp syntax, will return two
    /// values, the one for `[a]` and the one for any other char not in table.
    pub fn symbols_ids_negated(&self, symbols: &BTreeSet<char>) -> BTreeSet<u32> {
        let mut accept = BTreeSet::new();
        for symbol in &self.table {
            // this fails if negating only "partial sets". however in a normal execution should
            // NEVER happen (the same set passed as construction time should be passed here)
            if !symbols.contains(symbol.0) {
                accept.insert(*symbol.1);
            }
        }
        accept.insert(self.not_in_alphabet_id());
        accept
    }

    /// Converts a given ID to a list of chars
    ///
    /// [Epsilon](SymbolTable::epsilon_id) is represented with `ϵ` and
    /// [not in alphabet](SymbolTable::not_in_alphabet_id) is represented with `⊙`.
    /// If an ID that does not match anything in the symbol table, nor epsilon or not in alphabet
    /// is provided, this function returns `�`.
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
    /// assert_eq!(symbol.label(0), 'a'.to_string());
    /// assert_eq!(symbol.label(symbol.not_in_alphabet_id()), '⊙'.to_string());
    /// assert_eq!(symbol.label(symbol.epsilon_id()), 'ϵ'.to_string());
    /// assert_eq!(symbol.label(9999), '�'.to_string());
    /// ```
    pub fn label(&self, id: u32) -> String {
        match self.reverse.get(&id) {
            Some(val) => val.iter().copied().collect(),
            None => {
                if id == self.not_in_alphabet_id() {
                    '\u{2299}'.to_string() // ⊙
                } else if id == self.epsilon_id() {
                    '\u{03F5}'.to_string() // ϵ
                } else {
                    '\u{FFFD}'.to_string() // �
                }
            }
        }
    }

    /// Returns an iterator over the underlying table
    pub fn iter(&self) -> Iter<char, u32> {
        self.table.iter()
    }

    /// Converts the current symbol table into an array of bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut retval = Vec::new();
        for (index, chars) in &self.reverse {
            retval.extend(u32::to_le_bytes(*index));
            let all_symbols = chars.iter().collect::<String>();
            let all_symbols_bytes = all_symbols.as_bytes();
            retval.extend(u32::to_le_bytes(all_symbols_bytes.len() as u32));
            retval.extend_from_slice(all_symbols_bytes);
        }
        retval
    }

    /// Contructs a symbol table from an array of bytes previously generated with
    /// [SymbolTable::as_bytes].
    pub fn from_bytes(v: &[u8]) -> Result<Self, ParseError> {
        let malformed_err = "malformed symbol_table";
        if !v.is_empty() {
            let mut i = 0;
            let mut table = FxHashMap::default();
            let mut reverse = FxHashMap::default();
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
                let symbol_index = u32::from_le_bytes(symbol_index_bytes);
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
            Ok(SymbolTable { table, reverse })
        } else {
            Err(ParseError::DeserializeError {
                message: "empty symbol table".to_string(),
            })
        }
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
        assert_eq!(*symbol.table.get(&'a').unwrap(), 0);
        assert_eq!(symbol.ids(), 2);
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
        assert_eq!(symbol.symbol_id('f'), symbol.symbol_id('g'));
        assert_ne!(symbol.symbol_id('a'), symbol.symbol_id('b'));
    }

    #[test]
    fn symbol_table_empty() {
        let symbol = SymbolTable::default();
        assert_eq!(symbol.ids(), 1);
    }

    #[test]
    fn symbol_table_character_outside_alphabet() {
        let set1 = btreeset! {'a', 'b', 'c'};
        let set2 = btreeset! {'b', 'c', 'd'};
        let symbol = SymbolTable::new(btreeset! {set1, set2});
        assert_eq!(symbol.ids(), 4);
        assert_eq!(symbol.symbol_id('e'), symbol.not_in_alphabet_id());
    }

    #[test]
    fn symbol_table_get_set() {
        let set1 = btreeset! {'a', 'b', 'c', 'd', 'e', 'f', 'g',};
        let set2 = btreeset! {'d', 'e', 'f'};
        let set3 = btreeset! {'f','g','h'};
        let set = btreeset! {set1, set2, set3};
        let symbol = SymbolTable::new(set);
        assert_ne!(symbol.symbol_id('f'), symbol.symbol_id('g'));
        assert_eq!(symbol.symbol_id('d'), symbol.symbol_id('e'));

        let retrieve1 = symbol.symbols_ids(&btreeset! {'a', 'b', 'c'});
        assert_eq!(retrieve1.len(), 1); //[a, b, c] have the same value so only [0] returned
        let retrieve2 = symbol.symbols_ids(&btreeset! {'d', 'e', 'f'});
        assert_eq!(retrieve2.len(), 2); //[d, e] [f] are the sets so [1, 2] is returned
    }

    #[test]
    fn symbol_table_get_negated() {
        let set1 = btreeset! {'a', 'b', 'c'};
        let set2 = btreeset! {'b', 'c', 'd'};
        let symbol = SymbolTable::new(btreeset! {set1, set2});

        let negate_me = btreeset! {'b','c'};
        let negated = symbol.symbols_ids_negated(&negate_me);
        assert!(negated.contains(&symbol.symbol_id('a')));
        assert!(negated.contains(&symbol.symbol_id('d')));
        assert!(negated.contains(&symbol.symbol_id('㊈')));
        assert!(!negated.contains(&symbol.symbol_id('b')));
        assert!(!negated.contains(&symbol.symbol_id('c')));
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
