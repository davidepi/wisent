use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Iter;
use std::collections::BTreeSet;

mod conversion;
mod dfa;
mod simulator;
mod ureader;

pub use self::dfa::{Dfa, MultiDfa};
pub use self::simulator::{tokenize_file, tokenize_string, DfaSimulator, Token};
pub use self::ureader::UnicodeReader;

/// Table assigning unique numerical values (IDs) to symbols.
///
/// Symbols are grouped by productions: if two or more symbols are **always**
/// found in the same set (or subset) in every production, they are assigned the
/// same index. This effectively reduces the amount of edges required to build
/// the NFA/DFA. These groups are called equivalence classes.
///
/// As an example, the productions `'a'` and `[a-z]` can be split into `'a'` and
/// `[b-z]` because the symbols from `b` to `z` will result in the same move in
/// the NFA/DFA (as there are no other productions). So, each letter `'a'` in
/// the input can be converted to a number, let's say `1`, and each letter from
/// `'b'` to `'z'` can be converted to `2`, effectively reducing the possible
/// inputs to two single values instead of 26.
#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolTable {
    // table (char, assigned number). table.len() is the number for ANY char not in table.
    table: FxHashMap<char, u32>,
    // table for reverse lookup, given an ID prints the transition set (useful only for debug)
    reverse: FxHashMap<u32, BTreeSet<char>>,
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SymbolTable{{ {:?} }}", self.reverse)
    }
}

impl SymbolTable {
    /// Builds the symbol table given a set of sets.
    ///
    /// This function takes as input a set of sets, `symbols`.
    /// Each set represents a possible input for a production, for example the
    /// production `[a-z]*[a-zA-Z0-9]` will have two input sets `[a-z]` and
    /// `[a-zA-Z0-9]` whereas the production `a`* will have only `[a]`.
    ///
    /// The construction works by refining the input sets: given two sets `A`
    /// and `B` the intersection `A∩B` is removed from them and added as
    /// extra set. This continues until every intersection between every
    /// pair yields ∅.
    ///
    /// In the above example, the resulting input sets after refining will be
    /// `[a]`, `[b-z]` and `[A-Z0-9]`. This means that the two productions
    /// can be converted to `([a]|[b-z])*([a]|[b-z]|[A-Z0-9])` and `[a]*`.
    /// An unique number can be assigned to each of these sets reducing the DFA
    /// moves for each state from 62 to just 3.
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[abc, bcd]);
    ///
    /// assert_ne!(symbol.symbol_id('a'), symbol.symbol_id('d'));
    /// assert_eq!(symbol.symbol_id('b'), symbol.symbol_id('c'));
    /// ```
    pub fn new(symbols: &[BTreeSet<char>]) -> SymbolTable {
        // Refinement is done by taking a set `A` and comparing against all others
        // (called `B`). Three new sets are created: `A/(A∩B)`, `B/(A∩B)` and
        // `A∩B`. If  `A/(A∩B)` = `A∩B` (so the original is unmodified) only the
        // new B set is added to the next processing list and current processing
        // continues unmodified. Otherwise also the intersection is added to the
        // next processing list BUT the current processing continues with the new A set.
        // when the current list is emptied, A is pushed to the done set, and the
        // algorithm restarts with the next processing list.
        let mut todo = symbols.to_vec();
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
        // assign indices: unique to the same and increase only if something has been
        // inserted
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

    /// Returns multiple symbol tables joined into a single one.
    ///
    /// This method can be used to merge multiple symbol tables into one. Note
    /// that the value assigned to each symbol will be different between the
    /// joined version and the original one.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let sym_a = SymbolTable::new(&[abc]);
    /// let sym_b = SymbolTable::new(&[bcd]);
    ///
    /// assert_eq!(sym_a.symbol_id('a'), sym_a.symbol_id('c'));
    /// assert_eq!(sym_b.symbol_id('b'), sym_b.symbol_id('d'));
    ///
    /// let sym_joined = SymbolTable::join(&[sym_a, sym_b]);
    ///
    /// assert_ne!(sym_joined.symbol_id('a'), sym_joined.symbol_id('c'));
    /// assert_ne!(sym_joined.symbol_id('b'), sym_joined.symbol_id('d'));
    /// assert_eq!(sym_joined.symbol_id('b'), sym_joined.symbol_id('c'));
    /// ```
    pub fn join(tables: &[Self]) -> Self {
        let sets = tables
            .iter()
            .flat_map(|st| st.reverse.values())
            .cloned()
            .collect::<Vec<_>>();
        Self::new(&sets)
    }

    /// Returns the ID of a character not in the alphabet.
    ///
    /// This is a special ID, not contained in the symbol table, that represents
    /// any character not inside the symbol table.
    ///
    /// Its ID is the highest ID found in the symbol table + 1.
    pub fn not_in_alphabet_id(&self) -> u32 {
        self.reverse.len() as u32
    }

    /// Returns the ID of the epsilon symbol.
    ///
    /// The epsilon symbol represents a transaction without reading any other
    /// symbol. As such it cannot be inserted into the symbol table.
    /// However, it is needed for the DFA construction and the NFA
    /// construction and simulation. As such, this method provides the epsilon
    /// value for the current symbol table.
    ///
    /// This value is essentially an ID not used for any other symbol.
    pub fn epsilon_id(&self) -> u32 {
        self.reverse.len() as u32 + 1
    }

    /// Returns a set representing the entire set of IDs used by this symbol
    /// table.
    ///
    /// This set does not include the [Epsilon ID](SymbolTable::epsilon_id), as
    /// it is not considered a symbol.
    ///
    /// This method is useful to replace the "any character" symbol `.`.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let a = vec!['a'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[a]);
    ///
    /// assert_eq!(
    ///     symbol.any_value_id(),
    ///     [0, 1].into_iter().collect::<BTreeSet<_>>()
    /// );
    /// ```
    pub fn any_value_id(&self) -> BTreeSet<u32> {
        self.reverse
            .keys()
            .copied()
            .chain(std::iter::once(self.not_in_alphabet_id()))
            .collect()
    }

    /// Returns the amount of unique IDs assigned to the symbols.
    ///
    /// IDs are progressive, so this is also the size required for the
    /// transition table of any NFA/DFA. This ALWAYS includes the ID
    /// reserved for any character not in the alphabet, but DOES NOT count
    /// the epsilon ID. The motivation of this discrepancy is the fact that
    /// in the transition table for a DFA the "not in alphabet" transition
    /// is needed, wheread the epsilon transition is not. # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let a = vec!['a'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[a]);
    ///
    /// assert_eq!(symbol.ids(), 2);
    /// ```
    pub fn ids(&self) -> u32 {
        self.reverse.len() as u32 + 1
    }

    /// Returns the ID associated for a specific char.
    ///
    /// If the char is not inside the symbol table, an ID is returned anyway:
    /// this ID represents the transaction to be applied to [any character
    /// not in the table](SymbolTable::not_in_alphabet_id).
    /// This value is also equal to the number of the symbols inside the table
    /// (hence the reason why the table is immutable after construction).
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[abc, bcd]);
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
    /// A set can be associated to a single ID up to a number of IDs equal to
    /// the input set size.
    ///
    /// Also in this case, a [special number will be assigned to those symbols
    /// not in the symbol table](SymbolTable::not_in_alphabet_id).
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let abc = vec!['a', 'b', 'c'].into_iter().collect::<BTreeSet<_>>();
    /// let bcd = vec!['b', 'c', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[abc, bcd]);
    ///
    /// let input_set = vec!['b', 'd'].into_iter().collect::<BTreeSet<_>>();
    /// let mut encoded = symbol.symbols_ids(&input_set).into_iter();
    ///
    /// assert!(encoded.next().is_some()); // [b, c] (or [d])
    /// assert!(encoded.next().is_some()); // [d] (or [b, c])
    /// assert!(encoded.next().is_none());
    /// ```
    /// This example assigns three IDs to the symbols: `[a]`, `[b, c]` and
    /// `[d]`. A set of `[b, d]` will return two values: the one for `[b,
    /// c]` and the one for `[d]`.
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

    /// Converts a given ID to a list of chars
    ///
    /// [Epsilon](SymbolTable::epsilon_id) is represented with `ϵ` and
    /// [not in alphabet](SymbolTable::not_in_alphabet_id) is represented with
    /// `⊙`. If an ID that does not match anything in the symbol table,
    /// neither epsilon not not in alphabet is provided, this function
    /// returns `�`.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use wisent::lexer::SymbolTable;
    /// let a = vec!['a'].into_iter().collect::<BTreeSet<_>>();
    /// let symbol = SymbolTable::new(&[a]);
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
}

#[cfg(test)]
mod tests {
    use crate::lexer::SymbolTable;
    use maplit::btreeset;

    #[test]
    fn symbol_table_single_set() {
        let set1 = btreeset! {'a'};
        let symbol = SymbolTable::new(&[set1]);
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
        let symbol = SymbolTable::new(&[set1, set2, set3, set4, set5, set6]);
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
        let symbol = SymbolTable::new(&[set1, set2]);
        assert_eq!(symbol.ids(), 4);
        assert_eq!(symbol.symbol_id('e'), symbol.not_in_alphabet_id());
    }

    #[test]
    fn symbol_table_get_set() {
        let set1 = btreeset! {'a', 'b', 'c', 'd', 'e', 'f', 'g',};
        let set2 = btreeset! {'d', 'e', 'f'};
        let set3 = btreeset! {'f','g','h'};
        let symbol = SymbolTable::new(&[set1, set2, set3]);
        assert_ne!(symbol.symbol_id('f'), symbol.symbol_id('g'));
        assert_eq!(symbol.symbol_id('d'), symbol.symbol_id('e'));

        let retrieve1 = symbol.symbols_ids(&btreeset! {'a', 'b', 'c'});
        assert_eq!(retrieve1.len(), 1); //[a, b, c] have the same value so only [0] returned
        let retrieve2 = symbol.symbols_ids(&btreeset! {'d', 'e', 'f'});
        assert_eq!(retrieve2.len(), 2); //[d, e] [f] are the sets so [1, 2] is
                                        //[d, returned
    }

    #[test]
    fn symbol_table_join() {
        let set1 = btreeset! {'a', 'b', 'c'};
        let set2 = btreeset! {'b', 'c', 'd'};
        let set3 = btreeset! {'d', 'e'};
        let set4 = btreeset! {'e', 'f', 'g', 'h'};
        let set5 = btreeset! {'h', 'a'};
        let set6 = btreeset! {'h', 'c'};

        let sta = SymbolTable::new(&[set1, set3, set4, set5]); // [b, c] [d, e] [e, f, g] [h] [a]
        let stb = SymbolTable::new(&[set2, set6]); // [b, d] [h] [c]
        assert_eq!(sta.symbol_id('b'), sta.symbol_id('c'));
        assert_eq!(sta.symbol_id('f'), sta.symbol_id('g'));
        assert_eq!(stb.symbol_id('b'), stb.symbol_id('d'));
        let joined = SymbolTable::join(&[sta, stb]);

        assert_ne!(joined.symbol_id('b'), joined.symbol_id('c'));
        assert_ne!(joined.symbol_id('b'), joined.symbol_id('d'));
        assert_eq!(joined.symbol_id('f'), joined.symbol_id('g'));
    }
}
