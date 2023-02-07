use crate::error::ParseError;
use crate::fxhashset;
use crate::grammar::Grammar;
use crate::parser::conversion::flatten;
use crate::parser::{ParserSymbol, ENDLINE_VAL, EPSILON_VAL};
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

/// A grammar without left-recursion for LL parsers.
///
/// The struct `LLGrammar` can be used to construct LL-based parsers. This
/// struct can be built from an original [`Grammar`] with the method
/// [`LLGrammar::try_from`] only if the original grammar is not left-recursive
/// and contains a rule for each nonterminal used.
///
/// This does not mean the grammar is LL(k), as there may be FIRST/FIRST or
/// FIRST/FOLLOW conflicts that prevents building a parsing table.
#[derive(Clone)]
pub struct LLGrammar {
    token_names: Vec<String>,
    nonterminal_names: Vec<String>,
    nonterminals: Vec<Vec<Vec<ParserSymbol>>>,
    starting_rule: u32,
    first: Vec<FxHashSet<u32>>,
    follow: Vec<FxHashSet<u32>>,
}

impl TryFrom<&Grammar> for LLGrammar {
    type Error = ParseError;

    /// Converts a [`Grammar`] into a [`LLGrammar`].
    ///
    /// Returns [`ParseError::LLError`] if the grammar is left-recursive.
    /// Returns [`ParseError::SyntaxError`] if any production references
    /// undeclared rules.
    fn try_from(value: &Grammar) -> Result<Self, Self::Error> {
        let nonterminals = flatten(value)?;
        let starting_rule = value.starting_rule();
        let nonterminal_names = value
            .iter_nonterm()
            .map(|x| &x.head)
            .cloned()
            .collect::<Vec<_>>();
        if let Err(e) = check_left_recursion(&nonterminals) {
            let cycle = e
                .into_iter()
                .map(|x| nonterminal_names[x as usize].clone())
                .collect::<Vec<_>>()
                .join(" -> ");
            Err(ParseError::LLError {
                message: format!("grammar is left recursive.\nLeft recursive productions: {cycle}"),
            })
        } else {
            let token_names = value
                .iter_term()
                .map(|x| &x.head)
                .cloned()
                .collect::<Vec<_>>();
            let first = first(&nonterminals);
            let follow = follow(&nonterminals, &first, starting_rule);
            Ok(LLGrammar {
                token_names,
                nonterminal_names,
                nonterminals,
                starting_rule,
                first,
                follow,
            })
        }
    }
}

impl LLGrammar {
    /// Returns the first sets and the follow sets of the grammar.
    ///
    /// These sets represent the set of terminals appearing respectively at the
    /// start and to the right (following) of a production. The terminals
    /// contained in each set are represented by their index in the grammar.
    /// Two special symbols can appear in these sets: [`EPSILON_VAL`] and
    /// [`ENDLINE_VAL`].
    ///
    /// The returned vectors are indexed by the nonterminal production. This
    /// means that `first[0]` contains the first set of the first
    /// nonterminal production appearing in the grammar using
    /// [`Grammar::iter_nonterm`].
    /// The first set of each terminal production is trivial, being equal to the
    /// terminal itself. For this reason, these are not contained in the
    /// returned vector.
    ///
    /// **Note:** this method does not raise errors if there are conflicts.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::{LLGrammar, ENDLINE_VAL};
    /// let grammar_text = "LetterA: 'a';
    ///                     LetterB: 'b';
    ///                     LetterC: 'c';
    ///                     s: LetterA s | LetterB s | LetterC;";
    /// let grammar = Grammar::parse_bootstrap(grammar_text).unwrap();
    /// let grammar_ll = LLGrammar::try_from(&grammar).unwrap();
    /// let (first, follow) = grammar_ll.first_follow();
    ///
    /// assert!(first[0].contains(&2));
    /// assert!(follow[0].contains(&ENDLINE_VAL));
    /// ```
    pub fn first_follow(&self) -> (&[FxHashSet<u32>], &[FxHashSet<u32>]) {
        (&self.first, &self.follow)
    }

    /// Computes the LL parsing table for the given grammar.
    ///
    /// This table can be used in a [Table Driven Parser](super::LLParser).
    ///
    /// Expects the grammar and the index of the starting non-terminal as input.
    ///
    /// Returns [`ParseError::LLError`] if there are FIRST/FIRST or FIRST/FOLLOW
    /// conflicts.
    ///
    /// #Examples
    /// Basic usage:
    /// ```
    /// # use wisent::grammar::Grammar;
    /// # use wisent::parser::LLGrammar;
    /// let g = "sum: num PLUS num;
    ///          num: INT | REAL;
    ///          INT: [0-9]+;
    ///          REAL: [0-9]+ '.' [0-9]+;
    ///          PLUS: '+';";
    /// let grammar = Grammar::parse_bootstrap(g).unwrap();
    /// let grammar_ll = LLGrammar::try_from(&grammar).unwrap();
    /// let ll1_table = grammar_ll.parsing_table().unwrap();
    /// ```
    pub fn parsing_table(&self) -> Result<LLParsingTable, ParseError> {
        if let Err(e) = check_first_first(&self.nonterminals, &self.first) {
            let name = &self.nonterminal_names[e as usize];
            Err(ParseError::LLError {
                message: format!("FIRST/FIRST conflict in production {name}"),
            })
        } else if let Err(e) = check_first_follow(&self.first, &self.follow) {
            let name = &self.nonterminal_names[e as usize];
            Err(ParseError::LLError {
                message: format!("FIRST/FOLLOW conflict in production {name}"),
            })
        } else {
            let table = ll1_parsing_table(
                &self.nonterminals,
                self.token_names.len() as u32,
                &self.first,
                &self.follow,
            );
            Ok(LLParsingTable {
                terminals: self.token_names.len() as u32,
                start_index: self.starting_rule,
                nonterminals: self.nonterminals.clone(),
                table,
            })
        }
    }
}

/// Calculates the first of a series of production.
/// This method invokes [first_single_alternative] until saturation.
pub(super) fn first(prods: &[Vec<Vec<ParserSymbol>>]) -> Vec<FxHashSet<u32>> {
    let mut old = vec![FxHashSet::default(); prods.len()];
    let mut saturated = false;
    while !saturated {
        let mut new = Vec::with_capacity(prods.len());
        for prod in prods {
            new.push(
                prod.iter()
                    .flat_map(|alternative| first_single_alternative(alternative, &old))
                    .collect::<FxHashSet<_>>(),
            );
        }
        if old == new {
            saturated = true;
        } else {
            old = new;
        }
    }
    old
}

/// Calculates the firsts of a single alternative.
/// For example in the grammar S: A | T; it can be run to calculate the firsts
/// of S: A; or S: T; It takes as input the first of all the productions,
/// calculated using the [first()] function.
fn first_single_alternative(prod: &[ParserSymbol], first: &[FxHashSet<u32>]) -> FxHashSet<u32> {
    let mut retval = FxHashSet::default();
    for concat in prod {
        let child_first = match concat {
            ParserSymbol::Terminal(t) => fxhashset!(*t),
            ParserSymbol::NonTerminal(nt) => first[*nt as usize].clone(),
            ParserSymbol::Empty => fxhashset!(EPSILON_VAL),
        };
        let has_epsilon = child_first.contains(&EPSILON_VAL);
        retval.extend(child_first);
        if !has_epsilon {
            break;
        }
    }
    retval
}

/// Calculates the follow of a series of productions.
pub(super) fn follow(
    prods: &[Vec<Vec<ParserSymbol>>],
    first: &[FxHashSet<u32>],
    start: u32,
) -> Vec<FxHashSet<u32>> {
    let mut old = vec![FxHashSet::default(); prods.len()];
    old[start as usize].insert(ENDLINE_VAL);
    let mut saturated = false;
    while !saturated {
        let mut new = old.clone();
        for (prod_id, prod) in prods.iter().enumerate() {
            follow_single(prod_id, prod, first, &mut new);
        }
        if old == new {
            saturated = true;
        } else {
            old = new;
        }
    }
    old
}

/// Calculates the follow of a single production. However takes as input the
/// modifiable set of all the productions. Part of the [follow()] functions,
/// should not be called by itself.
fn follow_single(
    prod_id: usize,
    prod: &[Vec<ParserSymbol>],
    first: &[FxHashSet<u32>],
    follow: &mut [FxHashSet<u32>],
) {
    for alternative in prod {
        // If there is a production A ⇒ αΒ, or a production A ⇒ αΒβ where FIRST(β)
        // contains ε, then everything in FOLLOW(A) is in FOLLOW(B).
        for child in alternative.iter().rev() {
            match child {
                ParserSymbol::NonTerminal(nt) => {
                    let cur_follow = follow[prod_id].iter().copied().collect::<Vec<_>>();
                    follow[*nt as usize].extend(cur_follow);
                    if !first[*nt as usize].contains(&EPSILON_VAL) {
                        break;
                    }
                }
                _ => break,
            }
        }
        // If there is a production A ⇒ αΒβ, then everything in FIRST(β), except for ε,
        // is placed in FOLLOW(B)
        let mut peekable = alternative.iter().peekable();
        while let Some(b) = peekable.next() {
            if let ParserSymbol::NonTerminal(nt) = b {
                if let Some(beta) = peekable.peek() {
                    match beta {
                        ParserSymbol::Terminal(t) => {
                            follow[*nt as usize].insert(*t);
                        }
                        ParserSymbol::NonTerminal(beta_nt) => {
                            let beta_first = first[*beta_nt as usize]
                                .iter()
                                .filter(|&&f| f != EPSILON_VAL);
                            follow[*nt as usize].extend(beta_first);
                        }
                        _ => (),
                    }
                }
            }
        }
    }
}

/// Calculates the LL(1) parsing table.
fn ll1_parsing_table(
    prods: &[Vec<Vec<ParserSymbol>>],
    term_no: u32,
    first: &[FxHashSet<u32>],
    follow: &[FxHashSet<u32>],
) -> Vec<Vec<Option<(u32, u32)>>> {
    let mut table = vec![vec![None; term_no as usize + 1]; prods.len()]; // +1 is the $ sign
    for (prod_id, prod) in prods.iter().enumerate() {
        for (alternative_id, alternative) in prod.iter().enumerate() {
            let first = first_single_alternative(&alternative[..], first);
            for &term in first.iter().filter(|&&e| e != EPSILON_VAL) {
                table[prod_id][term as usize] = Some((prod_id as u32, alternative_id as u32));
            }
            if first.contains(&EPSILON_VAL) {
                for &term in follow[prod_id].iter().filter(|&&e| e != EPSILON_VAL) {
                    if term != ENDLINE_VAL {
                        table[prod_id][term as usize] =
                            Some((prod_id as u32, alternative_id as u32));
                    } else {
                        table[prod_id][term_no as usize] =
                            Some((prod_id as u32, alternative_id as u32));
                    }
                }
            }
        }
    }
    table
}

/// Checks if the list of nonterminals is left recursive.
/// Returns an error containing the first recursion example
fn check_left_recursion(nonterminals: &[Vec<Vec<ParserSymbol>>]) -> Result<(), Vec<u32>> {
    for (tocheck, rhs) in nonterminals.iter().enumerate() {
        let tocheck = tocheck as u32;
        let mut todo = Vec::new();
        for alt in rhs {
            if let Some(ParserSymbol::NonTerminal(nt)) = alt.first() {
                todo.push((tocheck, *nt))
            }
        }
        let mut doing = vec![tocheck];
        while let Some((parent, nonterminal)) = todo.pop() {
            if nonterminal == tocheck {
                return Err(doing);
            }
            doing.push(nonterminal);
            let len = todo.len();
            for alt in &nonterminals[nonterminal as usize] {
                if let Some(ParserSymbol::NonTerminal(nt)) = alt.first() {
                    todo.push((nonterminal, *nt));
                }
            }
            if todo.len() == len {
                let keep_no = doing.iter().rfind(|&&x| x == parent).copied().unwrap_or(0) as usize;
                doing.truncate(keep_no);
            }
        }
    }
    Ok(())
}

// Check FIRST/FIRST conflicts:
// each subproduction of a nonterminal x must have disjoint first sets
fn check_first_first(
    nonterminals: &[Vec<Vec<ParserSymbol>>],
    first: &[FxHashSet<u32>],
) -> Result<(), u32> {
    for (prod, rhs) in nonterminals.iter().enumerate().filter(|(_, x)| x.len() > 1) {
        let mut alt_sets = rhs
            .iter()
            .map(|alt| first_single_alternative(alt, first))
            .collect::<Vec<_>>();
        let mut cumulative = alt_sets.pop().unwrap();
        while let Some(alt_set) = alt_sets.pop() {
            if cumulative.intersection(&alt_set).count() == 0 {
                cumulative.extend(alt_set);
            } else {
                return Err(prod as u32);
            }
        }
    }
    Ok(())
}

// Check FIRST/FOLLOW conflicts:
// if a production is nullable (has ε in the first set) FOLLOW(x) should be
// different from FIRST(x)
fn check_first_follow(first: &[FxHashSet<u32>], follow: &[FxHashSet<u32>]) -> Result<(), u32> {
    for prod in 0..first.len() {
        if first[prod].contains(&EPSILON_VAL) && first[prod].intersection(&follow[prod]).count() > 0
        {
            return Err(prod as u32);
        }
    }
    Ok(())
}

/// Stores the parsing table for a LL(1) parser.
///
/// This struct can be constructed using the [LL1Grammar::ll1_table] method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LLParsingTable {
    /// The amount of terminal productions in the grammar.
    terminals: u32,
    /// The starting symbol of the grammar represented by this table.
    start_index: u32,
    /// Stores the flattened grammar.
    nonterminals: Vec<Vec<Vec<ParserSymbol>>>,
    /// The parsing table.
    /// First dimension is the current nonterminal production.
    /// Second dimension is the read terminal/token or ENDLINE_VAL.
    /// The table value is the expanded nonterminal production and alternative
    /// (which or alternative is chosen).
    table: Vec<Vec<Option<(u32, u32)>>>,
}

impl LLParsingTable {
    /// The value assigned to the EOF character($) in the current table.
    pub(super) fn eof_val(&self) -> u32 {
        self.terminals
    }

    /// Returns the index of the starting nonterminal production in the original
    /// grammar represented by this table.
    pub(super) fn start(&self) -> u32 {
        self.start_index
    }

    /// Given the current nonterminal production and the next token value,
    /// retireves the next set of symbols.
    ///
    /// Returns None if the corresponding entry does not exist in the parsing
    /// table.
    pub(super) fn entry(&self, nonterminal: u32, terminal: u32) -> Option<&[ParserSymbol]> {
        let (prod, alternate) = self
            .table
            .get(nonterminal as usize)?
            .get(terminal as usize)?
            .as_ref()?;
        Some(&self.nonterminals[*prod as usize][*alternate as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::{check_first_first, check_left_recursion};
    use crate::fxhashset;
    use crate::grammar::Grammar;
    use crate::parser::conversion::flatten;
    use crate::parser::ll::{check_first_follow, first, follow, ll1_parsing_table};
    use crate::parser::tests::grammar_428;
    use crate::parser::{ENDLINE_VAL, EPSILON_VAL};

    #[test]
    fn first_set() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let first = first(&canonical);
        assert_eq!(first[0], fxhashset! {2, 4});
        assert_eq!(first[1], fxhashset! {0, EPSILON_VAL});
        assert_eq!(first[2], fxhashset! {2, 4});
        assert_eq!(first[3], fxhashset! {1, EPSILON_VAL});
        assert_eq!(first[4], fxhashset! {2, 4});
    }

    #[test]
    fn follow_set() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        assert_eq!(follow[0], fxhashset! {3, ENDLINE_VAL});
        assert_eq!(follow[1], fxhashset! {3, ENDLINE_VAL});
        assert_eq!(follow[2], fxhashset! {0, 3, ENDLINE_VAL});
        assert_eq!(follow[3], fxhashset! {0, 3, ENDLINE_VAL});
        assert_eq!(follow[4], fxhashset! {0, 1, 3, ENDLINE_VAL});
    }

    #[test]
    fn parsing_table() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        let table = ll1_parsing_table(&canonical, g.len_term() as u32, &first, &follow);
        let expected = [
            [None, None, Some((0, 0)), None, Some((0, 0)), None],
            [Some((1, 0)), None, None, Some((1, 1)), None, Some((1, 1))],
            [None, None, Some((2, 0)), None, Some((2, 0)), None],
            [Some((3, 1)), Some((3, 0)), None, Some((3, 1)), None, Some((3, 1))],
            [None, None, Some((4, 0)), None, Some((4, 1)), None],
        ];
        assert_eq!(table, expected);
    }

    #[test]
    fn left_recursive_direct() {
        let text = "s: A | s;
                    A: 'a';";
        let grammar = Grammar::parse_bootstrap(text).unwrap();
        let canonical = flatten(&grammar).unwrap();
        let left_rec = check_left_recursion(&canonical);
        match left_rec {
            Err(e) => assert_eq!(e, vec![0]),
            _ => panic!(),
        }
    }

    #[test]
    fn left_recursive_indirect() {
        let text = "s: a | b;
                    a: e | T;
                    b: c T;
                    c: s;
                    e: ;
                    T: 'a';";
        let grammar = Grammar::parse_bootstrap(text).unwrap();
        let canonical = flatten(&grammar).unwrap();
        let left_rec = check_left_recursion(&canonical);
        match left_rec {
            Err(e) => assert_eq!(e, vec![0, 2, 3]),
            _ => panic!(),
        }
    }

    #[test]
    fn non_left_recursive() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let left_rec = check_left_recursion(&canonical);
        assert!(left_rec.is_ok());
    }

    #[test]
    fn first_first_conflict() {
        let text = "s: x B | y C;
                    x: A;
                    y: A;
                    A: 'a';
                    B: 'b';
                    C: 'c';";
        let grammar = Grammar::parse_bootstrap(text).unwrap();
        let canonical = flatten(&grammar).unwrap();
        let first = first(&canonical);
        match check_first_first(&canonical, &first) {
            Err(e) => assert_eq!(e, 0),
            _ => panic!(),
        }
    }

    #[test]
    fn no_first_first_conflict() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let first = first(&canonical);
        assert!(check_first_first(&canonical, &first).is_ok());
    }

    #[test]
    fn first_follow_conflict() {
        let text = "s: a b;
                    a: F | ;
                    b: F;
                    F: 'f';";
        let grammar = Grammar::parse_bootstrap(text).unwrap();
        let canonical = flatten(&grammar).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        match check_first_follow(&first, &follow) {
            Err(e) => assert_eq!(e, 1),
            _ => panic!(),
        }
    }

    #[test]
    fn no_first_follow_conflict() {
        let g = grammar_428();
        let canonical = flatten(&g).unwrap();
        let first = first(&canonical);
        let follow = follow(&canonical, &first, 0);
        assert!(check_first_follow(&first, &follow).is_ok());
    }
}
