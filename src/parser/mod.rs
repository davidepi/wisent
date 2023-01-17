mod conversion;
mod ll;

pub use self::ll::{first_follow, LL1ParsingTable};

/// Represents the empty value `ε` when appearing in the first set or follow set.
pub const EPSILON_VAL: u32 = 0xFFFFFFFE;
/// Represents the end of line value `$` when appearing in the follow set.
pub const ENDLINE_VAL: u32 = 0xFFFFFFFD;

/// Symbols used internally by the parser generator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ParserSymbol {
    /// A terminal production. Indexed using the original grammar terminal order.
    Terminal(u32),
    /// A non-terminal production.
    /// TODO: explain indexing when LR complete. This probably uses parsing table indexes.
    NonTerminal(u32),
    /// The empty production `ε`.
    Empty,
}

/// Grammar 4.28 of the dragon book. Page 217 on the second edition.
/// Used in parser tests.
#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;

    pub(super) fn grammar_428() -> Grammar {
        let g = "e : t e1;
             e1: Plus t e1 | ;
             t: f t1;
             t1: Star f t1 | ;
             f: Lpar e Rpar | Id;
             Plus: '+';
             Star: '*';
             Lpar: '(';
             Rpar: ')';
             Id: [0123456789]+;";
        Grammar::parse_bootstrap(g).unwrap()
    }
}
