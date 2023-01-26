use crate::lexer::Token;
use serde::{Deserialize, Serialize};

mod conversion;
mod ll;
mod simulator;

pub use self::ll::{LL1Grammar, LL1ParsingTable};
pub use self::simulator::{LLParser, PullParser, PushParser};

/// Represents the empty value `ε` when appearing in the first set or follow set.
pub const EPSILON_VAL: u32 = 0xFFFFFFFE;
/// Represents the end of line value `$` when appearing in the follow set.
pub const ENDLINE_VAL: u32 = 0xFFFFFFFD;

/// Symbols used internally by the parser generator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ParserSymbol {
    /// A terminal production. Indexed using the original grammar terminal order.
    Terminal(u32),
    /// A non-terminal production.
    /// TODO: explain indexing when LR complete. This probably uses parsing table indexes.
    NonTerminal(u32),
    /// The empty production `ε`.
    Empty,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ParseNode {
    ParserRule(u32),
    Terminal(Token),
}

impl std::fmt::Debug for ParseNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParserRule(arg0) => write!(f, "Rule({})", arg0),
            Self::Terminal(arg0) => {
                if arg0.mode == 0 {
                    write!(f, "Token({})@{}..{}", arg0.production, arg0.start, arg0.end)
                } else {
                    write!(
                        f,
                        "Token({}[MODE {}])@{}..{}",
                        arg0.production, arg0.mode, arg0.start, arg0.end
                    )
                }
            }
        }
    }
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
