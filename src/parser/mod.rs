use crate::lexer::Token;
use serde::{Deserialize, Serialize};

mod conversion;
mod ll;
mod lr;
mod simulator;

pub use self::ll::{LLGrammar, LLParsingTable};
pub use self::lr::{LRGrammar, LRParsingTable};
pub use self::simulator::{LLParser, PullParser, PushParser};

/// Represents the empty value `ε` when appearing in the first set or follow
/// set.
pub const EPSILON_VAL: u32 = 0xFFFFFFFE;
/// Represents the end of line value `$` when appearing in the follow set.
pub const ENDLINE_VAL: u32 = 0xFFFFFFFD;

/// Symbols used internally by the parser generator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
enum ParserSymbol {
    /// A terminal production. Indexed using the original grammar terminal
    /// order.
    Terminal(u32),
    /// A non-terminal production.
    /// TODO: explain indexing when LR complete. This probably uses parsing
    /// table indexes.
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
            Self::ParserRule(arg0) => write!(f, "Rule({arg0})"),
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

#[cfg(test)]
mod tests {
    use crate::grammar::Grammar;

    /// Grammar 4.28 of the dragon book second edition.
    /// Page 217.
    /// Used in parser tests.
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

    /// Grammar of exercise 4.40 of the dragon book second edition.
    /// Page 244.
    /// Used in parser tests.
    pub(super) fn grammar_440() -> Grammar {
        let g = "e1 : e;
             e: e Plus t | t;
             t: t Star f | f;
             f: Lpar e Rpar | Id;
             Plus: '+';
             Star: '*';
             Lpar: '(';
             Rpar: ')';
             Id: [0123456789]+;";
        Grammar::parse_bootstrap(g).unwrap()
    }

    /// Grammar of exercise 4.54 of the dragon book second edition.
    /// Page 263.
    /// Used in parser tests.
    pub(super) fn grammar_454() -> Grammar {
        let g = "s1: s;
            s: c c;
            c: C c | D;
            C: 'c';
            D: 'd';";
        Grammar::parse_bootstrap(g).unwrap()
    }

    /// Grammar of exercise 4.58 of the dragon book second edition.
    /// Page 267.
    /// Used in parser tests.
    pub(super) fn grammar_458() -> Grammar {
        let g = "s1: s;
            s: A a D | B b D | A b E | B a E;
            a: C;
            b: C;
            A: 'a';
            B: 'b';
            C: 'c';
            D: 'd';
            E: 'e';";
        Grammar::parse_bootstrap(g).unwrap()
    }

    /// Grammar of exercise 4.61 of the dragon book second edition.
    /// Page 271.
    /// Used in parser tests.
    pub(super) fn grammar_461() -> Grammar {
        let g = "s1: s;
            s: l EQ r | r;
            l: STAR r | ID;
            r: l;
            EQ: '=';
            STAR: '*';
            ID: 'id';";
        Grammar::parse_bootstrap(g).unwrap()
    }
}
