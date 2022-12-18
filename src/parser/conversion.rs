/// Terminal, Non-Terminal, And and Or only. No strings, allowed, only IDs.
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
enum CanonicalParserRuleElement {
    Terminal(u32),
    NonTerminal(u32),
    AND,
    OR,
}
