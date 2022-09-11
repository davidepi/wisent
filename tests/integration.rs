use wisent::error::ParseError;
use wisent::grammar::Grammar;
use wisent::lexer::Dfa;

const C_GRAMMAR: &str = include_str!("../resources/c_grammar.txt");

#[test]
fn save_and_retrieve_c_dfa() -> Result<(), ParseError> {
    let grammar = Grammar::parse_string(C_GRAMMAR)?;
    let dfa = Dfa::new(&grammar);
    let bytes = dfa.as_bytes();
    let deserialized = Dfa::from_bytes(&bytes)?;
    assert_eq!(dfa, deserialized);
    Ok(())
}
