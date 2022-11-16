use wisent::error::ParseError;
use wisent::grammar::Grammar;
use wisent::lexer::{Dfa, DfaSimulator};

const C_GRAMMAR: &str = include_str!("../resources/c_grammar.txt");

/* temporary disabled
#[test]
fn save_and_retrieve_c_dfa() -> Result<(), ParseError> {
    let grammar = Grammar::parse_antlr(C_GRAMMAR)?;
    let dfa = Dfa::new(&grammar);
    let bytes = dfa.as_bytes();
    let deserialized = Dfa::from_bytes(&bytes)?;
    assert_eq!(dfa, deserialized);
    Ok(())
}

#[test]
fn match_longest() -> Result<(), ParseError> {
    let g4 = "INT : [0-9]+ ;
              DOT : '.' ; // match period
              FLOAT : [0-9]+ '.' ; // match FLOAT upon '34.' not INT then DOT";
    let grammar = Grammar::parse_antlr(g4)?;
    let dfa = Dfa::new(&grammar);
    let input = "34.";
    let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].production, 2);
    Ok(())
}

#[test]
fn match_first() -> Result<(), ParseError> {
    let g4 = "DOC : '/**' .*? '*/
' ; // both rules match /** foo */, resolve to DOC
CMT : ' /* ' .*? ' */
' ;";
    let grammar = Grammar::parse_antlr(g4)?;
    let dfa = Dfa::new(&grammar);
let input = " /** foo */
";
    let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].production, 0);
    Ok(())
}

#[test]
fn match_nongreedy() -> Result<(), ParseError> {
let g4 = " /** Match anything except \n inside of double angle brackets */
STRING : '<<' ~'\n'*? '>>' ; // Input '<<foo>>>>' matches STRING then END
              END    : '>>' ;";
    let grammar = Grammar::parse_antlr(g4)?;
    let dfa = Dfa::new(&grammar);
    let input = "<<foo>>>>";
    let tokens = DfaSimulator::new(&dfa).tokenize(input.chars());
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].production, 0);
    assert_eq!(tokens[1].production, 1);
    Ok(())
}

*/
