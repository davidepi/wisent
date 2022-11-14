use clap::Parser;
use wisent::grammar::Grammar;
use wisent::lexer::Dfa;

/// Lexer generator using the ANTLR grammar syntax
///
/// Generates a JSON-encoded representation of a DFA. The DFA can be later used with the runtime
/// provided in this same project.
#[derive(Debug, Parser)]
#[command(author, version, about, long_about)]
struct Args {
    /// Path to the input ANTLR grammar
    input: String,
}

fn main() {
    let args = Args::parse();
    if let Ok(g) = Grammar::parse_grammar(&args.input) {
        let dfa = Dfa::new(&g);
        let json = serde_json::to_string(&dfa).expect("Could not serialize the generated DFA");
        println!("{}", json);
    } else {
        eprintln!("Failed to read grammar {}", &args.input);
    }
}
