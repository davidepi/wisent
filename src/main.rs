use clap::{Args, Parser, Subcommand};
use std::io::Write;
use wisent::error::ParseError;
use wisent::grammar::Grammar;
use wisent::lexer::MultiDfa;
use wisent::parser::{LLGrammar, ENDLINE_VAL, EPSILON_VAL};

/// Actions to be performed by the executable
#[derive(Subcommand)]
enum Command {
    /// Generates the transition table and equivalence classes for the scanner/tokenizer.
    Scanner(ScannerSC),
    /// Generates the parsing table for the parser.
    Parser(ParserSC),
    /// Generates the Rust enums matching each production number with its name.
    Bridge(BridgeSC),
    /// Generates the list of First and Follow of the target grammar.
    FirstFollow(FirstFollowSC),
}

/// Parser generator supporting multiple parsing techniques and input syntax.
#[derive(Parser)]
#[command(author, version, about)]
#[command(propagate_version = true)]
struct Cli {
    /// The action to be performed.
    #[command(subcommand)]
    action: Command,
}

/// Generates a representation of a DFA. The DFA can be later used with the runtime
/// provided in the same crate.
///
/// By default the table is printed in binary format using bincode, but JSON output can be used
/// as well.
#[derive(Args)]
struct ScannerSC {
    /// Path to the input grammar.
    grammar: String,
    /// When printing the table, use JSON insted of binary.
    #[arg(short, long, default_value_t = false)]
    json: bool,
}

/// Prints the bridge code, assigning each token to its numeric representation.
///
/// The bridge code is not necessary for the parser to work correctly, but makes the code easier to
/// understand.
#[derive(Args)]
struct BridgeSC {
    /// Path to the input grammar.
    grammar: String,
    /// Name for the enum holding the token names.
    #[arg(short, long, default_value_t = String::from("GrammarToken"))]
    token_name: String,
    /// Name for the enum holding the production names.
    #[arg(short, long, default_value_t = String::from("GrammarProduction"))]
    prod_name: String,
}

/// Generates the table for a table-driven parser. The parser can be later used with the runtime
/// provided in the same crate.
///
/// By default the table is printed in binary format using bincode, but JSON output can be used
/// as well.
#[derive(Args)]
struct ParserSC {
    /// Path to the input grammar.
    grammar: String,
    /// When printing the table, use JSON insted of binary.
    #[arg(short, long, default_value_t = false)]
    json: bool,
    // Index of the start production in the grammar file. If not provided, the first
    // production in the file will be used.
    //#[arg(short, long)]
    //start_production: Option<String>,
}

/// Prints first set and follow set for a grammar. This can be useful to write a recursive descent
/// parser or to check for conflicts.
#[derive(Args)]
struct FirstFollowSC {
    /// Path to the input grammar.
    grammar: String,
    // Index of the start production in the grammar file. If not provided, the first
    // production in the file will be used.
    //#[arg(short, long)]
    //start_production: Option<String>,
}

fn main() {
    let args = Cli::parse();
    let res = match args.action {
        Command::Scanner(args) => scanner_task(args),
        Command::Parser(args) => parser_task(args),
        Command::Bridge(args) => bridge_task(args),
        Command::FirstFollow(args) => first_follow_task(args),
    };
    if let Err(e) = res {
        eprintln!("{}", e);
        std::process::exit(1)
    }
}

fn scanner_task(args: ScannerSC) -> Result<(), ParseError> {
    let grammar = Grammar::parse_grammar(&args.grammar)?;
    let dfa = MultiDfa::new(&grammar);
    if args.json {
        let json = serde_json::to_string(&dfa).expect("Could not serialize the generated DFA");
        println!("{}", json);
    } else {
        let data = bincode::serialize(&dfa).expect("Could not serialize the generated DFA");
        let mut stdout = std::io::stdout();
        stdout.write_all(&data).unwrap();
        stdout.flush().unwrap();
    }
    Ok(())
}

fn parser_task(args: ParserSC) -> Result<(), ParseError> {
    let grammar = Grammar::parse_grammar(&args.grammar)?;
    let llgrammar = LLGrammar::try_from(&grammar)?;
    let table = llgrammar.parsing_table()?;
    if args.json {
        let json = serde_json::to_string(&table).expect("Could not serialize the generated parser");
        println!("{}", json);
    } else {
        let data = bincode::serialize(&table).expect("Could not serialize the generated parser");
        let mut stdout = std::io::stdout();
        stdout.write_all(&data).unwrap();
        stdout.flush().unwrap();
    }
    Ok(())
}

/// Print a string with the first letter uppercase and all the other ones lowercase
fn to_camel_case(val: &str) -> String {
    let mut iter = val.chars();
    let mut retval = String::new();
    if let Some(char) = iter.next() {
        retval.extend(char.to_uppercase());
    }
    while let Some(char) = iter.next() {
        if char == '_' {
            if let Some(next) = iter.next() {
                retval.extend(next.to_uppercase());
            }
        } else {
            retval.extend(char.to_lowercase())
        }
    }
    retval
}

fn bridge_task(args: BridgeSC) -> Result<(), ParseError> {
    let grammar = Grammar::parse_grammar(&args.grammar)?;
    let tokens = grammar.iter_term().map(|x| &x.head).collect::<Vec<_>>();
    let prods = grammar.iter_nonterm().map(|x| &x.head).collect::<Vec<_>>();
    println!("enum {} {{", &args.token_name);
    for (index, token) in tokens.into_iter().enumerate() {
        println!("    {} = {},", to_camel_case(token), index);
    }
    println!("}}\n");
    println!("enum {} {{", &args.prod_name);
    for (index, prod) in prods.into_iter().enumerate() {
        println!("    {} = {},", to_camel_case(prod), index);
    }
    println!("}}\n");
    Ok(())
}

fn first_follow_task(args: FirstFollowSC) -> Result<(), ParseError> {
    let grammar = Grammar::parse_grammar(&args.grammar)?;
    let llgrammar = LLGrammar::try_from(&grammar)?;
    let (firsts, follows) = llgrammar.first_follow();
    for (index, nonterminal) in grammar.iter_nonterm().enumerate() {
        print!("FIRST({}) = {{", nonterminal.head);
        let mut first_set = Vec::with_capacity(firsts[index].len());
        for &first in &firsts[index] {
            let first_name = match first {
                EPSILON_VAL => "ε".to_string(),
                ENDLINE_VAL => "$".to_string(),
                _ => grammar
                    .iter_term()
                    .nth(first as usize)
                    .unwrap()
                    .head
                    .clone(),
            };
            first_set.push(first_name);
        }
        first_set.sort_unstable();
        print!("{}", first_set.join(","));
        println!("}}");
        print!("FOLLOW({}) = {{", nonterminal.head);
        let mut follow_set = Vec::with_capacity(firsts[index].len());
        for &follow in &follows[index] {
            let follow_name = match follow {
                EPSILON_VAL => "ε".to_string(),
                ENDLINE_VAL => "$".to_string(),
                _ => grammar
                    .iter_term()
                    .nth(follow as usize)
                    .unwrap()
                    .head
                    .clone(),
            };
            follow_set.push(follow_name);
        }
        follow_set.sort_unstable();
        print!("{}", follow_set.join(","));
        println!("}}");
    }
    Ok(())
}
