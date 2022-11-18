#[macro_use]
extern crate criterion;

use criterion::Criterion;
use wisent::grammar::Grammar;
use wisent::lexer::{DfaSimulator, MultiDfa};

const C_GRAMMAR: &str = include_str!("../resources/c_grammar.txt");
const C_EXAMPLE: &str = include_str!("../resources/c_example.txt");

fn parse_c_grammar(c: &mut Criterion) {
    c.bench_function("c grammar [parse grammar]", |b| {
        b.iter(|| Grammar::parse_antlr(C_GRAMMAR))
    });
}

fn lex_c_grammar(c: &mut Criterion) {
    let g = Grammar::parse_antlr(C_GRAMMAR).unwrap();
    c.bench_function("c grammar [build dfa]", |b| b.iter(|| MultiDfa::new(&g)));
}

fn tokenize_c_file(c: &mut Criterion) {
    let g = Grammar::parse_antlr(C_GRAMMAR).unwrap();
    let dfa = MultiDfa::new(&g);
    c.bench_function("c simulation [tokenizing]", |b| {
        b.iter(|| DfaSimulator::new(&dfa).tokenize(C_EXAMPLE.chars()))
    });
}

criterion_group!(benches, parse_c_grammar, lex_c_grammar, tokenize_c_file);
criterion_main!(benches);
