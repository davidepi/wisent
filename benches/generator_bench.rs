#[macro_use]
extern crate criterion;

use criterion::{BatchSize, Criterion};
use wisent::grammar::Grammar;
use wisent::lexer::{Dfa, DfaSimulator};

fn parse_c_grammar(c: &mut Criterion) {
    c.bench_function("c grammar [parse grammar]", |b| {
        b.iter(|| Grammar::parse_grammar("./resources/c_grammar.txt"))
    });
}

fn lex_c_grammar(c: &mut Criterion) {
    let g = Grammar::parse_grammar("./resources/c_grammar.txt").unwrap();
    c.bench_function("c grammar [lex grammar]", |b| b.iter(|| Dfa::new(&g)));
}

fn tokenize_c_file(c: &mut Criterion) {
    let input = std::fs::read_to_string("./resources/c_example.txt").unwrap();
    let g = Grammar::parse_grammar("./resources/c_grammar.txt").unwrap();
    let dfa = Dfa::new(&g);
    c.bench_function("c simulation [tokenizing]", |b| {
        b.iter_batched(
            || (DfaSimulator::new(dfa.clone()), input.chars()),
            |(mut simulator, mut reader)| simulator.tokenize(&mut reader),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, parse_c_grammar, lex_c_grammar, tokenize_c_file);
criterion_main!(benches);
