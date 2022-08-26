#[macro_use]
extern crate criterion;

use criterion::Criterion;
use wisent::grammar::Grammar;
use wisent::lexer::Dfa;

fn parse_c_grammar(c: &mut Criterion) {
    c.bench_function("c grammar [parse grammar]", |b| {
        b.iter(|| Grammar::parse_grammar("./resources/c_grammar.txt"))
    });
}

fn lex_c_grammar(c: &mut Criterion) {
    let g = Grammar::parse_grammar("./resources/c_grammar.txt").unwrap();
    c.bench_function("c grammar [lex grammar]", |b| b.iter(|| Dfa::new(&g)));
}

criterion_group!(benches, parse_c_grammar, lex_c_grammar);
criterion_main!(benches);
