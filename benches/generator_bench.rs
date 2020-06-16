#[macro_use]
extern crate criterion;

use criterion::Criterion;
use wisent::grammar;

fn parse_c_grammar(c: &mut Criterion) {
    c.bench_function("c grammar [parse grammar]", |b| {
        b.iter(|| grammar::parse_grammar("./resources/c_grammar.txt"))
    });
}

criterion_group!(benches, parse_c_grammar);
criterion_main!(benches);
