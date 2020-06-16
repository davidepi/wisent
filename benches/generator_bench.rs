#[macro_use]
extern crate criterion;

use criterion::Criterion;
use wisent::grammar;

fn parse_c_grammar(c: &mut Criterion) {
    c.bench_function("c grammar", |b| {
        b.iter(|| grammar::parse_grammar("./resources/comment_rich_grammar.txt"))
    });
}

criterion_group!(benches, parse_c_grammar);
criterion_main!(benches);
