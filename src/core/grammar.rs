pub struct Grammar {}

pub fn parse_grammar(path: &str) -> std::io::Result<Grammar> {
    let grammar_content = std::fs::read_to_string(path)?;
    let _grammar_no_comments = remove_comments(&grammar_content);
    return Ok(Grammar {});
}

fn remove_comments(content: &String) -> String {
    let mut ret: String = String::new();
    for char in content.chars() {
        //TODO: Add actual implementation here
        ret.push(char);
    }
    ret
}
