pub mod core {
    pub fn print_hello() {
        println!("Hello World!");
    }
}

#[cfg(test)]
mod tests {
    use crate::core;
    #[test]
    fn it_works() {
        core::print_hello();
    }
}
