use std::io::Error as IOError;
use std::str::Chars;

/// Trait used to read a given amount of UTF-8 characters from different sources.
///
/// This trait is similar to [`Chars`] but handle the conversion from `u8` to `char` in most cases.
pub trait Utf8CharReader {
    /// Reads the given amount of UTF-8 character from the implementor.
    ///
    /// If not enough characters are available, this method returns all the available ones.
    /// # Examples
    /// Reading from [`str`] or [`String`]:
    /// ```
    /// use wisent::lexer::Utf8CharReader;
    ///
    /// let string = "🦀: «Hello!»";
    /// let mut chars = string.chars();
    /// assert_eq!(chars.read_chars(1).unwrap(), "🦀");
    /// assert!(chars.read_chars(2).is_ok());
    /// assert_eq!(chars.read_chars(8).unwrap(), "«Hello!»");
    /// ```
    fn read_chars(&mut self, chars_no: usize) -> Result<String, IOError>;
}

impl Utf8CharReader for Chars<'_> {
    fn read_chars(&mut self, chars_no: usize) -> Result<String, IOError> {
        Ok(self.take(chars_no).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::Utf8CharReader;
    use std::io::Error as IOError;

    const UTF8_INPUT: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy";

    #[test]
    fn utf8charreader_read_too_much_characters() -> Result<(), IOError> {
        let mut input = UTF8_INPUT.chars();
        assert_eq!(input.read_chars(6)?, "Příliš");
        assert_eq!(input.read_chars(999)?, " žluťoučký kůň úpěl ďábelské ódy");
        Ok(())
    }

    #[test]
    fn utf8charrreader_continue_reading_characters() -> Result<(), IOError> {
        let mut input = UTF8_INPUT.chars();
        assert!(input.read_chars(999).is_ok());
        assert_eq!(input.read_chars(999)?, "".to_string());
        Ok(())
    }
}
