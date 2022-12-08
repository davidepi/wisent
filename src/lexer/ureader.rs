/// The `UnicodeReader<R>` struct transforms a byte iterator into and UTF-8 char iterator.
///
/// Each call of `UnicodeIterator::next` will repeatedly pull bytes from the underlying byte
/// iterator until an UTF-8 character can be constructed or an `io::Error` is raised.
/// # Example
/// ```
/// # use std::io::{BufReader, Read};
/// # use wisent::lexer::UnicodeReader;
/// let unicode_string = "こんにちは";
/// let reader = BufReader::new(unicode_string.as_bytes());
/// let mut ureader = UnicodeReader::new(reader.bytes());
///
/// assert_eq!(ureader.next().unwrap().unwrap(), 'こ');
/// assert_eq!(ureader.next().unwrap().unwrap(), 'ん');
/// assert_eq!(ureader.next().unwrap().unwrap(), 'に');
/// assert_eq!(ureader.next().unwrap().unwrap(), 'ち');
/// assert_eq!(ureader.next().unwrap().unwrap(), 'は');
/// assert!(ureader.next().is_none());
/// ```
pub struct UnicodeReader<R: Iterator<Item = Result<u8, std::io::Error>>> {
    reader: R,
}

impl<R: Iterator<Item = Result<u8, std::io::Error>>> UnicodeReader<R> {
    /// Creates the `UnicodeReader<R>` consuming the given iterator
    /// # Examples
    /// ```
    /// # use std::io::{BufReader, Read};
    /// # use wisent::lexer::UnicodeReader;
    /// let unicode_string = "안녕하세요";
    /// let reader = BufReader::new(unicode_string.as_bytes());
    /// let ureader = UnicodeReader::new(reader.bytes());
    /// ```
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    /// Consumes the `UnicodeReader<R>` returning the remaining original iterator.
    /// # Examples
    /// ```
    /// # use std::io::{BufReader, Read};
    /// # use wisent::lexer::UnicodeReader;
    /// let unicode_string = "你好";
    /// let reader = BufReader::new(unicode_string.as_bytes());
    /// let mut ureader = UnicodeReader::new(reader.bytes());
    /// assert_eq!(ureader.next().unwrap().unwrap(), '你');
    ///
    /// let mut original_reader = ureader.into_inner();
    /// assert_eq!(original_reader.next().unwrap().unwrap(), 0xE5);
    /// assert_eq!(original_reader.next().unwrap().unwrap(), 0xA5);
    /// assert_eq!(original_reader.next().unwrap().unwrap(), 0xBD);
    /// assert!(original_reader.next().is_none());
    /// ```
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R: Iterator<Item = Result<u8, std::io::Error>>> Iterator for UnicodeReader<R> {
    type Item = Result<char, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut byte = match self.reader.next() {
            Some(next) => match next {
                Ok(val) => val,
                Err(e) => return Some(Err(e)),
            },
            None => return None,
        };
        let mut codepoint = 0;
        let mut state = 0;
        state = utf8_dfa(state, byte, &mut codepoint);
        while state != 0 && state != 1 {
            byte = match self.reader.next() {
                Some(next) => match next {
                    Ok(val) => val,
                    Err(e) => return Some(Err(e)),
                },
                None => {
                    return Some(Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "unexpected EOF while decoding utf8",
                    )))
                }
            };
            state = utf8_dfa(state, byte, &mut codepoint);
        }
        match state {
            0 => Some(Ok(unsafe { std::char::from_u32_unchecked(codepoint) })),
            _ => Some(Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "invalid utf8 data",
            ))),
        }
    }
}

const UTF8_TRANSITION_0: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 11, 6, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8,
];

const UTF8_TRANSITION_1: [u8; 108] = [
    0, 12, 24, 36, 60, 96, 84, 12, 12, 12, 48, 72, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 0, 12, 12, 12, 12, 12, 0, 12, 0, 12, 12, 12, 24, 12, 12, 12, 12, 12, 24, 12, 24, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12, 12, 12, 12, 24, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12, 12, 36, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12,
    12, 36, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
];

// utf8 decoder from bytes. See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
fn utf8_dfa(state: u8, byte: u8, codepoint: &mut u32) -> u8 {
    let tp = UTF8_TRANSITION_0[byte as usize];
    *codepoint = if state != 0 {
        (byte as u32 & 0x3F) | (*codepoint << 6)
    } else {
        (0xFF >> tp) & (byte as u32)
    };
    UTF8_TRANSITION_1[(state + tp) as usize]
}

#[cfg(test)]
mod tests {
    use super::UnicodeReader;
    use std::io::{BufReader, Cursor, Read};

    #[test]
    fn valid_utf8_one_byte_only() {
        let string = "x9RyE";
        let bufreader = BufReader::new(string.as_bytes());
        let ureader = UnicodeReader::new(bufreader.bytes());
        assert_eq!(
            string.chars().collect::<Vec<_>>(),
            ureader.map(|x| x.unwrap()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn valid_utf8_one_and_two_bytes_only() {
        let string = "xξRyζ";
        let bufreader = BufReader::new(string.as_bytes());
        let ureader = UnicodeReader::new(bufreader.bytes());
        assert_eq!(
            string.chars().collect::<Vec<_>>(),
            ureader.map(|x| x.unwrap()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn valid_utf8_one_and_two_and_three_bytes_only() {
        let string = "xξ字yζ";
        let bufreader = BufReader::new(string.as_bytes());
        let ureader = UnicodeReader::new(bufreader.bytes());
        assert_eq!(
            string.chars().collect::<Vec<_>>(),
            ureader.map(|x| x.unwrap()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn valid_utf8_one_and_two_and_three_and_four_btyes() {
        let string = "xξ字🤌ζ";
        let bufreader = BufReader::new(string.as_bytes());
        let ureader = UnicodeReader::new(bufreader.bytes());
        assert_eq!(
            string.chars().collect::<Vec<_>>(),
            ureader.map(|x| x.unwrap()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_first_bytes() {
        let data = vec![0x96, 0x00, 0x00, 0x00];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn missing_second_byte() {
        let data = vec![0xC3, 0x00, 0x00, 0x00];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn missing_third_byte() {
        let data = vec![0xE3, 0x81, 0x00, 0x00];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn missing_fourth_byte() {
        let data = vec![0xF0, 0x9F, 0x9C, 0x00];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    // additional tests from https://www.cl.cam.ac.uk/%7Emgk25/ucs/examples/UTF-8-test.txt

    #[test]
    fn invalid_fe_sequence() {
        let data = vec![0xFE];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_ff_sequence() {
        let data = vec![0xFF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_feff_sequence() {
        let data = vec![0xFE, 0xFE, 0xFF, 0xFF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_ascii_2bytes() {
        let data = vec![0xC0, 0xAF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_ascii_3bytes() {
        let data = vec![0xE0, 0x80, 0xAF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_ascii_4bytes() {
        let data = vec![0xF0, 0x80, 0x80, 0xAF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_ascii_5bytes() {
        let data = vec![0xF8, 0x80, 0x80, 0x80, 0xAF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_ascii_6bytes() {
        let data = vec![0xFC, 0x80, 0x80, 0x80, 0x80, 0xAF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_boundary_2bytes() {
        let data = vec![0xC1, 0xBF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_boundary_3bytes() {
        let data = vec![0xE0, 0x9F, 0xBF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_boundary_4bytes() {
        let data = vec![0xF0, 0x8F, 0xBF, 0xBF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_boundary_5bytes() {
        let data = vec![0xF8, 0x87, 0xBF, 0xBF, 0xBF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_boundary_6bytes() {
        let data = vec![0xFC, 0x83, 0xBF, 0xBF, 0xBF, 0xBF];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_null_2bytes() {
        let data = vec![0xC0, 0x80];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_null_3bytes() {
        let data = vec![0xE0, 0x80, 0x80];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_null_4bytes() {
        let data = vec![0xF0, 0x80, 0x80, 0x80];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_null_5bytes() {
        let data = vec![0xF8, 0x80, 0x80, 0x80, 0x80];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_overlong_null_6bytes() {
        let data = vec![0xFC, 0x80, 0x80, 0x80, 0x80, 0x80];
        let reader = BufReader::new(Cursor::new(data));
        let mut ureader = UnicodeReader::new(reader.bytes());
        assert!(ureader.next().unwrap().is_err());
    }

    #[test]
    fn invalid_single_surrogates() {
        let data = [
            vec![0xED, 0xA0, 0x80],
            vec![0xED, 0xAD, 0xBF],
            vec![0xED, 0xAE, 0x80],
            vec![0xED, 0xAF, 0xBF],
            vec![0xED, 0xB0, 0x80],
            vec![0xED, 0xBE, 0x80],
            vec![0xED, 0xBF, 0xBF],
        ];
        for surrogate in data {
            let reader = BufReader::new(Cursor::new(surrogate));
            let mut ureader = UnicodeReader::new(reader.bytes());
            assert!(ureader.next().unwrap().is_err());
        }
    }

    #[test]
    fn invalid_double_surrogates() {
        let data = [
            vec![0xED, 0xA0, 0x80, 0xED, 0xB0, 0x80],
            vec![0xED, 0xA0, 0x80, 0xED, 0xBF, 0xBF],
            vec![0xED, 0xAD, 0xBF, 0xED, 0xB0, 0x80],
            vec![0xED, 0xAD, 0xBF, 0xED, 0xBF, 0xBF],
            vec![0xED, 0xAE, 0x80, 0xED, 0xB0, 0x80],
            vec![0xED, 0xAE, 0x80, 0xED, 0xBF, 0xBF],
            vec![0xED, 0xAF, 0xBF, 0xED, 0xB0, 0x80],
            vec![0xED, 0xAF, 0xBF, 0xED, 0xBF, 0xBF],
        ];
        for surrogate in data {
            let reader = BufReader::new(Cursor::new(surrogate));
            let mut ureader = UnicodeReader::new(reader.bytes());
            assert!(ureader.next().unwrap().is_err());
        }
    }
}
