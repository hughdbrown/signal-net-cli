//! Training pairs and the `input => response` text format of the lab's pair list.

use std::fmt;

/// Longest input or response, in UTF-16 units (the network has 100 character slots).
pub const MAX_LEN: usize = 100;

/// One training example: the question and the exact response wanted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    pub input: String,
    pub output: String,
}

impl Pair {
    /// Builds a pair from anything string-like.
    #[must_use]
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }
}

impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} => {}", self.input, self.output)
    }
}

/// Keeps at most `max` UTF-16 units, like JavaScript's `slice(0, max)` (a surrogate pair that would be
/// split is dropped whole).
#[must_use]
pub fn truncate_utf16(s: &str, max: usize) -> &str {
    let mut units = 0;
    for (i, c) in s.char_indices() {
        units += c.len_utf16();
        if units > max {
            return &s[..i];
        }
    }
    s
}

/// Parses the pair list: one `input => response` per line (a literal `\n` also separates lines). Lines
/// without `=>` or with an empty side are skipped; both sides are trimmed and cut to 100 characters.
///
/// ```
/// let p = signal_net::parse_pairs("hi => hello\nnot a pair\n => x\nbye=>goodbye");
/// assert_eq!(p.len(), 2);
/// assert_eq!(p[1].output, "goodbye");
/// ```
#[must_use]
pub fn parse_pairs(text: &str) -> Vec<Pair> {
    text.replace("\\n", "\n")
        .lines()
        .filter_map(|line| {
            let (a, b) = line.split_once("=>")?;
            let input = truncate_utf16(a.trim(), MAX_LEN);
            let output = truncate_utf16(b.trim(), MAX_LEN);
            (!input.is_empty() && !output.is_empty()).then(|| Pair::new(input, output))
        })
        .collect()
}

/// Writes pairs back in the list format, one per line.
#[must_use]
pub fn format_pairs(pairs: &[Pair]) -> String {
    let mut s = String::new();
    for (i, p) in pairs.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str(&p.to_string());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_like_the_page() {
        let p = parse_pairs("a => a\r\nhi=>hello\\nsay cat => cat\nbad line\nx =>   \n => y");
        assert_eq!(
            p,
            vec![
                Pair::new("a", "a"),
                Pair::new("hi", "hello"),
                Pair::new("say cat", "cat")
            ]
        );
    }

    #[test]
    fn splits_on_first_arrow_only() {
        let p = parse_pairs("a => b => c");
        assert_eq!(p, vec![Pair::new("a", "b => c")]);
    }

    #[test]
    fn truncates_to_100_units() {
        let long = "x".repeat(150);
        let p = parse_pairs(&format!("{long} => {long}"));
        assert_eq!(p[0].input.len(), 100);
        assert_eq!(truncate_utf16("ab😀", 3), "ab");
        assert_eq!(truncate_utf16("ab😀", 4), "ab😀");
    }

    #[test]
    fn format_round_trips() {
        let p = vec![Pair::new("hi", "hello"), Pair::new("bye", "goodbye")];
        assert_eq!(parse_pairs(&format_pairs(&p)), p);
    }
}
