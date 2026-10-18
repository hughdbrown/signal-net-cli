//! Signal Network Lab v3.1, ported from the single-page browser app to a library and command-line tool.

mod corpus;
mod pairs;
mod rng;

pub use corpus::{STAGES, Stage, gen_pair, generated_list};
pub use pairs::{MAX_LEN, Pair, format_pairs, parse_pairs, truncate_utf16};
pub use rng::{Rng, hash, hs};
