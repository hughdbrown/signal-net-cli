//! Signal Network Lab v3.1, ported from the single-page browser app to a library and command-line tool.
//!
//! Two learning engines share one lab:
//! - the original **signal network**: 500,000 processing neurons on a 1000x500 grid, signals that
//!   carry their question's payload along tag-matched weighted links, path credit when they reach an
//!   output bit, drains, spawning and link rewiring;
//! - the **two-stage** engine: a sparse context layer hashed from the whole question and a readout
//!   trained with a local error-correcting rule.
//!
//! The simulation follows the JavaScript step for step (same random stream, same float32 storage), so
//! a given seed produces the same network and the same training log, and model files move between
//! the page and this crate.
//!
//! ```no_run
//! use signal_net::{Lab, Settings};
//! let mut lab = Lab::browser_startup(Settings::default());
//! lab.start_training().unwrap();
//! for _ in 0..3000 {
//!     lab.step();
//! }
//! for e in lab.drain_events() {
//!     println!("{}", e.text);
//! }
//! ```

mod corpus;
mod lab;
mod pairs;
mod rng;
mod signal;
mod trainer;
mod two_stage;

pub use corpus::{STAGES, Stage, gen_pair, generated_list};
pub use lab::{
    Checksums, DEFAULT_SEED, Engine, Event, EventKind, Focus, Lab, MAXS, Mode, Settings, Source,
    TrainMode,
};
pub use pairs::{MAX_LEN, Pair, format_pairs, parse_pairs, truncate_utf16};
pub use rng::{Rng, hash, hs};
pub use two_stage::{C2, context_neurons};
