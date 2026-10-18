//! The generated language: question/answer generator, the curriculum stages and the small language
//! model that mode 3 (language) grades against.

use std::collections::{HashMap, HashSet};

use crate::pairs::Pair;
use crate::rng::{Rng, hs};

const NW: [&str; 19] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
];
const AN: [(&str, &str); 6] = [
    ("cat", "meow"),
    ("dog", "woof"),
    ("cow", "moo"),
    ("duck", "quack"),
    ("sheep", "baa"),
    ("pig", "oink"),
];
const TC: [(&str, &str); 6] = [
    ("sky", "blue"),
    ("grass", "green"),
    ("sun", "yellow"),
    ("snow", "white"),
    ("coal", "black"),
    ("rose", "red"),
];
const FD: [&str; 6] = ["pizza", "cake", "soup", "rice", "bread", "apples"];
const WD: [&str; 14] = [
    "cat", "dog", "sun", "moon", "tree", "bird", "fish", "frog", "star", "rain", "book", "door",
    "lamp", "ship",
];
type Greeting = (&'static [&'static str], &'static [&'static str]);
const GR: [Greeting; 6] = [
    (
        &["hi", "hey", "hello"],
        &["hello", "hi there", "hello friend"],
    ),
    (&["good morning"], &["good morning", "have a nice day"]),
    (
        &["how are you", "how do you do"],
        &["i am fine", "i am doing well"],
    ),
    (
        &["what is your name", "who are you"],
        &["i am a net", "my name is net"],
    ),
    (&["thanks", "thank you"], &["you are welcome", "no problem"]),
    (&["bye", "see you", "goodbye"], &["goodbye", "see you soon"]),
];
/// Characters the language model predicts over (`$` ends a response).
pub(crate) const AL: [char; 28] = [
    ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '$',
];

/// Generates one question/answer pair; `genPair(lv, rn)` in the JavaScript.
///
/// `lv < 0` selects level `-lv - 1` exactly (0 echo, 1 greetings, 2 facts, 3 arithmetic); `lv >= 9`
/// picks any level; otherwise the level is `lv` half the time and at most `lv` the rest. With
/// `random_reply` a greeting gets any of its accepted replies instead of the fixed one.
pub fn gen_pair(rng: &mut Rng, lv: i32, random_reply: bool) -> Pair {
    let l = if lv < 0 {
        -lv - 1
    } else if lv >= 9 {
        rng.ri(4) as i32
    } else if rng.rnd() < 0.5 {
        lv
    } else {
        rng.ri((lv + 1) as usize) as i32
    };
    match l {
        0 => {
            let w = *rng.pick(&WD);
            let say = *rng.pick(&["say ", "repeat "]);
            Pair::new(format!("{say}{w}"), w)
        }
        1 => {
            let (qs, rs) = *rng.pick(&GR);
            let q = *rng.pick(qs);
            let r = if random_reply {
                *rng.pick(rs)
            } else {
                rs[hs(q) as usize % rs.len()]
            };
            Pair::new(q, r)
        }
        2 => match rng.ri(3) {
            0 => {
                let (a, s) = *rng.pick(&AN);
                Pair::new(
                    format!("what does the {a} say"),
                    format!("the {a} says {s}"),
                )
            }
            1 => {
                let (a, c) = *rng.pick(&TC);
                Pair::new(format!("what color is the {a}"), format!("the {a} is {c}"))
            }
            _ => {
                let f = *rng.pick(&FD);
                Pair::new(format!("do you like {f}"), format!("yes i like {f}"))
            }
        },
        _ => {
            let a = rng.ri(10);
            let b = rng.ri(10);
            let (na, nb, ns) = (NW[a], NW[b], NW[a + b]);
            Pair::new(
                format!("what is {na} plus {nb}"),
                format!("{na} plus {nb} is {ns}"),
            )
        }
    }
}

/// The page's fixed training list: 6000 draws from levels 1-3, one pair per distinct question, sorted
/// by total length (131 pairs).
pub fn generated_list(rng: &mut Rng) -> Vec<Pair> {
    let mut seen = HashSet::new();
    let mut list = Vec::new();
    for _ in 0..6000 {
        let lv = -2 - rng.ri(3) as i32;
        let p = gen_pair(rng, lv, false);
        if seen.insert(p.input.clone()) {
            list.push(p);
        }
    }
    list.sort_by_key(|p| p.input.encode_utf16().count() + p.output.encode_utf16().count());
    list
}

/// How a curriculum stage produces its pairs.
#[derive(Debug, Clone, Copy)]
enum StageGen {
    /// Copy a random string of `lo..=hi` symbols from the alphabet.
    Copy(&'static [u8], usize, usize),
    /// A [`gen_pair`] level.
    Level(i32),
    /// Any earlier stage.
    Mixture,
}

/// One stage of the lab-style curriculum (training mode 0).
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    pub phase: &'static str,
    pub name: &'static str,
    held_out: bool,
    gen_: StageGen,
}

/// The nine curriculum stages; promotion happens after `PROMOTE` pairs are perfected in a stage.
pub const STAGES: [Stage; 9] = [
    Stage {
        phase: "PRETRAIN",
        name: "1 copy 1-3 of 8 balanced symbols",
        held_out: false,
        gen_: StageGen::Copy(b"pqrstuvw", 1, 3),
    },
    Stage {
        phase: "PRETRAIN",
        name: "2 copy 1-4 of 16 letters",
        held_out: false,
        gen_: StageGen::Copy(b"abcdefghijklmnop", 1, 4),
    },
    Stage {
        phase: "PRETRAIN",
        name: "3 copy 2-6 letters",
        held_out: false,
        gen_: StageGen::Copy(b"abcdefghijklmnopqrstuvwxyz", 2, 6),
    },
    Stage {
        phase: "PRETRAIN",
        name: "4 copy 2-6 letters (held-out strings)",
        held_out: true,
        gen_: StageGen::Copy(b"abcdefghijklmnopqrstuvwxyz", 2, 6),
    },
    Stage {
        phase: "PRETRAIN",
        name: "5 say a word (held-out)",
        held_out: true,
        gen_: StageGen::Level(-1),
    },
    Stage {
        phase: "SFT",
        name: "6 small talk (instruction to response)",
        held_out: false,
        gen_: StageGen::Level(-2),
    },
    Stage {
        phase: "SFT",
        name: "7 facts",
        held_out: false,
        gen_: StageGen::Level(-3),
    },
    Stage {
        phase: "REASONING",
        name: "8 arithmetic (held-out sums)",
        held_out: true,
        gen_: StageGen::Level(-4),
    },
    Stage {
        phase: "RL/ANNEAL",
        name: "9 mixture of all stages, half learning rate",
        held_out: false,
        gen_: StageGen::Mixture,
    },
];

fn gen_stage_once(rng: &mut Rng, stage: usize) -> Pair {
    match STAGES[stage].gen_ {
        StageGen::Copy(alphabet, lo, hi) => {
            let n = lo + rng.ri(hi - lo + 1);
            let s: String = (0..n).map(|_| char::from(*rng.pick(alphabet))).collect();
            Pair::new(s.clone(), s)
        }
        StageGen::Level(lv) => gen_pair(rng, lv, false),
        StageGen::Mixture => {
            let other = rng.ri(STAGES.len() - 1);
            gen_from(rng, other, false)
        }
    }
}

/// Draws a pair from a stage; for held-out stages, keeps drawing (up to 40 times) until the pair is on
/// the requested side of the 1-in-10 validation split. `genFrom` in the JavaScript.
pub(crate) fn gen_from(rng: &mut Rng, stage: usize, validation: bool) -> Pair {
    let accept = |p: &Pair| {
        !STAGES[stage].held_out
            || hs(&format!("{}>{}", p.input, p.output)).is_multiple_of(10) == validation
    };
    let mut p = gen_stage_once(rng, stage);
    for _ in 1..40 {
        if accept(&p) {
            break;
        }
        p = gen_stage_once(rng, stage);
    }
    p
}

/// Character-trigram, word-bigram and question-word/answer-word statistics of the generated corpus.
#[derive(Debug, Default)]
pub struct Lm {
    pub(crate) tri: HashMap<String, u32>,
    pub(crate) bg: HashMap<String, u32>,
    pub(crate) vp: HashSet<String>,
    pub(crate) vw: HashSet<String>,
    cx: HashMap<String, u32>,
    iwc: HashMap<String, u32>,
    cxi: HashMap<String, HashSet<String>>,
}

fn inc(m: &mut HashMap<String, u32>, k: String) {
    *m.entry(k).or_insert(0) += 1;
}

impl Lm {
    /// Builds the model from 3000 random pairs of every level; `buildLM` in the JavaScript.
    pub fn build(rng: &mut Rng) -> Self {
        let mut lm = Self::default();
        for _ in 0..3000 {
            let p = gen_pair(rng, 9, true);
            let s: Vec<char> = format!("^^{}$", p.output).chars().collect();
            for i in 2..s.len() {
                inc(&mut lm.tri, [s[i - 2], s[i - 1], s[i]].iter().collect());
                inc(&mut lm.tri, [s[i - 2], s[i - 1]].iter().collect());
            }
            let ws: Vec<&str> = p.output.split(' ').collect();
            let mut pw = "<";
            for &w in &ws {
                inc(&mut lm.bg, format!("{pw}|{w}"));
                lm.vw.insert(w.to_owned());
                for (i, c) in w.char_indices() {
                    lm.vp.insert(w[..i + c.len_utf8()].to_owned());
                }
                pw = w;
            }
            inc(&mut lm.bg, format!("{pw}|>"));
            for x in p.input.split(' ') {
                inc(&mut lm.iwc, x.to_owned());
                for &y in &ws {
                    inc(&mut lm.cx, format!("{x}|{y}"));
                    lm.cxi.entry(x.to_owned()).or_default().insert(y.to_owned());
                }
            }
        }
        lm
    }

    /// Prefixes of answer words that usually follow the words of this question; `mkAssoc`.
    pub(crate) fn assoc(&self, input: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        for x in input.split(' ') {
            let Some(words) = self.cxi.get(x) else {
                continue;
            };
            let total = f64::from(self.iwc.get(x).copied().unwrap_or(1).max(1));
            for w in words {
                let n = f64::from(self.cx.get(&format!("{x}|{w}")).copied().unwrap_or(0));
                if n / total >= 0.25 {
                    for (i, c) in w.char_indices() {
                        out.insert(w[..i + c.len_utf8()].to_owned());
                    }
                }
            }
        }
        out
    }

    pub(crate) fn tri(&self, k: &str) -> u32 {
        self.tri.get(k).copied().unwrap_or(0)
    }

    pub(crate) fn bg(&self, k: &str) -> u32 {
        self.bg.get(k).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_list_has_every_question_once() {
        let mut rng = Rng::new(42);
        let list = generated_list(&mut rng);
        assert_eq!(list.len(), 131);
        let distinct: HashSet<_> = list.iter().map(|p| &p.input).collect();
        assert_eq!(distinct.len(), list.len());
        assert!(
            list.iter().any(|p| p.input == "what is seven plus seven"
                && p.output == "seven plus seven is fourteen")
        );
    }

    #[test]
    fn held_out_split_is_respected() {
        let mut rng = Rng::new(7);
        for _ in 0..50 {
            let p = gen_from(&mut rng, 7, false);
            assert_ne!(hs(&format!("{}>{}", p.input, p.output)) % 10, 0);
        }
    }

    #[test]
    fn language_model_knows_the_vocabulary() {
        let mut rng = Rng::new(9);
        let lm = Lm::build(&mut rng);
        assert!(lm.vw.contains("hello"));
        assert!(lm.vp.contains("hel"));
        assert!(lm.bg("<|the") > 0);
        let assoc = lm.assoc("do you like pizza");
        assert!(assoc.contains("ye") && assoc.contains("pizza"));
    }
}
