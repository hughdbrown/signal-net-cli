//! The two-stage engine: a sparse context layer and an error-correcting readout.
//!
//! 1. Context layer: a question switches on a sparse set of `C2` context neurons, picked by hashing
//!    pieces of the WHOLE input (letter@slot, word, word@index, word pair, full string).
//! 2. Readout: every output bit has a link from every context neuron. After a trial, for each bit that
//!    came out wrong or only barely right, the links from the neurons that were on move toward the
//!    target (a local error-correcting rule with a margin). No random routing: same question, same
//!    answer.
//! 3. Interleaved: every epoch trains all pairs in a fresh shuffled order.

use std::collections::HashSet;

use crate::lab::{Focus, Lab, Mode, P, TO, TwoPass};
use crate::rng::{hash, hs_units};

/// Context neurons.
pub const C2: usize = 16_384;
/// Context neurons switched on per feature.
const C2H: u32 = 4;

/// Context neurons a question switches on, in first-seen order; `ctx2` in the JavaScript.
#[must_use]
pub fn context_neurons(input: &str) -> Vec<usize> {
    let inp: Vec<u16> = input.encode_utf16().collect();
    let words: Vec<&[u16]> = inp.split(|&u| u == u16::from(b' ')).collect();
    let mut features: Vec<Vec<u16>> = Vec::new();
    let text = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
    for (i, &u) in inp.iter().enumerate() {
        let mut f = text(&format!("pc{i}"));
        f.push(u);
        features.push(f);
    }
    for (i, w) in words.iter().enumerate() {
        features.push([&text("w")[..], w].concat());
        features.push([&text(&format!("wi{i}"))[..], w].concat());
        if i > 0 {
            features.push([&text("b")[..], words[i - 1], &text("|")[..], w].concat());
        }
    }
    features.push([&text("all")[..], &inp[..]].concat());

    let mut seen_features = HashSet::new();
    let mut seen = HashSet::new();
    let mut on = Vec::new();
    for f in features {
        if !seen_features.insert(f.clone()) {
            continue;
        }
        let h = hs_units(&f);
        for k in 0..C2H {
            let a = hash(h, k + 11) as usize % C2;
            if seen.insert(a) {
                on.push(a);
            }
        }
    }
    on
}

/// Turns readout scores into text: a bit is on when its score is positive.
fn text2(z: &[f32]) -> String {
    let s: String = (0..100)
        .map(|c| {
            let mut x = 0u8;
            for b in 0..8 {
                if z[c * 8 + b] > 0.0 {
                    x |= 1 << b;
                }
            }
            if (32..127).contains(&x) {
                char::from(x)
            } else if x == 0 {
                ' '
            } else {
                '·'
            }
        })
        .collect();
    s.trim_end().to_owned()
}

impl Lab {
    /// Whether the two-stage readout has been trained or loaded.
    #[must_use]
    pub fn has_readout(&self) -> bool {
        self.w2.is_some()
    }

    /// Readout scores for a set of active context neurons, into `self.z2`.
    fn score2(&mut self, active: &[usize]) {
        let mut z = [-0.5f64; TO];
        if let Some(w2) = &self.w2 {
            for &a in active {
                for (zo, &w) in z.iter_mut().zip(&w2[a * TO..a * TO + TO]) {
                    *zo += f64::from(w);
                }
            }
        }
        for (dst, src) in self.z2.iter_mut().zip(z) {
            *dst = src as f32;
        }
    }

    /// The two-stage answer to a question (no learning).
    pub fn answer_two_stage(&mut self, input: &str) -> String {
        let active = context_neurons(crate::pairs::truncate_utf16(input, 100));
        self.score2(&active);
        self.o2 = text2(&self.z2);
        self.o2.clone()
    }

    /// One trial of the two-stage engine; `step2` in the JavaScript.
    pub(crate) fn step2(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        if self.e2.mode != Some(self.mode) {
            self.e2 = TwoPass {
                ord: Vec::new(),
                p: 0,
                ex: 0,
                ep: self.e2.ep,
                mode: Some(self.mode),
            };
        }
        if self.e2.p >= self.e2.ord.len() {
            if !self.e2.ord.is_empty() {
                let head = if self.mode == Mode::Eval {
                    "EVAL: ".to_owned()
                } else {
                    format!("EPOCH {}: ", self.e2.ep + 1)
                };
                self.notice(format!(
                    "{head}{}/{} exact (two-stage)",
                    self.e2.ex,
                    self.e2.ord.len()
                ));
                if self.mode == Mode::Eval {
                    self.mode = Mode::Idle;
                    return;
                }
                self.e2.ep += 1;
                self.epoch = self.e2.ep;
            }
            self.e2.ord = (0..self.pairs.len())
                .filter(|&i| self.settings.focus != Focus::Checked || self.sel[i])
                .collect();
            if self.mode != Mode::Eval {
                for i in (1..self.e2.ord.len()).rev() {
                    let j = self.rng.ri(i + 1);
                    self.e2.ord.swap(i, j);
                }
            }
            self.e2.p = 0;
            self.e2.ex = 0;
            if self.e2.ord.is_empty() {
                self.mode = Mode::Idle;
                return;
            }
        }
        self.ci = self.e2.ord[self.e2.p];
        self.e2.p += 1;
        self.set_pair(self.ci);
        let input = self.pairs[self.ci].input.clone();
        let active = context_neurons(&input);
        self.score2(&active);
        let got = text2(&self.z2);
        let ex = got == self.tgt_str;
        if self.mode == Mode::Train {
            let lr = 0.3 / (active.len() as f64).sqrt();
            let mut w2 = self.w2.take().unwrap_or_else(|| vec![0.0; C2 * TO]);
            for o in 0..TO {
                let want = self.tb[o] == 1;
                let margin = if want { self.z2[o] } else { -self.z2[o] };
                if margin < 1.0 {
                    let d = if want { lr } else { -lr };
                    for &a in &active {
                        let j = a * TO + o;
                        w2[j] = (f64::from(w2[j]) + d) as f32;
                    }
                }
            }
            self.w2 = Some(w2);
        }
        for o in 0..TO {
            self.acc_s[o] = if self.z2[o] > 0.0 { 2.0 } else { 0.0 };
        }
        for &a in &active {
            self.act[(a * 30) % P] = 255;
        }
        self.o2 = got;
        self.e2.ex += usize::from(ex);
        if let Some(l) = self.last.get_mut(self.ci) {
            *l = i8::from(ex);
        }
        self.trials_done += 1;
        self.em = self.em * 0.97 + if ex { 0.03 } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_deterministic_and_sparse() {
        let a = context_neurons("what is two plus three");
        assert_eq!(a, context_neurons("what is two plus three"));
        assert!(a.len() > 20 && a.len() < 400);
        assert_ne!(a, context_neurons("what is three plus two"));
        assert!(a.iter().all(|&n| n < C2));
    }

    #[test]
    fn untrained_readout_is_silent() {
        assert_eq!(text2(&[-0.5; TO]), "");
    }
}
