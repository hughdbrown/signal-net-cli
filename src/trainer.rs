//! Trials, grading and every learning rule of the signal engine.

use std::collections::HashSet;
use std::fmt::Write as _;

use crate::corpus::{AL, STAGES, gen_from, gen_pair};
use crate::lab::{
    CH, D, DM, Event, EventKind, Focus, GM, GW, HM, K, L, LAM, Lab, LangScore, MINT, Mode, NRESP,
    OD, OO, P, PERFECT, PROMOTE, PT, Source, Stuck, TO, TrainMode,
};
use crate::pairs::Pair;
use crate::signal::{add_clamp, merge};

/// Character shown for an output slot: printable ASCII as itself, an empty slot as a space and
/// anything else as `·`.
fn slot_char(x: u8) -> char {
    if (32..127).contains(&x) {
        char::from(x)
    } else if x == 0 {
        ' '
    } else {
        '·'
    }
}

/// JavaScript's `toFixed`: like `{:.d}`, but an exact tie rounds up instead of to even.
#[allow(clippy::float_cmp)] // detecting an exact decimal tie is the point
pub(crate) fn js_fixed(x: f64, digits: i32) -> String {
    let scale = 10f64.powi(digits);
    let scaled = x * scale;
    let d = digits as usize;
    if scaled.fract().abs() == 0.5 && scaled / scale == x {
        format!("{:.d$}", (scaled + 0.5).floor() / scale)
    } else {
        format!("{x:.d$}")
    }
}

/// Eligibility weight of hop `z` of a path: later hops (nearer the output) count more.
fn el(z: usize) -> f64 {
    0.4 + (0.6 * z as f64) / (L - 1) as f64
}

impl Lab {
    pub(crate) fn mode_is(&self, m: TrainMode) -> bool {
        self.settings.train_mode == m
    }

    /// Modes graded per character token (curriculum and meaning).
    pub(crate) fn tokm(&self) -> bool {
        self.mode_is(TrainMode::Curriculum) || self.mode_is(TrainMode::Meaning)
    }

    /// Grader strictness, 0..1; only the language mode tightens.
    #[must_use]
    pub fn strictness(&self) -> f64 {
        if self.mode_is(TrainMode::Response) || self.tokm() || !self.settings.adaptive {
            0.0
        } else {
            let lo = if self.mode_is(TrainMode::Language) {
                0.3
            } else {
                0.0
            };
            self.em.max(lo).min(1.0)
        }
    }

    fn set_train_mode(&mut self, m: TrainMode) {
        self.settings.train_mode = m;
    }

    /// Loads pair `idx` as the current question.
    pub(crate) fn set_pair(&mut self, idx: usize) {
        let Some(p) = self.pairs.get(idx) else { return };
        let (input, output) = (p.input.clone(), p.output.clone());
        self.tw = output.split(' ').map(str::to_owned).collect();
        self.assoc = self
            .lm
            .as_ref()
            .map(|lm| lm.assoc(&input))
            .unwrap_or_default();
        self.codes = input
            .chars()
            .map(|c| crate::rng::first_unit(c) as u8)
            .collect();
        self.codes.truncate(100);
        self.tgt_units = output.encode_utf16().collect();
        self.tgt_str = output;
        self.tb.fill(0);
        for (c, &u) in self.tgt_units.iter().take(100).enumerate() {
            let x = u & 255;
            for b in 0..8 {
                self.tb[c * 8 + b] = ((x >> b) & 1) as u8;
            }
        }
    }

    /// Target character code at slot `c` (0 past the end), as `charCodeAt(c) & 255`.
    fn target_code(&self, c: usize) -> u8 {
        self.tgt_units.get(c).map_or(0, |&u| (u & 255) as u8)
    }

    fn single_pair(&mut self, p: Pair) -> usize {
        self.pairs = vec![p];
        self.last = vec![-1];
        self.sel = vec![true];
        self.order = vec![0];
        self.ptr = 1;
        0
    }

    fn next_idx(&mut self) -> usize {
        if self.order.is_empty() {
            self.order = (0..self.pairs.len()).collect();
        }
        if self.mode == Mode::Eval {
            let i = self.order.get(self.ptr).copied().unwrap_or(0);
            self.ptr += 1;
            return i;
        }
        if self.mode_is(TrainMode::Curriculum) && self.mode == Mode::Train {
            let p = self.cur_next();
            return self.single_pair(p);
        }
        if self.settings.source == Source::Generated {
            let p = gen_pair(&mut self.rng, self.gl, false);
            return self.single_pair(p);
        }
        if self.settings.focus == Focus::OneAtATime {
            return self.cur_pair;
        }
        if self.ptr >= self.order.len() {
            self.ptr = 0;
            self.epoch += 1;
            self.shuffle_order();
        }
        let i = self.order.get(self.ptr).copied().unwrap_or(0);
        self.ptr += 1;
        i
    }

    /// Runs the trial schedule: presents a question for 20 ticks, waits for up to 5 submitted
    /// responses and grades each one; `phase()` in the JavaScript.
    pub(crate) fn phase(&mut self) {
        if self.mode == Mode::Idle {
            self.learn = false;
            if self.txt_ttl > 0 {
                self.txt_ttl -= 1;
                self.stim();
            }
            return;
        }
        if self.pt == 0 {
            self.ensure_sub();
            self.first_hit = -1;
            self.gn = 0;
            self.hn = 0;
            self.dn = 0;
            self.ec = 0;
            self.nr = 0;
            self.last_sub = -1_000_000_000;
            self.trial_ex = false;
            self.sub_hit = false;
            self.sc = 0;
            self.stamp = ((u32::from(self.stamp) % 65_535) + 1) as u16;
            self.acc_s.fill(0.0);
            self.ci = self.next_idx();
            self.set_pair(self.ci);
            if self.mode_is(TrainMode::Curriculum) && self.mode == Mode::Train {
                for q in 0..TO {
                    self.bq[q] =
                        (0.985 * f64::from(self.bq[q]) + 0.015 * f64::from(self.tb[q])) as f32;
                }
            }
            self.learn = self.mode == Mode::Train;
        }
        if self.pt < 20 {
            self.stim();
        }
        self.pt += 1;
        let dead =
            self.pt >= 40 && self.pt % 16 == 0 && !self.cur.stamp[..self.ca].contains(&self.stamp);
        let limit = if self.settings.wait_all { 6000 } else { PT };
        let cut = dead
            || self.pt >= limit
            || (self.nr == 0 && self.first_hit >= 0 && self.pt >= (self.first_hit + 250).max(MINT));
        if self.sub_hit || (cut && self.nr == 0) {
            if self.grade() {
                self.trial_ex = true;
                self.perfect_now = true;
            }
            self.nr += 1;
            self.sub_hit = false;
            self.sc = 0;
            self.last_sub = self.pt;
        }
        if self.nr < NRESP && !cut && !self.perfect_now {
            return;
        }
        self.perfect_now = false;
        if self.mode_is(TrainMode::Curriculum) && self.mode == Mode::Train {
            self.cur_end();
        }
        if let Some(l) = self.last.get_mut(self.ci) {
            *l = i8::from(self.trial_ex);
        }
        self.pt = 0;
        self.trials_done += 1;
        if self.mode == Mode::Eval {
            self.e_n += 1;
            self.e_ok += u64::from(self.trial_ex);
            if self.ptr >= self.order.len() {
                self.mode = Mode::Idle;
                self.notice(format!("EVAL: {}/{} exact", self.e_ok, self.e_n));
            }
        } else if self.settings.focus == Focus::OneAtATime {
            self.streak = if self.trial_ex { self.streak + 1 } else { 0 };
            self.cur_tries += 1;
            // Move on only after 7 trials in a row whose response contains the correct answer; wrap
            // back to pair 1 only after the whole list has been gone through.
            if self.streak >= PERFECT {
                self.streak = 0;
                self.cur_tries = 0;
                if self.cur_pair + 1 < self.pairs.len() {
                    self.cur_pair += 1;
                } else {
                    self.cur_pair = 0;
                    self.epoch += 1;
                }
            }
        }
    }

    /// Records the path of a stimulus signal that reached output bit `q`; in response mode, rewards it.
    pub(crate) fn credit(&mut self, i: usize, q: usize, p: u32) {
        let o = i * L;
        let resp = self.mode_is(TrainMode::Response);
        if !resp && self.hn < HM {
            let h = self.hn;
            self.hp[h * L..h * L + L].copy_from_slice(&self.cur.trace[o..o + L]);
            self.hc[h] = (q >> 3) as u8;
            self.hpp[h] = p;
            self.hw[h] = self.tb[q];
            self.hb[h] = q as u16;
            self.hn += 1;
        }
        if !resp {
            return;
        }
        // Response mode: every path that reaches an output bit is good.
        if self.gn < GM {
            let g = self.gn;
            self.gt[g * L..g * L + L].copy_from_slice(&self.cur.trace[o..o + L]);
            self.gn += 1;
        } else if self.rng.rnd() < 0.1 {
            let g = self.rng.ri(GM);
            self.gt[g * L..g * L + L].copy_from_slice(&self.cur.trace[o..o + L]);
        }
        let jj = self.cur.trace[o + L - 1 - self.rng.ri(20)];
        if jj >= 0 {
            self.push_rg(jj as usize / K);
        }
        for z in 0..L {
            let j = self.cur.trace[o + z];
            if j < 0 {
                continue;
            }
            let j = j as usize;
            self.w[j] = add_clamp(self.w[j], 0.02 * el(z), f64::MIN, 1.0);
            let m = self.rng.r32() & self.rng.r32();
            self.tag[j] = merge(self.tag[j], p, m);
        }
    }

    /// Remembers a neuron in a rewarded area; rewired links are pulled toward these.
    fn push_rg(&mut self, n: usize) {
        self.rg[self.rgi & 255] = n as i32;
        self.rgi += 1;
    }

    fn set_thr(&mut self) {
        for s in 0..100 {
            let m = self.acc_s[s * 8..s * 8 + 8]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            self.ths[s] = self.th.max(0.3 * f64::from(m)) as f32;
        }
    }

    /// Reads 100 characters from an output accumulator using the grading thresholds.
    pub(crate) fn readout(&mut self, use_background: bool) -> String {
        self.set_thr();
        let a = if use_background {
            &self.acc
        } else {
            &self.acc_s
        };
        (0..100)
            .map(|c| {
                let mut x = 0u8;
                for b in 0..8 {
                    if a[c * 8 + b] > self.ths[c] {
                        x |= 1 << b;
                    }
                }
                slot_char(x)
            })
            .collect()
    }

    /// The on-screen output: signals of this question that arrived since the last graded response.
    pub(crate) fn readout_shown(&mut self) -> String {
        for i in 0..TO {
            if self.acc_s[i] < self.acc_b[i] {
                self.acc_b[i] = self.acc_s[i];
            }
            self.acc_d[i] = self.acc_s[i] - self.acc_b[i];
        }
        (0..100)
            .map(|c| {
                let d = &self.acc_d[c * 8..c * 8 + 8];
                let m = d.iter().copied().fold(0.0f32, f32::max);
                let th = self.th.max(0.3 * f64::from(m));
                let mut x = 0u8;
                for (b, &v) in d.iter().enumerate() {
                    if f64::from(v) > th {
                        x |= 1 << b;
                    }
                }
                slot_char(x)
            })
            .collect()
    }

    fn struggle_text(&self) -> String {
        let mut a: Vec<(f32, usize, &str)> = Vec::new();
        for i in 0..800 {
            if self.fm[i] > 0.3 {
                a.push((self.fm[i], i, "dark"));
            }
            if self.fx[i] > 0.3 {
                a.push((self.fx[i], i, "stray"));
            }
        }
        a.sort_by(|p, q| q.0.total_cmp(&p.0));
        let mut bs = [0f32; 8];
        for c in 0..40 {
            for (b, v) in bs.iter_mut().enumerate() {
                *v = (f64::from(*v)
                    + (f64::from(self.fm[c * 8 + b]) + f64::from(self.fx[c * 8 + b])) / 40.0)
                    as f32;
            }
        }
        let mut wb = 0;
        for b in 1..8 {
            if bs[b] > bs[wb] {
                wb = b;
            }
        }
        let spots: Vec<String> = a
            .iter()
            .take(5)
            .map(|(r, i, k)| {
                format!(
                    "slot {} bit {} {k} {}%",
                    i >> 3,
                    i & 7,
                    (f64::from(*r) * 100.0).round()
                )
            })
            .collect();
        let mut s = if spots.is_empty() {
            "none".to_owned()
        } else {
            spots.join(" · ")
        };
        if bs[wb] > 0.08 {
            let _ = write!(
                s,
                " | worst bit overall: {wb} ({}% avg)",
                (f64::from(bs[wb]) * 100.0).round()
            );
        }
        s
    }

    /// Length of the response read into `ys` (up to its last non-empty slot).
    fn response_len(&self) -> usize {
        self.ys.iter().rposition(|&y| y != 0).map_or(0, |c| c + 1)
    }

    /// Share of the response's words that are real words (lowercase letters and spaces only).
    fn resp_strict(&self) -> f64 {
        let len = self.response_len();
        let Some(lm) = self.lm.as_ref() else {
            return 0.0;
        };
        if len == 0 {
            return 0.0;
        }
        let mut s = String::with_capacity(len);
        for &y in &self.ys[..len] {
            if !(y.is_ascii_lowercase() || y == b' ') {
                return 0.0;
            }
            s.push(char::from(y));
        }
        let ws: Vec<&str> = s.split(' ').filter(|x| !x.is_empty()).collect();
        if ws.is_empty() {
            return 0.0;
        }
        let ok = ws
            .iter()
            .filter(|&&x| lm.vw.contains(x) || self.tw.contains(x))
            .count();
        ok as f64 / ws.len() as f64
    }

    /// Mode-3 score of the response: per-character plausibility, words, grammar, relevance, ending.
    fn lang_score(&self) -> LangScore {
        let mut o = LangScore {
            r: 0.0,
            cs: [-1.0; 100],
        };
        let len = self.response_len();
        let Some(lm) = self.lm.as_ref() else { return o };
        if len == 0 {
            return o;
        }
        let s: Vec<char> = self.ys[..len]
            .iter()
            .map(|&y| {
                if y.is_ascii_lowercase() || y == b' ' {
                    char::from(y)
                } else {
                    '#'
                }
            })
            .collect();
        let st: Vec<char> = ['^', '^'].into_iter().chain(s.iter().copied()).collect();
        let mut wd = String::new();
        let mut pw = String::from("<");
        let mut sum = 0.0;
        for c in 0..len {
            let ch = s[c];
            let mut sc = 0.0;
            if ch == '#' {
                if !wd.is_empty() {
                    pw = std::mem::take(&mut wd);
                }
            } else {
                let a: String = [st[c], st[c + 1]].iter().collect();
                let mut mx = 0.0;
                let mut pv = 0.0;
                for &x in &AL {
                    let v = f64::from(lm.tri(&format!("{a}{x}"))) + 0.1;
                    if v > mx {
                        mx = v;
                    }
                    if x == ch {
                        pv = v;
                    }
                }
                let t = pv / mx;
                if ch == ' ' {
                    let pf = if lm.vw.contains(&wd) { 1.0 } else { 0.0 };
                    let cf = if self.assoc.contains(&wd) && lm.vw.contains(&wd) {
                        1.0
                    } else {
                        0.0
                    };
                    let g = if lm.bg(&format!("{pw}|{wd}")) > 0 {
                        1.0
                    } else {
                        0.0
                    };
                    sc = 0.2 * t + 0.3 * pf + 0.3 * cf + 0.2 * g;
                    pw = std::mem::take(&mut wd);
                } else {
                    wd.push(ch);
                    let vp = if lm.vp.contains(&wd) { 1.0 } else { 0.0 };
                    let asc = if self.assoc.contains(&wd) { 1.0 } else { 0.0 };
                    sc = 0.3 * t + 0.35 * vp + 0.35 * asc;
                }
            }
            o.cs[c] = sc as f32;
            sum += sc;
        }
        let end_word = if wd.is_empty() { &pw } else { &wd };
        let e = if lm.vw.contains(&wd) && lm.bg(&format!("{end_word}|>")) > 0 {
            1.0
        } else {
            0.0
        };
        let ctxw = if self.assoc.contains(&wd) || pw == "<" {
            1.0
        } else {
            0.0
        };
        o.r = ((0.75 * sum) / len as f64 + 0.15 * e + 0.1 * ctxw).min(1.0) * self.rs;
        o
    }

    fn in_tok(&self, g: usize) -> bool {
        (self.ys[self.hc[g] as usize] >> (self.hb[g] & 7)) & 1 == 1
    }

    fn path(&self, g: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.hp[g * L..g * L + L]
            .iter()
            .enumerate()
            .filter(|&(_, &j)| j >= 0)
            .map(|(z, &j)| (z, j as usize))
    }

    /// Grades one response and applies every reward and penalty; `grade()` in the JavaScript.
    /// Returns whether the response counts toward advancing (exact, or containing the target).
    fn grade(&mut self) -> bool {
        self.set_thr();
        let got = self.readout(false).trim_end().to_owned();
        if self.nr == 0 {
            self.first_got.clone_from(&got);
            self.len = self.tgt_units.len();
        }
        let len = self.len;
        let (mut pc, mut tc, mut extra) = (0usize, 0usize, 0usize);
        let mut ex = true;
        let mut any = false;
        let fresh = self.first_hit >= 0;
        for c in 0..100 {
            let x = if c < len { self.target_code(c) } else { 0 };
            let mut y = 0u8;
            for b in 0..8 {
                if self.acc_s[c * 8 + b] > self.ths[c] {
                    y |= 1 << b;
                }
            }
            any |= y != 0;
            self.ys[c] = y;
            self.tk[c] = u8::from(x == y);
            if x == y {
                if c < len {
                    pc += 1;
                } else {
                    tc += 1;
                }
            } else if c < len {
                ex = false;
            } else {
                extra += 1;
            }
        }
        if extra > 0 {
            ex = false;
        }
        let tokm = self.tokm();
        let resp = self.mode_is(TrainMode::Response);
        if self.learn && tokm && !self.tgt_str.is_empty() {
            for c in 0..(len + 2).min(100) {
                let x = if c < len { self.target_code(c) } else { 0 };
                let y = self.ys[c];
                for b in 0..8 {
                    let i = c * 8 + b;
                    let lit = (y >> b) & 1 == 1;
                    if (x >> b) & 1 == 1 {
                        self.fm[i] =
                            (f64::from(self.fm[i]) * 0.92 + if lit { 0.0 } else { 0.08 }) as f32;
                    } else {
                        self.fx[i] =
                            (f64::from(self.fx[i]) * 0.92 + if lit { 0.08 } else { 0.0 }) as f32;
                    }
                }
            }
            self.struggle = self.struggle_text();
        }
        self.rs = self.resp_strict();
        let ls = self.mode_is(TrainMode::Language).then(|| self.lang_score());
        if let Some(ls) = &ls {
            ex = ls.r >= 0.85;
        }
        let mx = self.max_x();
        self.last_reach = mx;
        self.rc = self.rc * 0.97 + 0.03 * mx as f64;
        self.hr = self.hr * 0.97 + if any { 0.03 } else { 0.0 };
        if resp && self.hr > 0.85 && self.learn {
            self.set_train_mode(TrainMode::Meaning);
            self.hr = 0.5;
            self.notice("AUTO: outputs are reliable, switching to mode 2".to_owned());
        }
        let tokm = self.tokm();
        let resp = self.mode_is(TrainMode::Response);
        // Protected links: every link on a path that built a correct character in THIS response.
        let mut ps: HashSet<usize> = HashSet::new();
        if self.learn && tokm {
            for g in 0..self.hn {
                if self.tk[self.hc[g] as usize] == 0 {
                    continue;
                }
                ps.extend(self.path(g).map(|(_, j)| j));
            }
        }
        let tail_room = if len == 100 { 1 } else { 100 - len };
        let q0 = (pc as f64 / len as f64) * (0.5 + (0.5 * tc as f64) / tail_room as f64);
        let any_r = any && fresh;
        let q = if resp {
            if any_r { 0.3 } else { 0.0 }
        } else {
            ls.as_ref().map_or(q0, |l| l.r)
        };
        let st = self.strictness();
        let t0 = self.tr.max(MINT as f64);
        let pt = self.pt as f64;
        let sv = if pt <= t0 {
            1.0
        } else {
            (1.0 - (pt - t0) / (2.0 * t0)).max(0.0)
        };
        let tm = 0.5 + sv;
        if self.first_hit >= 0 {
            self.tr = (self.tr * 0.9 + self.first_hit as f64 * 0.1).clamp(100.0, 4000.0);
        }
        if self.learn && !any_r {
            self.add_sub_link();
        }
        if self.learn {
            let reward = 0.04 * q.powf(1.0 + 8.0 * st) * tm - 0.02 * st * (1.0 - q);
            let sd = if any_r { reward } else { -0.03 };
            for g in 0..self.sc {
                for z in 0..L {
                    let j = self.spc[g * L + z];
                    if j >= 0 {
                        let j = j as usize;
                        self.w[j] = add_clamp(self.w[j], sd, 0.05, 1.0);
                    }
                }
            }
            if let Some(ls) = &ls {
                for g in 0..self.hn {
                    let sc = f64::from(ls.cs[self.hc[g] as usize]);
                    if sc < 0.0 {
                        continue;
                    }
                    let lc = 0.3 + 0.35 * (self.ga / 0.8).min(1.0);
                    let d = 0.06
                        * (sc - lc)
                        * if sc > lc {
                            1.0
                        } else {
                            1.0 + 3.0 * self.strictness()
                        };
                    for z in 0..L {
                        let j = self.hp[g * L + z];
                        if j < 0 {
                            continue;
                        }
                        let j = j as usize;
                        let e = el(z);
                        self.w[j] = add_clamp(self.w[j], d * e, 0.01, 1.0);
                        if sc > 0.7 {
                            self.cons[j] = add_clamp(self.cons[j], 0.05 * e, f64::MIN, 1.0);
                            if z == L - 1 {
                                self.push_rg(j / K);
                            }
                        }
                        if sc > 0.8 {
                            let m = self.rng.r32() & self.rng.r32();
                            self.tag[j] = merge(self.tag[j], self.hpp[g], m);
                        }
                    }
                }
            }
            if self.gn > 0 {
                let bonus = if any_r { reward } else { 0.0 };
                for g in 0..self.gn {
                    for z in 0..L {
                        let j = self.gt[g * L + z];
                        if j < 0 {
                            continue;
                        }
                        let j = j as usize;
                        if !(bonus < 0.0 && (ps.contains(&j) || self.mlk[j] > 0.5)) {
                            self.w[j] = add_clamp(self.w[j], bonus, 0.05, 1.0);
                        }
                    }
                }
            }
        }
        if tokm && self.learn {
            self.token_credit(&ps);
        }
        let tgt_t = self.tgt_str.trim_end().to_owned();
        let contains = !tgt_t.is_empty() && got.contains(&tgt_t);
        // Success lock: when this response holds the target, lock in every path that built a correct
        // character, so the next (noisy) trial cannot immediately undo it.
        if self.learn && tokm && (ex || contains) {
            for g in 0..self.hn {
                if self.tk[self.hc[g] as usize] == 0 {
                    continue;
                }
                for z in 0..L {
                    let j = self.hp[g * L + z];
                    if j < 0 {
                        continue;
                    }
                    let j = j as usize;
                    let e = el(z);
                    self.gk[j] = self.gk[j].max(0.9);
                    if ex {
                        self.mlk[j] = 1.0;
                    }
                    self.cons[j] = add_clamp(self.cons[j], 0.25 * e, f64::MIN, 1.0);
                    self.w[j] = add_clamp(self.w[j], 0.08 * e, f64::MIN, 1.0);
                }
            }
        }
        if self.settings.smart && self.learn && tokm && !ex {
            self.smart_fix();
            self.slot_boost();
        }
        if self.learn && !resp {
            self.drain_learn(ex);
        }
        if self.mode_is(TrainMode::Curriculum)
            && self.learn
            && self.nr == 0
            && !self.first_got.is_empty()
            && !ex
            && !self.first_got.contains(&self.tgt_str)
        {
            self.repeat_penalty(&ps);
        }
        self.gn = 0;
        self.hn = 0;
        self.dn = 0;
        for a in &mut self.acc_s {
            *a = (f64::from(*a) * 0.5) as f32;
        }
        self.acc_b.copy_from_slice(&self.acc_s);
        let exf = if ex { 1.0 } else { 0.0 };
        self.em = self.em * 0.97 + 0.03 * exf;
        self.ba = self.ba * 0.97 + 0.03 * self.rs;
        match &ls {
            Some(l) if l.r >= 0.0 => self.ga = self.ga * 0.95 + 0.05 * l.r,
            Some(_) => {}
            None => self.ga = self.ga * 0.95 + 0.05 * exf,
        }
        if self.settings.source == Source::Generated && self.ga > 0.78 && self.gl < 3 {
            self.gl += 1;
            self.ga = 0.3;
        }
        let near = (resp && got.contains(&self.tgt_str)) || contains;
        let mark = if ex {
            "✓ "
        } else if near {
            "≈ "
        } else {
            "✗ "
        };
        let lang = ls
            .as_ref()
            .map(|l| format!("lang {} L{} ", js_fixed(l.r, 2), self.gl))
            .unwrap_or_default();
        let input = self.pairs.get(self.ci).map_or("", |p| p.input.as_str());
        let shown = if got.is_empty() {
            "(silent)"
        } else {
            got.as_str()
        };
        let stale = if any && !fresh {
            " (STALE: nothing from this input arrived)"
        } else {
            ""
        };
        let text = format!(
            "{lang}{mark}#{} {input} => {} | {shown} | chars right {pc}/{len} | valid-word score {}{stale} | submitted t{} (speed x{}, transit ~{})",
            self.nr + 1,
            self.tgt_str,
            js_fixed(self.rs, 2),
            self.pt,
            js_fixed(tm, 2),
            self.tr.round(),
        );
        self.events.push(Event {
            kind: EventKind::Response,
            text,
        });
        ex || near
    }

    /// Advantage-weighted credit for every recorded path, per output character (modes 0 and 2).
    fn token_credit(&mut self, ps: &HashSet<usize>) {
        for c in 0..100 {
            let r = if self.tk[c] == 1 { 1.0 } else { -1.0 };
            let bt = f64::from(self.bt[c]);
            self.adv[c] = (r - bt + 0.2 * r).clamp(-2.0, 2.0) as f32;
            self.bt[c] = (0.9 * bt + 0.1 * r) as f32;
        }
        let lrs = self.lrs;
        for g in 0..self.hn {
            let c = self.hc[g] as usize;
            let ok = self.tk[c] == 1;
            let in_tok = self.in_tok(g);
            let wanted = if self.hw[g] == 1 { 1.0 } else { -1.0 };
            let a =
                (f64::from(self.adv[c]) + if in_tok { LAM * wanted } else { 0.0 }).clamp(-2.0, 2.0);
            for z in 0..L {
                let j = self.hp[g * L + z];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let e = el(z);
                if !ok && ps.contains(&j) {
                    continue; // shared with a correct character: leave untouched
                }
                if !ok && self.mlk[j] > 0.5 {
                    // mastered by an earlier pair: only a long run of conflicting failures unlocks it
                    self.mlk[j] = (f64::from(self.mlk[j]) - 0.02) as f32;
                    continue;
                }
                if ok {
                    self.gk[j] = add_clamp(self.gk[j], 0.1 * e, f64::MIN, 1.0);
                } else if self.gk[j] > 0.5 && (a < 0.0 || !in_tok) {
                    continue;
                }
                if in_tok {
                    self.w[j] = add_clamp(self.w[j], 0.05 * lrs * a * e, 0.01, 1.0);
                    let m = self.rng.r32() & self.rng.r32();
                    if a > 0.0 {
                        self.tag[j] = merge(self.tag[j], self.hpp[g], m);
                    } else if a < -0.3 {
                        self.tag[j] = merge(self.tag[j], !self.hpp[g], m);
                    }
                    if ok && a > 0.3 {
                        self.cons[j] = add_clamp(self.cons[j], 0.05 * e, f64::MIN, 1.0);
                    }
                } else if !ok {
                    self.w[j] = (f64::from(self.w[j]) - 0.01 * lrs * e).max(0.01) as f32;
                }
            }
            if ok && in_tok {
                let jj = self.hp[g * L + L - 1 - self.rng.ri(20)];
                if jj >= 0 {
                    self.push_rg(jj as usize / K);
                }
            }
        }
    }

    /// Curriculum: punish paths when the network repeats a wrong answer it gave recently.
    fn repeat_penalty(&mut self, ps: &HashSet<usize>) {
        let rep = self
            .rh
            .iter()
            .zip(&self.rt)
            .rev()
            .take(8)
            .filter(|(h, t)| **h == self.first_got && **t != self.tgt_str)
            .count();
        if rep == 0 {
            return;
        }
        let pen = 0.012 * self.lrs * rep.min(3) as f64;
        for g in 0..self.hn {
            // Repeat-failure penalty must not erode paths that built a correct character.
            if self.tk[self.hc[g] as usize] == 1 {
                continue;
            }
            for z in 0..L {
                let j = self.hp[g * L + z];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                if self.gk[j] > 0.5 || ps.contains(&j) || self.mlk[j] > 0.5 {
                    continue;
                }
                self.w[j] = (f64::from(self.w[j]) - pen * el(z)).max(0.01) as f32;
                if rep >= 3 && z >= L - 8 && self.rng.rnd() < 0.3 {
                    self.tag[j] = self.rng.r32();
                }
            }
        }
    }

    /// Furthest x reached by this question's signals.
    fn max_x(&self) -> usize {
        (0..self.ca)
            .filter(|&i| self.cur.stamp[i] == self.stamp)
            .map(|i| self.cur.neuron[i] as usize)
            .filter(|&n| n < P)
            .map(|n| n % GW)
            .max()
            .unwrap_or(0)
    }

    /// Target bytes and "slot is wrong" flags of the current response.
    fn wrong_slots(&self) -> ([u8; 100], [bool; 100]) {
        let mut xs = [0u8; 100];
        for (c, x) in xs
            .iter_mut()
            .enumerate()
            .take(self.tgt_units.len().min(100))
        {
            *x = self.target_code(c);
        }
        let mut wr = [false; 100];
        for c in 0..100 {
            wr[c] = self.ys[c] != xs[c];
        }
        (xs, wr)
    }

    /// Links on paths that built a correct character (never punished or rewired).
    fn protected_links(&self, wr: &[bool; 100]) -> HashSet<usize> {
        let mut prot = HashSet::new();
        for g in 0..self.hn {
            if !wr[self.hc[g] as usize] {
                prot.extend(self.path(g).map(|(_, j)| j));
            }
        }
        prot
    }

    /// Drain learning: put drains on the paths of unwanted signals, and strengthen or weaken the drains
    /// that fired depending on whether wanted bits stayed dark.
    fn drain_learn(&mut self, ex: bool) {
        let n = self.tgt_units.len();
        let (xs, wr) = self.wrong_slots();
        let miss = (0..n.min(100)).any(|c| xs[c] & !self.ys[c] != 0);
        let prot = self.protected_links(&wr);
        let mut made = 0;
        for g in 0..self.hn {
            if made >= 5 {
                break;
            }
            let c = self.hc[g] as usize;
            let b = (self.hb[g] & 7) as usize;
            if !wr[c] || (self.ys[c] >> b) & 1 == 0 || (xs[c] >> b) & 1 == 1 {
                continue;
            }
            let chance = if self.fx[c * 8 + b] > CH { 0.95 } else { 0.6 };
            if self.rng.rnd() > chance {
                continue;
            }
            let jl = self.hp[g * L + L - 1 - self.rng.ri(6)];
            if jl < 0 || prot.contains(&(jl as usize)) || self.mlk[jl as usize] > 0.5 {
                continue;
            }
            let jl = jl as usize;
            let nn = jl / K;
            if nn >= P {
                continue;
            }
            // Never put a drain on a neuron that carried a correctly produced character.
            if (0..K).any(|k| prot.contains(&(nn * K + k)) || self.mlk[nn * K + k] > 0.5) {
                continue;
            }
            let mut km = None;
            let mut kd = None;
            let mut wm = 9.0f32;
            for k in 0..K {
                let j = nn * K + k;
                if self.tgt[j] >= OD as i32 {
                    kd = Some(k);
                    break;
                }
                if j != jl
                    && !prot.contains(&j)
                    && self.mlk[j] < 0.5
                    && self.tgt[j] < OO as i32
                    && self.w[j] < wm
                {
                    wm = self.w[j];
                    km = Some(k);
                }
            }
            if let Some(kd) = kd {
                let j = nn * K + kd;
                let m = self.rng.r32() & self.rng.r32();
                self.w[j] = add_clamp(self.w[j], 0.08, f64::MIN, 1.0);
                self.tag[j] = merge(self.tag[j], self.hpp[g], m);
                made += 1;
            } else if let Some(km) = km {
                let j = nn * K + km;
                self.tgt[j] = (OD + self.rng.ri(D)) as i32;
                self.tag[j] = self.hpp[g];
                self.w[j] = 0.15; // low weight: only captures signals whose payload matches its tag
                self.cons[j] = 0.0;
                made += 1;
            }
        }
        for d in 0..self.dn.min(DM) {
            let jd = self.dp[d * L + L - 1];
            if jd < 0 {
                continue;
            }
            let jd = jd as usize;
            if ex {
                self.w[jd] = add_clamp(self.w[jd], 0.02, f64::MIN, 1.0);
                self.cons[jd] = add_clamp(self.cons[jd], 0.02, f64::MIN, 1.0);
            } else if miss {
                self.w[jd] = (f64::from(self.w[jd]) - 0.12).max(0.02) as f32;
                self.cons[jd] = (f64::from(self.cons[jd]) * 0.7) as f32;
                if self.rng.rnd() < 0.5 {
                    let m = self.rng.r32() & self.rng.r32();
                    self.tag[jd] = merge(self.tag[jd], !self.dpp[d], m);
                }
            }
        }
    }

    /// Starved-bit repair: copy the final hop of a path that lit another wanted bit in the same slot
    /// onto a wanted bit that stayed dark.
    fn slot_boost(&mut self) {
        let n = self.tgt_units.len().min(100);
        let chronic = (0..n)
            .map(|c| {
                let x = self.target_code(c);
                (0..8)
                    .filter(|&b| {
                        (x >> b) & 1 == 1 && (self.ys[c] >> b) & 1 == 0 && self.fm[c * 8 + b] > CH
                    })
                    .count()
            })
            .sum::<usize>();
        let limit = 2 + chronic.min(3);
        let mut done = 0;
        for c in 0..n {
            if done >= limit {
                break;
            }
            let x = self.target_code(c);
            let y = self.ys[c];
            if x == y || x & !y == 0 {
                continue; // only slots missing at least one wanted bit
            }
            let src: Vec<usize> = (0..self.hn)
                .filter(|&g| {
                    let b = self.hb[g] & 7;
                    self.hc[g] as usize == c && (x >> b) & 1 == 1 && (y >> b) & 1 == 1
                })
                .collect();
            if src.is_empty() {
                continue;
            }
            for b in 0..8 {
                if (x >> b) & 1 == 0 || (y >> b) & 1 == 1 {
                    continue;
                }
                let chance = if self.fm[c * 8 + b] > CH { 0.97 } else { 0.5 };
                if self.rng.rnd() > chance {
                    continue;
                }
                let g = src[self.rng.ri(src.len())];
                let jl = self.hp[g * L + L - 1];
                if jl < 0 {
                    continue;
                }
                let jl = jl as usize;
                let nn = jl / K;
                let mut km = None;
                let mut wm = 9.0f32;
                for k in 0..K {
                    let j = nn * K + k;
                    if j != jl
                        && self.w[j] < wm
                        && self.tgt[j] < OO as i32
                        && self.gk[j] < 0.5
                        && self.mlk[j] < 0.5
                    {
                        wm = self.w[j];
                        km = Some(k);
                    }
                }
                let Some(km) = km else { continue };
                let j = nn * K + km;
                self.tgt[j] = (OO + c * 8 + b) as i32;
                self.tag[j] = self.hpp[g];
                self.w[j] = 0.6;
                self.cons[j] = 0.2;
                done += 1;
                break;
            }
        }
    }

    /// Smart connecting: weaken the last hops of paths that lit unwanted bits, and add hindsight links
    /// from neurons this question's signals sit on to wanted bits that stayed dark.
    fn smart_fix(&mut self) {
        let n = self.tgt_units.len();
        let (xs, wr) = self.wrong_slots();
        let wl: Vec<usize> = (0..n.min(100)).filter(|&c| wr[c]).collect();
        let nw = wr.iter().filter(|&&x| x).count();
        if nw == 0 {
            return;
        }
        let prot = self.protected_links(&wr);
        let pw = 0.12 * (3.0 / nw as f64).min(1.0) * self.lrs;
        for g in 0..self.hn {
            let c = self.hc[g] as usize;
            let b = self.hb[g] & 7;
            if wr[c] && (self.ys[c] >> b) & 1 == 1 && (xs[c] >> b) & 1 == 0 {
                let mut hit = false;
                for z in L - 3..L {
                    let j = self.hp[g * L + z];
                    if j < 0 {
                        continue;
                    }
                    let j = j as usize;
                    if !prot.contains(&j) && self.gk[j] < 0.5 && self.mlk[j] < 0.5 {
                        self.w[j] = (f64::from(self.w[j]) - pw).max(0.01) as f32;
                        self.cons[j] = (f64::from(self.cons[j]) * 0.7) as f32;
                        hit = true;
                    }
                }
                if hit {
                    self.sm_pen += 1;
                }
            }
        }
        if wl.is_empty() {
            return;
        }
        let fail = (1.0 - self.em).clamp(0.0, 1.0);
        let pr = (0.1 + 0.5 * fail).min(0.6);
        let mut cand = Vec::new();
        for _ in 0..80 {
            if cand.len() >= 12 || self.ca == 0 {
                break;
            }
            let i = self.rng.ri(self.ca);
            if self.cur.stamp[i] == self.stamp && (self.cur.neuron[i] as usize) < P {
                cand.push(i);
            }
        }
        if cand.is_empty() {
            return;
        }
        let hl: Vec<usize> = wl
            .iter()
            .copied()
            .filter(|&c| {
                (0..8).any(|b| {
                    (xs[c] >> b) & 1 == 1 && (self.ys[c] >> b) & 1 == 0 && self.fm[c * 8 + b] > CH
                })
            })
            .collect();
        let mx = 2 + hl.len().min(3);
        let mut made = 0;
        let mut q = 0;
        while q < wl.len() + hl.len() && made < mx {
            q += 1;
            let c = if !hl.is_empty() && self.rng.rnd() < 0.7 {
                hl[self.rng.ri(hl.len())]
            } else {
                wl[self.rng.ri(wl.len())]
            };
            let (x, y) = (xs[c], self.ys[c]);
            for b in 0..8 {
                if made >= mx {
                    break;
                }
                if (x >> b) & 1 == 0 || (y >> b) & 1 == 1 {
                    continue;
                }
                let chance = if self.fm[c * 8 + b] > CH { 0.95 } else { pr };
                if self.rng.rnd() >= chance {
                    continue;
                }
                let i = cand[self.rng.ri(cand.len())];
                let nn = self.cur.neuron[i] as usize;
                let mut km = None;
                let mut wm = 9.0f32;
                for k in 0..K {
                    let j = nn * K + k;
                    if self.w[j] < wm
                        && self.tgt[j] < OO as i32
                        && !prot.contains(&j)
                        && self.gk[j] < 0.5
                        && self.mlk[j] < 0.5
                    {
                        wm = self.w[j];
                        km = Some(k);
                    }
                }
                let Some(km) = km else { continue };
                let j = nn * K + km;
                self.tgt[j] = (OO + c * 8 + b) as i32;
                self.tag[j] = self.cur.payload[i];
                self.w[j] = 0.3;
                self.cons[j] = 0.2;
                made += 1;
                self.sm_fix += 1;
            }
        }
    }

    /// Next curriculum pair: keep the stuck pair until perfected, else draw 6 candidates from the
    /// current stage and prefer ones answered badly so far.
    fn cur_next(&mut self) -> Pair {
        let si = self.stage.min(STAGES.len() - 1);
        let warm = ((self.st_n + 1) as f64 / 30.0).min(1.0);
        let anneal = if si == STAGES.len() - 1 { 0.5 } else { 1.0 };
        self.lrs =
            (warm * (0.2 + 0.4 * (1.0 + (std::f64::consts::PI * self.ta.min(1.0)).cos())) * anneal)
                .max(0.15);
        if let Some(stk) = &self.stk {
            return stk.pair.clone();
        }
        let cand: Vec<Pair> = (0..6).map(|_| gen_from(&mut self.rng, si, false)).collect();
        let ws: Vec<f64> = cand
            .iter()
            .map(|c| 1.1 - self.ca_map.get(&c.output).copied().unwrap_or(0.0))
            .collect();
        let sum: f64 = ws.iter().sum();
        let mut r = self.rng.rnd() * sum;
        let mut i = 0;
        while i < 5 && r >= ws[i] {
            r -= ws[i];
            i += 1;
        }
        let pair = cand[i].clone();
        self.stk = Some(Stuck {
            pair: pair.clone(),
            streak: 0,
        });
        pair
    }

    fn cur_end(&mut self) {
        let t = self.tgt_str.clone();
        if let Some(stk) = &mut self.stk {
            stk.streak = if self.trial_ex { stk.streak + 1 } else { 0 };
            if stk.streak >= PERFECT {
                self.stk = None;
                self.mst += 1;
            }
        }
        let exf = if self.trial_ex { 1.0 } else { 0.0 };
        let prev = self.ca_map.get(&t).copied().unwrap_or(0.0);
        self.ca_map.insert(t.clone(), prev * 0.8 + 0.2 * exf);
        let got = if self.first_got.is_empty() {
            "(silent)".to_owned()
        } else {
            self.first_got.clone()
        };
        self.rh.push_back(got);
        self.rt.push_back(t);
        if self.rh.len() > 40 {
            self.rh.pop_front();
            self.rt.pop_front();
        }
        self.ta = self.ta * 0.9 + 0.1 * exf;
        self.st_n += 1;
        if self.mst >= PROMOTE {
            if self.stage >= STAGES.len() - 1 {
                self.notice(format!(
                    "CHECKPOINT: {PROMOTE} more pairs perfected in the final mixture; training continues"
                ));
                self.mst = 0;
            } else {
                self.notice(format!(
                    "PROMOTED: stage {} passed ({PROMOTE} pairs each answered exactly)",
                    STAGES[self.stage].name
                ));
                self.stage += 1;
                self.stk = None;
                self.mst = 0;
                self.va = 0.0;
                self.vn = 0.0;
                self.st_n = 0;
                self.ta = 0.0;
                self.ca_map.clear();
                self.rh.clear();
                self.rt.clear();
            }
        }
    }

    /// Distinct recent outputs are at least 70% of distinct recent targets (no mode collapse).
    fn diverse(&self) -> bool {
        let o: HashSet<&String> = self.rh.iter().collect();
        let t: HashSet<&String> = self.rt.iter().collect();
        t.len() < 4 || o.len() as f64 >= 0.7 * t.len() as f64
    }

    pub(crate) fn curriculum_status(&self) -> String {
        let s = &STAGES[self.stage.min(STAGES.len() - 1)];
        let distinct_out: HashSet<&String> = self.rh.iter().collect();
        let distinct_tgt: HashSet<&String> = self.rt.iter().collect();
        format!(
            "[{}] stage {} | pairs perfected this stage {}/{PROMOTE} | distinct outputs {} vs targets {}{} | LR x{}",
            s.phase,
            s.name,
            self.mst,
            distinct_out.len(),
            distinct_tgt.len(),
            if self.diverse() {
                ""
            } else {
                " | COLLAPSE: promotion blocked"
            },
            js_fixed(self.lrs, 2)
        )
    }

    /// Sends text into the network without learning (the page's "Send text"). The answer builds up
    /// over the following ticks; read it with [`Lab::shown_output`].
    pub fn send_text(&mut self, text: &str) {
        let units: Vec<u16> = text.encode_utf16().take(100).collect();
        self.codes = crate::rng::first_units(&units).map(|u| u as u8).collect();
        self.stamp = ((u32::from(self.stamp) % 65_535) + 1) as u16;
        self.acc_s.fill(0.0);
        self.txt_ttl = 20;
    }

    /// What the output neurons show now: this question's signals since the last graded response.
    pub fn shown_output(&mut self) -> String {
        self.readout_shown().trim_end().to_owned()
    }

    /// Output read from all signals, including leftovers from earlier questions.
    pub fn background_output(&mut self) -> String {
        self.readout(true).trim().to_owned()
    }

    /// Runs one display frame while idle and returns the output if it has just settled (unchanged for
    /// 15 frames) on something new; the page logs those lines under "Talk to it".
    pub fn talk_frame(&mut self) -> Option<String> {
        for _ in 0..self.settings.ticks_per_frame {
            self.step();
        }
        let s = self.readout_shown();
        if s == self.last_str {
            self.stable += 1;
        } else {
            self.stable = 0;
            self.last_str.clone_from(&s);
        }
        let t = s.trim();
        if self.stable == 15 && !t.is_empty() && t != self.last_logged {
            t.clone_into(&mut self.last_logged);
            return Some(format!("t{}: {t}", self.tick));
        }
        None
    }
}
