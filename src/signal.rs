//! The signal network: wiring, signal propagation, spawning and link pruning.

use crate::lab::{
    D, DRP, ENF, ENN, Engine, GH, GW, K, L, Lab, MAXS, MINT, Mode, N, OD, OI, OO, OS, OV, OX0, OY0,
    P, RF, SX, SY, TI, TO, V,
};
use crate::rng::{hash, pay};

/// The page's (approximate) 2π for random angles; kept as written so random points match.
#[allow(clippy::approx_constant)]
const ANGLE: f64 = 6.2832;

/// JavaScript's `Math.round` (halves round up, also for negatives).
pub(crate) fn js_round(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// Adds `d` in double precision and stores back to single precision, clamped to `[lo, hi]`; this is
/// how every `Float32Array` update in the page behaves.
pub(crate) fn add_clamp(x: f32, d: f64, lo: f64, hi: f64) -> f32 {
    (f64::from(x) + d).clamp(lo, hi) as f32
}

/// Merges the bits of `src` selected by `m` into `dst`.
pub(crate) fn merge(dst: u32, src: u32, m: u32) -> u32 {
    (dst & !m) | (src & m)
}

/// Map x position of a neuron.
pub(crate) fn px(n: usize) -> f64 {
    if n == OS {
        SX as f64
    } else if n < P {
        (n % GW) as f64
    } else if n < OO {
        0.0
    } else if n < OV {
        (OX0 + (n - OO) % 20) as f64
    } else if n >= OD {
        ((n - OD) % 40) as f64 * (GW as f64 / 40.0)
    } else {
        1.0 + ((n - OV) % 100) as f64 * 0.4
    }
}

/// Map y position of a neuron.
pub(crate) fn py(n: usize) -> f64 {
    if n == OS {
        SY as f64
    } else if n < P {
        (n / GW) as f64
    } else if n < OO {
        (n - OI) as f64 * (GH as f64 / 800.0)
    } else if n < OV {
        (OY0 + (n - OO) / 20) as f64
    } else if n >= OD {
        (GH + 10 + ((n - OD) / 40) * 2) as f64
    } else {
        ((n - OV) / 100) as f64 * (GH as f64 / 100.0)
    }
}

fn in_grid(x: i64, y: i64) -> bool {
    x >= 0 && x < GW as i64 && y >= 0 && y < GH as i64
}

impl Lab {
    fn new_target_unlimited(&mut self, n: usize) -> i32 {
        if n < P {
            let r = self.rng.rnd();
            if r < 0.000_05 {
                return OS as i32;
            }
            if r < 0.0017 {
                return (OO + self.rng.ri(TO)) as i32;
            }
            if r < 0.0017 + DRP {
                return (OD + self.rng.ri(D)) as i32;
            }
        }
        let mut t = self.rng.ri(P);
        if t == n {
            t = (t + 1) % P;
        }
        t as i32
    }

    /// Picks a target for link `k` of neuron `n`; `newTarget` in the JavaScript.
    pub(crate) fn new_target(&mut self, n: usize, k: usize) -> i32 {
        if self.settings.unlimited_reach {
            return self.new_target_unlimited(n);
        }
        if n < P && self.rng.rnd() < DRP {
            return (OD + self.rng.ri(D)) as i32;
        }
        let reach = if n < P {
            if k >= 4 { 40.0 } else { 4.0 }
        } else {
            30.0
        };
        let (x0, y0) = (px(n), py(n));
        if n < P {
            let (dx, dy) = (SX as f64 - x0, SY as f64 - y0);
            if k < 4 && dx * dx + dy * dy <= 9.0 && self.rng.rnd() < 0.25 {
                return OS as i32;
            }
        }
        if n < P
            && k < 4
            && x0 >= (OX0 - 3) as f64
            && y0 >= (OY0 - 3) as f64
            && self.rng.rnd() < 0.25
        {
            for _ in 0..20 {
                let (x, y) = self.random_point(x0, y0, reach);
                if x >= OX0 as i64 && x < GW as i64 && y >= OY0 as i64 && y < GH as i64 {
                    return (OO as i64 + (y - OY0 as i64) * 20 + (x - OX0 as i64)) as i32;
                }
            }
        }
        for _ in 0..100 {
            let (x, y) = self.random_point(x0, y0, reach);
            if in_grid(x, y) {
                let t = (y * GW as i64 + x) as usize;
                if t != n {
                    return t as i32;
                }
            }
        }
        if n < P {
            ((n + 1) % P) as i32
        } else {
            self.rng.ri(P) as i32
        }
    }

    fn random_point(&mut self, x0: f64, y0: f64, reach: f64) -> (i64, i64) {
        let r = reach * self.rng.rnd().sqrt();
        let a = self.rng.rnd() * ANGLE;
        (js_round(x0 + r * a.cos()), js_round(y0 + r * a.sin()))
    }

    /// Makes sure some processing neuron links to the submit neuron.
    pub(crate) fn ensure_sub(&mut self) {
        let os = OS as i32;
        if (0..P * K).any(|j| self.tgt[j] == os && self.w[j] > 0.0) {
            return;
        }
        let n = SY * GW + SX - 1;
        let j = n * K + self.rng.ri(K);
        self.tgt[j] = os;
        self.w[j] = 0.3;
    }

    /// Adds a link to the submit neuron after a silent response.
    pub(crate) fn add_sub_link(&mut self) -> bool {
        for _ in 0..40 {
            let r = 3.0 * self.rng.rnd().sqrt();
            let a = self.rng.rnd() * ANGLE;
            let x = if self.settings.unlimited_reach {
                self.rng.ri(GW) as i64
            } else {
                js_round(SX as f64 + r * a.cos())
            };
            let y = if self.settings.unlimited_reach {
                self.rng.ri(GH) as i64
            } else {
                js_round(SY as f64 + r * a.sin())
            };
            if !in_grid(x, y) {
                continue;
            }
            let n = (y * GW as i64 + x) as usize;
            let mut has = false;
            let mut bj = None;
            let mut bw = 9.0f32;
            for q in 0..K {
                let j = n * K + q;
                if self.tgt[j] == OS as i32 {
                    has = true;
                    break;
                }
                if self.w[j] < bw {
                    bw = self.w[j];
                    bj = Some(j);
                }
            }
            let Some(bj) = bj.filter(|_| !has) else {
                continue;
            };
            self.tgt[bj] = OS as i32;
            self.w[bj] = 0.3;
            return true;
        }
        false
    }

    /// Gives a model saved before drain neurons existed a few weak drain links.
    pub(crate) fn seed_drains(&mut self, frac: f64) {
        let mut need = (P as f64 * K as f64 * frac).round() as i64;
        let mut tries = need * 60;
        while need > 0 && tries > 0 {
            tries -= 1;
            let j = self.rng.ri(P * K);
            if self.tgt[j] >= OO as i32 || self.w[j] > 0.3 || self.cons[j] > 0.1 {
                continue;
            }
            self.tgt[j] = (OD + self.rng.ri(D)) as i32;
            self.tag[j] = self.rng.r32();
            self.w[j] = (0.2 + self.rng.rnd() * 0.2) as f32;
            self.cons[j] = 0.0;
            need -= 1;
        }
    }

    /// Wires every neuron at random; `init()` in the JavaScript.
    pub(crate) fn init(&mut self) {
        self.nt.fill(0);
        self.spawn_tot = 0;
        self.spawn_tick = 0;
        self.dr_t = 0;
        self.dn = 0;
        self.sub_hit = false;
        self.sc = 0;
        self.first_hit = -1;
        self.tr = 1500.0;
        for n in 0..N {
            let output = (OO..OV).contains(&n) || n >= OS;
            for k in 0..K {
                let i = n * K + k;
                if output {
                    self.w[i] = 0.0;
                    continue;
                }
                self.tgt[i] = self.new_target(n, k);
                self.w[i] = (0.1 + self.rng.rnd() * 0.4) as f32;
                self.tag[i] = self.rng.r32();
                self.cons[i] = 0.0;
            }
        }
        if self.settings.identity_prior {
            for q in 0..TI {
                let j = (OI + q) * K;
                self.tgt[j] = (OO + q) as i32;
                self.w[j] = 0.9;
                self.cons[j] = 0.3;
            }
        }
        self.ensure_sub();
        self.rgi = 0;
        self.cur.energy.fill(0);
        self.act.fill(0);
        self.acc.fill(0.0);
        self.ca = 0;
        self.tick = 0;
        self.keep = 1.0;
        self.epoch = 0;
        self.em = 0.0;
        self.ba = 0.0;
        self.pt = 0;
    }

    /// Adds a signal at neuron `n` with payload `p`, from question `f` (0 = noise).
    pub(crate) fn push(&mut self, n: usize, p: u32, f: u16) {
        if self.ca >= MAXS {
            return;
        }
        let i = self.ca;
        self.cur.neuron[i] = n as i32;
        self.cur.stamp[i] = f;
        self.cur.payload[i] = p;
        self.cur.birth[i] = self.tick;
        self.cur.energy[i] = if f != 0 { ENF } else { ENN };
        if f != 0 {
            self.cur.trace[i * L..i * L + L].fill(-1);
        }
        self.ca += 1;
    }

    /// Sends the current question's bits into the input neurons.
    pub(crate) fn stim(&mut self) {
        let nb: u32 = self.codes.iter().map(|c| c.count_ones()).sum();
        let per_bit = if nb <= 40 {
            4
        } else if nb <= 80 {
            2
        } else {
            1
        };
        for s in 0..self.codes.len().min(100) {
            let c = self.codes[s];
            let p = pay(u32::from(c), s as u32);
            for b in 0..8 {
                if (c >> b) & 1 == 1 {
                    for _ in 0..per_bit {
                        self.push(OI + s * 8 + b, p, self.stamp);
                    }
                }
            }
        }
    }

    fn sort_by_age(&mut self) {
        self.cnt.fill(0);
        let age =
            |tick: u32, birth: u32| (i64::from(tick) - i64::from(birth)).clamp(0, 1023) as usize;
        for i in 0..self.ca {
            self.cnt[age(self.tick, self.cur.birth[i]) + 1] += 1;
        }
        for b in 0..1024 {
            self.cnt[b + 1] += self.cnt[b];
        }
        for i in 0..self.ca {
            let a = age(self.tick, self.cur.birth[i]);
            self.ord[self.cnt[a] as usize] = i as u32;
            self.cnt[a] += 1;
        }
    }

    /// Moves a signal along one weighted, tag-matched link of neuron `n`; `fwd` in the JavaScript.
    /// The signal's path history must be in `self.tr_buf`.
    fn fwd(&mut self, n: usize, f: u16, p: u32, bt: u32, e: u16) {
        let b = n * K;
        let mut s = 0.0;
        for k in 0..K {
            let j = b + k;
            let ww = self.w[j];
            let v = if ww > 0.0 {
                f64::from(ww) * f64::from(self.ex_tab[(p ^ self.tag[j]).count_ones() as usize])
            } else {
                0.0
            };
            self.pk[k] = v;
            s += v;
        }
        if s <= 0.0 || self.na >= self.cap {
            return;
        }
        let mut r = self.rng.rnd() * s;
        let mut k = 0;
        while k < K - 1 && r >= self.pk[k] {
            r -= self.pk[k];
            k += 1;
        }
        let j = b + k;
        self.usd[j] = 1;
        // Signals do not decay: energy is carried through unchanged.
        if f != 0 && e > 0 {
            for z in (0..=L - 4).rev() {
                let u = self.tr_buf[z];
                if u < 0 {
                    break;
                }
                if (u as usize / K) as i32 == self.tgt[j] {
                    self.w[j] = add_clamp(self.w[j], 0.01, f64::MIN, 1.0);
                    self.cons[j] = add_clamp(self.cons[j], 0.005, f64::MIN, 1.0);
                    break;
                }
            }
        }
        if e == 0 {
            return;
        }
        let q = self.na;
        self.na += 1;
        let nx = &mut self.next;
        nx.energy[q] = e;
        nx.neuron[q] = self.tgt[j];
        nx.stamp[q] = f;
        nx.payload[q] = p;
        nx.birth[q] = bt;
        if f != 0 {
            let o = q * L;
            nx.trace[o..o + L - 1].copy_from_slice(&self.tr_buf[1..L]);
            nx.trace[o + L - 1] = j as i32;
        }
    }

    /// Advances the simulation by one tick (one trial for the two-stage engine).
    pub fn step(&mut self) {
        self.th = 0.5 + 2.5 * self.strictness();
        if self.settings.engine == Engine::TwoStage && self.mode != Mode::Idle {
            self.step2();
            return;
        }
        self.cap = self.settings.hard_cap;
        for _ in 0..self.settings.noise {
            let n = self.rng.ri(P);
            let p = self.rng.r32();
            self.push(n, p, 0);
        }
        if self.vis_ttl > 0 {
            self.vis_ttl -= 1;
            for i in 0..V {
                let b = f64::from(self.img[i]);
                if b > 0.02 && self.rng.rnd() < b * 0.03 {
                    self.push(OV + i, hash(i as u32, 7), 0);
                }
            }
        }
        self.phase();
        self.na = 0;
        self.spawn_tick = 0;
        let homeo = if self.settings.spawn {
            (1.0 - self.ca as f64 / self.settings.spawn_target.min(self.cap) as f64).max(0.0)
        } else {
            0.0
        };
        let sorted = self.ca as f64 > self.cap as f64 * 0.2;
        if sorted {
            self.sort_by_age();
        }
        let ip = f64::from(self.settings.interaction) / 100.0;
        let tk = self.tick.wrapping_add(1);
        if ip > 0.0 {
            for i in 0..self.ca {
                let n = self.cur.neuron[i] as usize;
                if n >= P {
                    continue;
                }
                if self.ft[n] == tk {
                    if self.rng.rnd() < ip {
                        let other = self.fi[n] as usize;
                        let m = self.rng.r32();
                        let a = self.cur.payload[i];
                        let b = self.cur.payload[other];
                        self.cur.payload[i] = merge(a, b, m);
                        self.cur.payload[other] = merge(b, a, m);
                    }
                } else {
                    self.ft[n] = tk;
                    self.fi[n] = i as u32;
                }
            }
        }
        for oi in 0..self.ca {
            let i = if sorted { self.ord[oi] as usize } else { oi };
            let n = self.cur.neuron[i] as usize;
            let f = self.cur.stamp[i];
            let p = self.cur.payload[i];
            let bt = self.cur.birth[i];
            if n == OS {
                if self.pt >= MINT {
                    if self.pt - self.last_sub >= RF {
                        self.sub_hit = true;
                        if self.learn && f != 0 && self.sc < 8 {
                            let o = self.sc * L;
                            self.spc[o..o + L].copy_from_slice(&self.cur.trace[i * L..i * L + L]);
                            self.sc += 1;
                        }
                    }
                } else if self.learn && f != 0 && self.ec < 20 {
                    // Submitting before the network had time to think weakens the path taken.
                    self.ec += 1;
                    for z in 0..L {
                        let j = self.cur.trace[i * L + z];
                        if j >= 0 {
                            let j = j as usize;
                            self.w[j] = (f64::from(self.w[j]) - 0.03).max(0.05) as f32;
                        }
                    }
                }
                continue;
            }
            if n >= OD {
                self.dr_t += 1;
                if self.learn && f == self.stamp && self.dn < crate::lab::DM {
                    let o = self.dn * L;
                    self.dp[o..o + L].copy_from_slice(&self.cur.trace[i * L..i * L + L]);
                    self.dpp[self.dn] = p;
                    self.dn += 1;
                }
                continue;
            }
            if (OO..OV).contains(&n) {
                let q = n - OO;
                self.acc[q] += 1.0;
                if f == self.stamp {
                    if self.first_hit < 0 {
                        self.first_hit = self.pt;
                    }
                    self.acc_s[q] += 1.0;
                    if self.learn {
                        self.credit(i, q, p);
                    }
                }
                continue;
            }
            if n < P {
                self.act[n] = 255;
            }
            if f != 0 {
                self.tr_buf
                    .copy_from_slice(&self.cur.trace[i * L..i * L + L]);
            }
            if self.settings.spawn && n < P && f == self.stamp {
                let lt = self.nt[n];
                if lt > 0
                    && self.nf[n] == self.stamp
                    && tk.wrapping_sub(lt) <= self.settings.spawn_window
                    && homeo > 0.0
                    && self.rng.rnd() < self.settings.spawn_rate * homeo
                {
                    let m = self.rng.r32();
                    let cp = merge(p, self.np[n], m);
                    let before = self.na;
                    self.fwd(n, f, cp, self.tick, self.cur.energy[i]);
                    if self.na > before {
                        self.spawn_tot += 1;
                        self.spawn_tick += 1;
                    }
                }
                self.nt[n] = tk;
                self.np[n] = p;
                self.nf[n] = self.stamp;
            }
            self.fwd(n, f, p, bt, self.cur.energy[i]);
        }
        if self.na >= self.cap {
            self.cap_hits += 1;
        }
        std::mem::swap(&mut self.cur, &mut self.next);
        self.ca = self.na;
        self.prune();
        for a in &mut self.acc {
            *a = (f64::from(*a) * 0.93) as f32;
        }
        self.tick = self.tick.wrapping_add(1);
    }

    fn regrow(&mut self, j: usize) {
        let n = j / K;
        let k = j % K;
        let mut t = None;
        if self.rgi > 0 && self.rng.rnd() < 0.5 {
            let m = self.rg[self.rng.ri(self.rgi.min(256))] as usize;
            let dx = px(m) - px(n);
            let dy = py(m) - py(n);
            let d = dx.hypot(dy);
            let d = if d == 0.0 { 1.0 } else { d };
            let reach = if n < P {
                if k >= 4 { 40.0 } else { 4.0 }
            } else {
                30.0
            };
            let s = if self.settings.unlimited_reach {
                1.0
            } else {
                (reach / d).min(1.0)
            };
            let x = js_round(px(n) + dx * s + (self.rng.rnd() - 0.5) * 2.0);
            let y = js_round(py(n) + dy * s + (self.rng.rnd() - 0.5) * 2.0);
            if in_grid(x, y) && (y * GW as i64 + x) as usize != n {
                t = Some((y * GW as i64 + x) as i32);
            }
        }
        let t = match t {
            Some(t) => t,
            None => self.new_target(n, k),
        };
        self.tgt[j] = t;
        self.w[j] = 0.2;
        self.tag[j] = self.rng.r32();
        self.cons[j] = 0.0;
    }

    /// Slowly weakens 1000 random links (used ones faster); weak links are rewired.
    fn prune(&mut self) {
        for _ in 0..1000 {
            let j = self.rng.ri((P + TI) * K);
            if self.tgt[j] == OS as i32 || self.mlk[j] > 0.5 {
                continue;
            }
            let c = f64::from(self.cons[j]);
            if c > 0.0 {
                self.cons[j] = (c * 0.9995) as f32;
            }
            let gf = 1.0 - 0.97 * f64::from(self.gk[j]);
            if self.gk[j] > 0.0 {
                self.gk[j] = (f64::from(self.gk[j]) * 0.99) as f32;
            }
            let dec = if self.usd[j] == 1 {
                self.usd[j] = 0;
                0.002
            } else {
                0.0002
            };
            self.w[j] = (f64::from(self.w[j]) - dec * (1.0 - c) * gf) as f32;
            if self.w[j] <= 0.02 {
                self.gk[j] = 0.0;
                self.regrow(j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_round_rounds_halves_up() {
        assert_eq!(js_round(2.5), 3);
        assert_eq!(js_round(-2.5), -2);
        assert_eq!(js_round(-2.6), -3);
    }

    #[test]
    fn positions_cover_every_block() {
        assert!((px(OS) - SX as f64).abs() < f64::EPSILON);
        assert!((px(OO) - OX0 as f64).abs() < f64::EPSILON);
        assert!((py(OO + 799) - (OY0 + 39) as f64).abs() < f64::EPSILON);
        assert!((py(OD + 41) - (GH + 12) as f64).abs() < f64::EPSILON);
    }
}
