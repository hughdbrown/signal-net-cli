//! Random numbers and hashing, bit-for-bit compatible with the browser lab.
//!
//! The JavaScript keeps its generator state in a 32-bit integer and does its hashing with a mix of
//! floating-point and 32-bit integer arithmetic. These functions reproduce that exactly, so the same
//! seed gives the same network in both versions.

/// The xorshift32 generator behind every random decision in the simulation.
#[derive(Debug, Clone)]
pub struct Rng {
    seed: u32,
}

impl Rng {
    /// Creates a generator. A zero seed never changes, so callers should pass a non-zero one.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// The current generator state (what a model file stores as `seed`).
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Replaces the generator state.
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
    }

    /// The next raw 32-bit state; `r32()` in the JavaScript.
    pub fn r32(&mut self) -> u32 {
        let mut s = self.seed;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.seed = s;
        s
    }

    /// A uniform number in `[0, 1)`; `rnd()` in the JavaScript.
    pub fn rnd(&mut self) -> f64 {
        f64::from(self.r32()) / 4_294_967_296.0
    }

    /// A uniform integer in `[0, n)`; `ri(n)` in the JavaScript. Returns 0 when `n` is 0.
    pub fn ri(&mut self, n: usize) -> usize {
        (self.rnd() * n as f64) as usize
    }

    /// Picks one element of a non-empty slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let i = self.ri(items.len());
        &items[i]
    }
}

/// `ToInt32` of a non-negative, integer-valued product below 2^64, as JavaScript's `^` applies it.
fn js_u32(x: f64) -> u32 {
    (x as u64) as u32
}

/// The lab's integer hash. The first multiplications happen in floating point, as in JavaScript.
#[must_use]
pub fn hash(a: u32, b: u32) -> u32 {
    let mut h = js_u32(f64::from(a) * 2_654_435_761.0) ^ js_u32(f64::from(b) * 40_503.0);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_519);
    h ^ (h >> 13)
}

/// The first UTF-16 unit of a character (`c.charCodeAt(0)` for a one-character JavaScript string).
#[must_use]
pub fn first_unit(c: char) -> u16 {
    let mut buf = [0u16; 2];
    c.encode_utf16(&mut buf)[0]
}

/// Iterates the code points of UTF-16 text and yields each one's first unit, keeping lone surrogates.
pub fn first_units(units: &[u16]) -> impl Iterator<Item = u16> + '_ {
    char::decode_utf16(units.iter().copied()).map(|r| match r {
        Ok(c) => first_unit(c),
        Err(e) => e.unpaired_surrogate(),
    })
}

/// String hash over code points; `hs(s)` in the JavaScript.
#[must_use]
pub fn hs(s: &str) -> u32 {
    s.chars().fold(0, |x, c| hash(x, u32::from(first_unit(c))))
}

/// [`hs`] over UTF-16 text.
#[must_use]
pub fn hs_units(units: &[u16]) -> u32 {
    first_units(units).fold(0, |x, u| hash(x, u32::from(u)))
}

/// The payload a stimulus signal carries for character code `c` at slot `s`.
#[must_use]
pub fn pay(c: u32, s: u32) -> u32 {
    (hash(c, 0) & 0xffff_0000) | (hash(s, 1) & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_javascript() {
        // Values printed by tools/js-golden.mjs from the page's own functions.
        assert_eq!(hash(0, 0), 0);
        assert_eq!(hash(1, 2), 136_417_595);
        assert_eq!(hash(4_000_000_000, 77), 3_632_329_790);
        assert_eq!(hs("hello world"), 1_007_580_946);
        assert_eq!(pay(104, 3), 3_567_849_811);
    }

    #[test]
    fn ri_stays_in_range() {
        let mut r = Rng::new(123_456_789);
        for _ in 0..10_000 {
            assert!(r.ri(7) < 7);
        }
        assert_eq!(r.ri(0), 0);
    }

    #[test]
    fn utf16_hash_agrees_with_str_hash() {
        let s = "héllo 😀";
        let units: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(hs(s), hs_units(&units));
    }
}
